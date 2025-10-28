# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Wikipedia HTML dumps to DocLayNet-style annotated images with random styling."""

import atexit
import base64
import io
import logging
import os
import json
import random
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice

import click
import tqdm
import webdataset
import zstandard as zstd
from PIL import Image
from playwright.sync_api import sync_playwright

from latex_augment.fonts import list_latin_fonts, list_simplified_chinese_fonts
from latex_augment.convert.html2markdown import html2markdown

EXTRACT_DOCLAYNET_JS = open(f"{os.path.dirname(__file__)}/extract_doclaynet.js").read()
WIKIPEDIA_CSS = open(f"{os.path.dirname(__file__)}/wikipedia.css").read()


def preprocess_html(html: str, *, css: str, title: str = None):
    """Preprocess HTML to remove unwanted elements and make it more suitable for OCR."""
    html = html.replace("</head>", f"<style>{css}</style></head>")
    script = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // hide stuff that is harmful for OCR
        document.querySelectorAll([
            '[role="navigation"]',
            '[role="note"]', 
            '.ambox',     // metadata with repeated phrases
            '.autocollapse',
            '.catlinks',  // looks like a table but is not
            '.infobox',   // <table> used for layout, not as an actual table
            '.metadata',  // repeated phrases ("This page was last edited on" etc.)
            '.navbar',
            '.navbox',
            '.mw-cite-backlink',  // superscript carets etc that are not visible in HTML
            '.mw-editsection',       // [Edit] links
            '.mw-editsection-like',  // [Edit] links
            '.noprint',   // phrases like "edit this page on Wikipedia" etc.
            '.sidebar',   // <table> used for layout, not as an actual table
            'figure',     // we have no images in the Wikipedia dump
            'table table' // nested tables used for layout, not as an actual table
        ].join(', ')).forEach(element => {
            element.style.display = 'none';
        });

        // hide broken images
        document.querySelectorAll('img').forEach(img => {
            if (!img.complete || img.naturalWidth === 0) {
                img.style.display = 'none';
            }
        });

        // hide outlier tables
        document.querySelectorAll('table').forEach(table => {
            const rows = table.querySelectorAll('tr');
            if (rows.length <= 1 || rows.length >= 50) {
                table.style.display = 'none';
            } else {
                const maxCols = Math.max(...Array.from(rows).map(row => {
                    const cells = row.querySelectorAll('td, th');
                    return Array.from(cells).reduce((sum, cell) => {
                        return sum + (parseInt(cell.getAttribute('colspan')) || 1);
                    }, 0);
                }));
                const cells = table.querySelectorAll('td, th');
                const spanCells = Array.from(cells).reduce((sum, cell) => {
                    return sum + (cell.hasAttribute('colspan') || cell.hasAttribute('rowspan'));
                }, 0);
                const styleCells = Array.from(cells).reduce((sum, cell) => {
                    return sum + (cell.hasAttribute('style') || cell.hasAttribute('class'));
                }, 0);
                if (maxCols <= 1 || maxCols >= 50 || spanCells >= 50 || styleCells >= 50) {
                    table.style.display = 'none';
                }
                //console.log(`table ${rows.length}x${maxCols} cells ${cells.length} span ${spanCells} style ${styleCells}`);
            }
        });

        // remove scrollbar
        document.body.style.overflow = 'hidden';
    """
    if title is not None:
        script += """
            const title = document.createElement('h1');
            title.textContent = "%s";
            document.body.insertBefore(title, document.body.firstChild);
        """ % title.replace('"', '\\"')
    script += """
    });
    </script>
    """
    html = html.replace("</head>", f"{script}</head>")
    return html


TLS = threading.local()


def dump_html(
    html: str, *, image_width: int, image_height: int, scrollpos: float = 0.0, debug: bool = False
):
    """Dump HTML to image and content with bounding boxes."""
    global TLS
    if getattr(TLS, "playwright", None) is None:
        TLS.playwright = sync_playwright().start()
        atexit.register(TLS.playwright.stop)
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(tmpdir + "/index.html", "w") as f:
            f.write(html)
        # try and prevent dbus errors like:
        # Failed to connect to the bus: Failed to connect to socket /run/dbus/system_bus_socket: No such file or directory
        browser = TLS.playwright.chromium.launch(
            env={
                "DBUS_SESSION_BUS_ADDRESS": "/dev/null",
            },
            # headless=False,
        )
        try:
            page = browser.new_page(
                viewport={"width": image_width, "height": image_height}
            )
            page.goto(f"file://{tmpdir}/index.html")
            page.on("console", lambda msg: logging.error(f"JavaScript: {msg.text}"))
            content = page.evaluate(EXTRACT_DOCLAYNET_JS, {"scrollpos": scrollpos, "debug": debug})
            page.screenshot(path=f"{tmpdir}/screenshot.png")
            img = Image.open(f"{tmpdir}/screenshot.png")
            return img, content
        finally:
            browser.close()


def dump_debug_html(img, ann, reference_url: str):
    """Create debug HTML with image and annotation boxes with hover information."""
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            .container {{
                position: relative;
                display: inline-block;
            }}
            .box {{
                position: absolute;
                border: 2px solid;
                pointer-events: none;
            }}
            .tooltip {{
                position: absolute;
                background: white;
                border: 1px solid black;
                padding: 5px;
                display: none;
                overflow: auto;
                z-index: 100;
                max-width: 80%;
            }}
            .box-container:hover .tooltip {{
                display: block;
            }}
            .box-container {{
                position: absolute;
                z-index: 10;
            }}
        </style>
    </head>
    <body>
        <a href="{reference_url}" target="_blank">{reference_url}</a><hr/>
        <div class="container">
            <img src="data:image/png;base64,{img_base64}" />
    """

    colors = {
        "Text": "blue",
        "Title": "red",
        "List-item": "green",
        "Table": "purple",
        "Figure": "orange",
        "Header": "teal",
        "Footer": "brown",
    }

    def escape_html(text):
        return text.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    for i, box in enumerate(ann):
        category = box["category_id"]
        content = box["content"]
        left, top, right, bottom = box["bbox"]
        color = colors.get(category, "gray")
        width = right - left
        height = bottom - top

        html += f"""
            <div class="box-container" style="left: {left}px; top: {top}px; width: {width}px; height: {height}px;">
                <div class="box" style="border-color: {color}; width: 100%; height: 100%;"></div>
                <div class="tooltip">
                    <strong>{escape_html(category)}</strong> ({left}, {top}, {right}, {bottom})<br>
                    {escape_html(content)}
                </div>
            </div>
        """

    html += """
        </div>
    </body>
    </html>
    """

    return html


def sample_style(rng: random.Random, latin_fonts: list[str], chinese_fonts: list[str]):
    if rng.random() < 0.7:
        background = "inherit"
    elif rng.random() < 0.5:
        background = f"oklch({rng.uniform(0.6, 1.0)} {rng.uniform(0.0, 0.3)} {rng.uniform(0, 360)});"
    else:
        background = f"linear-gradient({rng.uniform(0, 360)}deg, oklch({rng.uniform(0.6, 1.0)} {rng.uniform(0.0, 0.3)} {rng.uniform(0, 360)}), oklch({rng.uniform(0.6, 1.0)} {rng.uniform(0.0, 0.3)} {rng.uniform(0, 360)}))"
    if rng.random() < 0.5:
        color = "inherit"
    else:
        color = f"oklch({rng.uniform(0.0, 0.4)} {rng.uniform(0.0, 0.4)} {rng.uniform(0, 360)})"
    if rng.random() < 0.8:
        link_color = "inherit"
    else:
        link_color = f"oklch({rng.uniform(0.0, 0.4)} {rng.uniform(0.0, 0.4)} {rng.uniform(0, 360)})"
    font_family_en = rng.choice(latin_fonts)
    font_family_zh = rng.choice(chinese_fonts)
    font_size = rng.uniform(0.95, 1.3) if rng.random() < 0.5 else 1
    font_weight = rng.uniform(100, 900)
    header_font_family_en = (
        rng.choice(latin_fonts) if rng.random() < 0.5 else font_family_en
    )
    header_font_family_zh = (
        rng.choice(chinese_fonts) if rng.random() < 0.5 else font_family_zh
    )
    alt_font_family_en = (
        rng.choice(latin_fonts) if rng.random() < 0.5 else font_family_en
    )
    alt_font_family_zh = (
        rng.choice(chinese_fonts) if rng.random() < 0.5 else font_family_zh
    )
    header_font_size = font_size * 2 ** rng.uniform(-0.5, 0.5)
    header_font_weight = rng.uniform(font_weight, 900)
    if rng.random() < 0.2:
        header_border = f"border-bottom: {rng.uniform(0, 3)}px solid rgba(150, 150, 150, {rng.uniform(0.0, 1.0)});"
    else:
        header_border = "border-bottom: none;"
    if rng.random() < 0.1:
        header_text_shadow = f"text-shadow: {rng.uniform(-5, 5)}px {rng.uniform(-5, 5)}px {rng.uniform(5, 10)}px oklch({rng.uniform(0.7, 1.0)} {rng.uniform(0.0, 0.4)} {rng.uniform(0, 360)});"
    else:
        header_text_shadow = ""
    return f"""
    body {{
        letter-spacing: {rng.uniform(-0.05, 0.2)}rem;
        word-spacing: {rng.uniform(-0.05, 0.5)}rem;
        line-height: {rng.uniform(1, 2)};
        margin-left: {rng.uniform(0, 5)}rem;
        margin-right: {rng.uniform(0, 5)}rem;
        background: {background};
        color: {color};
        font-family: "{font_family_en}";
        font-weight: {font_weight};
        font-size: {font_size}rem;
        &:lang(en) {{
            font-family: "{font_family_en}";
        }}
        &:lang(zh) {{
            font-family: "{font_family_zh}";
        }}
    }}
    a {{
        color: {link_color} !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        margin-top: {rng.uniform(0.5, 4)}rem;
        margin-bottom: {rng.uniform(0.5, 4)}rem;
        font-family: "{header_font_family_en}";
        font-weight: {header_font_weight};
        text-align: {rng.choice(["left", "left", "center"])};
        letter-spacing: {rng.uniform(-0.05, 0.5)}rem;
        {header_border}
        {header_text_shadow}
        &:lang(en) {{
            font-family: "{header_font_family_en}" !important;
        }}
        &:lang(zh) {{
            font-family: "{header_font_family_zh}" !important;
        }}
    }}
    h1 {{
        font-size: {2.35 * header_font_size}rem !important;
    }}
    h2 {{
        font-size: {1.76 * header_font_size}rem !important;
    }}
    h3 {{
        font-size: {1.51 * header_font_size}rem !important;
    }}
    h4 {{
        font-size: {1.33 * header_font_size}rem !important;
    }}
    h5 {{
        font-size: {1.17 * header_font_size}rem !important;
    }}
    h6 {{
        font-size: {1.08 * header_font_size}rem !important;
    }}
    p {{
        margin-top: {rng.uniform(0, 1)}rem;
        margin-bottom: {rng.uniform(0, 1)}rem;
    }}
    section:nth-child(odd) {{
        font-family: "{alt_font_family_en}";
        &:lang(en) {{
            font-family: "{alt_font_family_en}" !important;
        }}
        &:lang(zh) {{
            font-family: "{alt_font_family_zh}" !important;
        }}
    }}
    table, figure {{
        background: {rng.choice(["inherit", "white", background])};
    }}
    """


def generate_sample(line: str, *, debug: bool = False, seed: int = 0, latin_fonts: list[str] = None, chinese_fonts: list[str] = None):
    title = wiki_id = None
    try:
        page = json.loads(line)
        html = page["article_body"]["html"]
        title = page["name"]
        wiki_id = page["identifier"]
        rng = random.Random(seed ^ wiki_id)
        scrollpos = rng.random()
        # image_width = int(1024 * 0.35 ** rng.random())  # 360..1024
        image_height = int(1280 * 0.5 ** rng.random())  # 640..1280
        image_width = 1024
        # image_height = 1280
        css = WIKIPEDIA_CSS + sample_style(rng, latin_fonts=latin_fonts, chinese_fonts=chinese_fonts)
        html = preprocess_html(html, css=css, title=title)
        if debug:
            with open("page.html", "w") as f:
                f.write(html.replace("document.body.style.overflow", "//"))
        img, html_blocks = dump_html(
            html,
            image_width=image_width,
            image_height=image_height,
            scrollpos=scrollpos,
            debug=debug,
        )
        if debug:
            img.save("page.png")
            with open("page.content.json", "w") as f:
                json.dump(html_blocks, f, indent=2, ensure_ascii=False)
        try:
            blocks = [
                {
                    "bbox": b["bbox"],
                    "category_id": b["category_id"],
                    "content": html2markdown(b["content"], debug=debug),
                }
                for b in html_blocks
            ]
        except Exception as e:
            logging.error(f"Markdown conversion error on page {title!r} ({wiki_id}): {e.__class__.__name__}: {e}")
            return None
        if not blocks:
            logging.warning(f"Ignoring empty page {title!r} ({wiki_id})")
            return None
        label = {
            "metadata": {
                "page_width_px": image_width,
                "page_height_px": image_height,
                "title": title,
                "wiki_id": wiki_id,
                "url": page["url"],
            },
            "ann": blocks,
        }
        if debug:
            with open("page.debug.html", "w") as f:
                f.write(dump_debug_html(img, blocks, page["url"]))
            with open("page.doclaynet.json", "w") as f:
                json.dump(blocks, f, indent=2, ensure_ascii=False)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        sample = {
            "__key__": f"{wiki_id:07d}_{int(scrollpos * 1000000):06d}",
            "png": png,
            "doclaynet.json": label,
        }
        return sample
    except Exception as e:
        if debug:
            raise
        logging.exception(f"Ignoring error on page {title!r} ({wiki_id}):")
        return None


def iterlines(fd):
    """Iterate over lines from a binary file descriptor."""
    buf = b""
    while True:
        val = fd.read(4096)
        if not val:
            break
        buf += val
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line
    if buf:
        yield buf


def batched(iterable, n):
    """Create batches of size n from an iterable."""
    # batched("ABCDEFG", 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path())
@click.option("--node-id", type=int, default=0)
@click.option("--num-nodes", type=int, default=1)
@click.option("--samples-per-shard", type=int, default=10000)
@click.option("--seed", type=int, default=0)
@click.option("--debug", is_flag=True)
def main(
    path: str,
    outdir: str,
    node_id: int,
    num_nodes: int,
    samples_per_shard: int,
    seed: int,
    debug: bool,
):
    # path = "enwiki-500k.jsonl.zst"
    # outdir = "enwiki_v1/temp"
    # path = "zhwiki-500k.jsonl.zst"
    # outdir = "zhwiki_v1/temp"
    # seed = 1337

    # lines = open("zhwiki-1k.jsonl", "rb").readlines()
    # page_id = random.randint(0, len(lines) - 1)
    # print(page_id)
    # generate_sample(lines[page_id], debug=True, seed=seed)
    # FIXME

    if debug:
        with zstd.open(path, "rb") as f:
            for line in iterlines(f):
                sample = generate_sample(line, seed=seed, latin_fonts=["sans-serif"], chinese_fonts=["sans-serif"], debug=True)
                print(sample["doclaynet.json"])
        return

    latin_fonts = sorted(set([font.family for font in list_latin_fonts()]))
    chinese_fonts = sorted(set([font.family for font in list_simplified_chinese_fonts()]))

    assert 0 <= node_id < num_nodes, "--node-id is out of range"
    os.makedirs(outdir, exist_ok=True)
    # limit number of Chromium workers to avoid OOM
    with ProcessPoolExecutor(max_workers=48) as executor:

        def write_shard(path, inputs, size):
            with webdataset.TarWriter(f"{outdir}/{path}") as writer:
                gen = partial(generate_sample, seed=seed, latin_fonts=latin_fonts, chinese_fonts=chinese_fonts)
                res = executor.map(gen, inputs)
                for sample in tqdm.tqdm(res, total=size, desc=path):
                    if sample is not None:
                        writer.write(sample)

        with zstd.open(path, "rb") as f:
            lines = islice(iterlines(f), node_id, None, num_nodes)
            for shard_id, shard in enumerate(batched(lines, samples_per_shard)):
                write_shard(
                    f"shard_{node_id:03d}_{shard_id:03d}.tar", shard, samples_per_shard
                )


if __name__ == "__main__":
    main()
