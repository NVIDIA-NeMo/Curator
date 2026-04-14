# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP viewer for ``omnicorpus_blur_statistics.py`` ``--export-dir`` output.

Multi-page UI (separate routes) over one process: home, blur, CLIP, image/word ratio, sample
galleries, and **CLIP sample detail** (``/samples/clip/item?bin=…&i=…``) with image + pair text.
Shared ``/style.css`` and ``/app.js``. Serves ``distribution.json`` at ``/api/distribution.json``
and export files under ``/media/…``.

Run::

    python tutorials/multimodal/annotation/omnicorpus_distribution_viewer.py /path/to/export-dir

Open http://127.0.0.1:8765/ (defaults; use ``--host`` / ``--port``).
"""

from __future__ import annotations

import argparse
import html
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _safe_media_relpath(raw: str) -> str | None:
    """Return a safe path relative to export root, or ``None``."""
    if not raw or raw.startswith("/"):
        return None
    parts = Path(raw).parts
    if ".." in parts:
        return None
    return str(Path(*parts)) if parts else None


def _guess_type(path: Path) -> str:
    t, _ = mimetypes.guess_type(str(path))
    return t or "application/octet-stream"


def _nav_html(active: str) -> str:
    nav_active = "samples_clip" if active == "samples_clip_item" else active
    items: list[tuple[str, str, str]] = [
        ("/", "home", "Home"),
        ("/blur", "blur", "Blur"),
        ("/clip", "clip", "CLIP"),
        ("/ratio", "ratio", "Image / words"),
        ("/samples/blur", "samples_blur", "Samples · blur"),
        ("/samples/clip", "samples_clip", "Samples · CLIP"),
    ]
    links: list[str] = []
    for href, page_id, label in items:
        cls = " active" if page_id == nav_active else ""
        links.append(f'<a class="nav-item{cls}" href="{href}">{html.escape(label)}</a>')
    return f'<header class="site-header"><h1 class="site-title">OmniCorpus stats</h1><nav class="topnav">{" ".join(links)}</nav></header>'


def _html_shell(*, title: str, page_id: str) -> str:
    t = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{t}</title>
  <link rel="stylesheet" href="/style.css"/>
</head>
<body data-page="{html.escape(page_id, quote=True)}">
{_nav_html(page_id)}
<main id="main" class="main"><p class="sub">Loading…</p></main>
<script src="/app.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function () {{
  omnistatRun(document.body.getAttribute("data-page") || "home");
}});
</script>
</body>
</html>"""


STYLE_CSS = """
:root { --bg: #0f1419; --card: #1a2332; --text: #e7ecf3; --muted: #8b9cb3; --accent: #5b9fd8; --bar: #3d7ab8; }
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; padding: 0 1.25rem 1.5rem; background: var(--bg); color: var(--text); line-height: 1.45; }
.site-header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.75rem 1.25rem; padding: 1rem 0; border-bottom: 1px solid #2a3545; margin-bottom: 1rem; }
.site-title { font-size: 1.15rem; font-weight: 600; margin: 0; }
.topnav { display: flex; flex-wrap: wrap; gap: 0.35rem 0.5rem; }
.nav-item { color: var(--muted); text-decoration: none; font-size: 0.88rem; padding: 0.25rem 0.5rem; border-radius: 4px; }
.nav-item:hover { color: var(--text); background: #243044; }
.nav-item.active { color: #0a0e14; background: var(--accent); }
.main { max-width: 1100px; }
h2 { font-size: 1.05rem; margin: 1.25rem 0 0.5rem; color: var(--accent); border-bottom: 1px solid #2a3545; padding-bottom: 0.25rem; }
.sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 1rem; }
.err { background: #3d2020; color: #f5b5b5; padding: 0.75rem 1rem; border-radius: 6px; margin: 1rem 0; }
.card { background: var(--card); border-radius: 8px; padding: 1rem 1.1rem; margin-bottom: 1rem; border: 1px solid #243044; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 0.75rem; margin: 0.5rem 0; }
a.card-link { display: block; text-decoration: none; color: inherit; border: 1px solid #33465c; border-radius: 8px; padding: 0.85rem 1rem; background: var(--card); }
a.card-link:hover { border-color: var(--accent); }
a.card-link strong { color: var(--accent); }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #2a3545; }
th { color: var(--muted); font-weight: 500; }
tr:last-child td { border-bottom: none; }
.hbar-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.2rem 0; font-size: 0.82rem; }
.hbar-label { width: 10.5rem; flex-shrink: 0; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hbar-track { flex: 1; height: 18px; background: #243044; border-radius: 4px; overflow: hidden; }
.hbar-fill { height: 100%; background: linear-gradient(90deg, var(--bar), #6ab0e8); border-radius: 4px; min-width: 2px; }
.hbar-meta { width: 5rem; text-align: right; color: var(--muted); flex-shrink: 0; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 0.25rem 1rem; font-size: 0.88rem; margin: 0.5rem 0; }
.kv dt { color: var(--muted); }
.tabs { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.5rem 0 0.75rem; }
.tabs button { background: #243044; border: 1px solid #33465c; color: var(--text); padding: 0.35rem 0.65rem; border-radius: 6px; cursor: pointer; font-size: 0.82rem; }
.tabs button.on { background: var(--accent); border-color: #7eb8e8; color: #0a0e14; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 0.65rem; margin-top: 0.5rem; }
.tile { background: #121a26; border-radius: 6px; overflow: hidden; border: 1px solid #2a3545; }
.tile img { width: 100%; height: 140px; object-fit: cover; display: block; background: #0a0e14; }
.tile .cap { padding: 0.35rem 0.45rem; font-size: 0.72rem; color: var(--muted); word-break: break-all; }
.tile a { color: var(--accent); font-size: 0.72rem; }
.tile .tile-detail { display: block; margin-top: 0.25rem; font-size: 0.72rem; }
.clip-detail-img { max-width: min(100%, 720px); height: auto; display: block; border-radius: 6px; margin-top: 0.5rem; }
.pair-block { margin: 1rem 0; padding-top: 0.75rem; border-top: 1px solid #2a3545; }
.pair-block:first-of-type { border-top: none; padding-top: 0; }
.pair-meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.35rem; }
.pair-text { font-size: 0.9rem; white-space: pre-wrap; word-break: break-word; }
"""


APP_JS = r"""
(function (global) {
  function esc(s) {
    if (s == null) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function hbarBlock(title, rows) {
    if (!rows || !rows.length) return "";
    var maxC = Math.max.apply(null, [1].concat(rows.map(function (r) { return r.count; })));
    var h = "<h2>" + esc(title) + "</h2><div class='card'>";
    rows.forEach(function (r) {
      var pct = (r.count / maxC * 100).toFixed(1);
      h += "<div class='hbar-row'><span class='hbar-label' title='" + esc(r.bin) + "'>" + esc(r.bin) + "</span>";
      h += "<div class='hbar-track'><div class='hbar-fill' style='width:" + pct + "%'></div></div>";
      h += "<span class='hbar-meta'>" + r.count + " · " + (r.pct != null ? Number(r.pct).toFixed(2) : "") + "%</span></div>";
    });
    h += "</div>";
    return h;
  }
  function kvBlock(title, obj) {
    if (!obj || typeof obj !== "object") return "";
    var h = "<h2>" + esc(title) + "</h2><div class='card'><dl class='kv'>";
    Object.keys(obj).forEach(function (k) {
      var v = obj[k];
      h += "<dt>" + esc(k) + "</dt><dd>" + esc(typeof v === "object" ? JSON.stringify(v) : v) + "</dd>";
    });
    h += "</dl></div>";
    return h;
  }
  function mediaPath(rel) {
    if (!rel) return "";
    return "/media/" + String(rel).split("/").filter(Boolean).map(encodeURIComponent).join("/");
  }
  function gallerySection(modality, samples) {
    if (!samples || typeof samples !== "object") return "<p class='sub'>No sample manifest for " + esc(modality) + ".</p>";
    var bins = Object.keys(samples).sort();
    if (!bins.length) return "<p class='sub'>No bins.</p>";
    var h = "<p class='sub'>Bins load images from <code>/media/…</code>.</p>";
    h += "<div class='tabs' id='tabs-" + modality + "'>";
    bins.forEach(function (b, i) {
      h += "<button type='button' class='" + (i === 0 ? "on" : "") + "' data-bin='" + esc(b) + "'>" + esc(b) + "</button>";
    });
    h += "</div><div id='gal-" + modality + "'></div>";
    return h;
  }
  function renderGallery(modality, samples, activeBin) {
    var host = document.getElementById("gal-" + modality);
    if (!host) return;
    var entries = samples[activeBin] || [];
    var h = "<div class='gallery'>";
    entries.forEach(function (e, idx) {
      var rel = e.saved_relative;
      var pair = e.pair_json_relative;
      var src = mediaPath(rel);
      h += "<div class='tile'>";
      if (src) h += "<a href='" + src + "' target='_blank' rel='noopener'><img loading='lazy' src='" + src + "' alt=''/></a>";
      else h += "<div class='cap'>(no image bytes)</div>";
      h += "<div class='cap'>sid " + esc(e.sample_id) + " · pos " + esc(e.position) + "</div>";
      if (pair) h += "<div class='cap'><a href='" + mediaPath(pair) + "' target='_blank' rel='noopener'>pair.json</a></div>";
      if (modality === "clip") {
        var detailHref = "/samples/clip/item?bin=" + encodeURIComponent(activeBin) + "&i=" + idx;
        h += "<a class='tile-detail' href='" + detailHref + "'>Image + pair text →</a>";
      }
      h += "</div>";
    });
    h += "</div>";
    host.innerHTML = h;
  }
  function renderPairPayload(pj) {
    var img = pj.image || {};
    var pairs = pj.pairs || [];
    var h = "";
    if (img.clip_max != null) h += "<p class='sub'>image.clip_max: " + esc(String(img.clip_max)) + "</p>";
    pairs.forEach(function (p) {
      h += "<article class='pair-block'><div class='pair-meta'>text_position " + esc(p.text_position);
      h += " · clip_similarity " + esc(p.clip_similarity == null ? "—" : String(p.clip_similarity)) + "</div>";
      h += "<pre class='pair-text'>" + esc(p.text || "") + "</pre></article>";
    });
    if (!pairs.length) h += "<p class='sub'>No text rows in pair JSON.</p>";
    return h;
  }
  function runClipSampleItemPage(d, main) {
    var back = "<p><a class='nav-item' style='display:inline-block;padding:0.35rem 0.6rem' href='/samples/clip'>← CLIP samples</a></p>";
    var q = new URLSearchParams(window.location.search);
    var bin = q.get("bin");
    var i = parseInt(q.get("i") || "-1", 10);
    if (!bin || i < 0 || !d.clip_samples) {
      main.innerHTML = back + "<div class='err'>Need query <code>?bin=…&amp;i=…</code> (bin label and index in that bin’s manifest). Open from <a href='/samples/clip'>CLIP samples</a>.</div>";
      return;
    }
    var list = d.clip_samples[bin];
    if (!list || !list[i]) {
      main.innerHTML = back + "<div class='err'>No sample for this bin/index.</div>";
      return;
    }
    var e = list[i];
    var imgUrl = mediaPath(e.saved_relative);
    var h = back;
    h += "<section class='card'><h2>Image</h2>";
    h += "<p class='sub'>bin <strong>" + esc(bin) + "</strong> · index " + i + " · sample_id " + esc(e.sample_id) + " · position " + esc(e.position) + "</p>";
    if (e.clip_max != null) h += "<p class='sub'>clip_max: " + esc(String(e.clip_max)) + "</p>";
    if (imgUrl) h += "<img class='clip-detail-img' src='" + imgUrl + "' alt='CLIP sample'/>";
    else h += "<p class='sub'>(no image bytes)</p>";
    h += "</section><section class='card'><h2>Pair text</h2><div id='clip-pair-body'><p class='sub'>Loading…</p></div></section>";
    main.innerHTML = h;
    var body = document.getElementById("clip-pair-body");
    if (!body) return;
    var pr = e.pair_json_relative;
    if (!pr) {
      body.innerHTML = "<p class='sub'>No pair JSON path in manifest.</p>";
      return;
    }
    fetch(mediaPath(pr)).then(function (r) {
      if (!r.ok) throw new Error(r.status + " " + r.statusText);
      return r.json();
    }).then(function (pj) {
      body.innerHTML = renderPairPayload(pj);
    }).catch(function (err) {
      body.innerHTML = "<div class='err'>Could not load pair JSON: " + esc(err.message) + "</div>";
    });
  }
  function wireTabs(modality, samples) {
    var bar = document.getElementById("tabs-" + modality);
    if (!bar) return;
    var buttons = bar.querySelectorAll("button");
    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        buttons.forEach(function (b) { b.classList.remove("on"); });
        btn.classList.add("on");
        renderGallery(modality, samples, btn.getAttribute("data-bin"));
      });
    });
    if (buttons.length) renderGallery(modality, samples, buttons[0].getAttribute("data-bin"));
  }
  function homePage(d) {
    var h = "<p class='sub'>API: <code>/api/distribution.json</code></p>";
    h += "<section class='card'><strong>image_rows</strong>: " + esc(d.image_rows) + "</section>";
    h += "<h2>Sections</h2><div class='cards'>";
    h += "<a class='card-link' href='/blur'><strong>Blur</strong><br/><span class='sub'>Histogram &amp; metadata</span></a>";
    h += "<a class='card-link' href='/clip'><strong>CLIP</strong><br/><span class='sub'>Per image &amp; by sample_id</span></a>";
    h += "<a class='card-link' href='/ratio'><strong>Image / words</strong><br/><span class='sub'>Per-sample ratio</span></a>";
    h += "<a class='card-link' href='/samples/blur'><strong>Samples · blur</strong><br/><span class='sub'>Thumbnails by bin</span></a>";
    h += "<a class='card-link' href='/samples/clip'><strong>Samples · CLIP</strong><br/><span class='sub'>Thumbnails; each tile links to image + pair text</span></a>";
    h += "</div>";
    return h;
  }
  function blurPage(d) {
    if (!d.blur) return "<p class='sub'>No blur section in distribution.json.</p>";
    var o = {
      sharpness: d.blur.sharpness,
      bin_width: d.blur.bin_width,
      n_bins: d.blur.n_bins,
      scored_images: d.blur.scored_images,
      missing_sharpness_images: d.blur.missing_sharpness_images,
    };
    return kvBlock("Blur metadata", o) + hbarBlock("Blur histogram", d.blur.rows || []);
  }
  function clipPage(d) {
    if (!d.clip) return "<p class='sub'>No clip section in distribution.json.</p>";
    var h = kvBlock("CLIP metadata", {
      clip_scores: d.clip.clip_scores,
      statistic: d.clip.statistic,
      scored_images: d.clip.scored_images,
      missing_clip_score_images: d.clip.missing_clip_score_images,
    });
    h += hbarBlock("CLIP histogram (per image)", d.clip.rows || []);
    var bs = d.clip.by_sample_id;
    if (bs) {
      h += kvBlock("CLIP by sample_id", {
        unique_samples_with_image_rows: bs.unique_samples_with_image_rows,
        scored_samples: bs.scored_samples,
        missing_clip_score_samples: bs.missing_clip_score_samples,
      });
      if (bs.max && bs.max.rows) h += hbarBlock("CLIP by sample_id — max", bs.max.rows);
      if (bs.min && bs.min.rows) h += hbarBlock("CLIP by sample_id — min", bs.min.rows);
    }
    return h;
  }
  function ratioPage(d) {
    if (!d.sample_image_word_ratio) return "<p class='sub'>No sample_image_word_ratio in distribution.json.</p>";
    return kvBlock("Image / words per sample", d.sample_image_word_ratio)
      + hbarBlock("Image/words ratio histogram", d.sample_image_word_ratio.ratio_rows || []);
  }
  global.omnistatRun = function (pageId) {
    var main = document.getElementById("main");
    if (!main) return;
    fetch("/api/distribution.json").then(function (r) {
      if (!r.ok) throw new Error(r.status + " " + r.statusText);
      return r.json();
    }).then(function (d) {
      if (pageId === "samples_clip_item") {
        runClipSampleItemPage(d, main);
        return;
      }
      var html = "";
      if (pageId === "home") html = homePage(d);
      else if (pageId === "blur") html = blurPage(d);
      else if (pageId === "clip") html = clipPage(d);
      else if (pageId === "ratio") html = ratioPage(d);
      else if (pageId === "samples_blur") {
        html = "<h2>Blur samples</h2>" + gallerySection("blur", d.blur_samples);
      } else if (pageId === "samples_clip") {
        html = "<h2>CLIP samples</h2><p class='sub'>Use <strong>Image + pair text →</strong> on a tile for a full page with text.</p>" + gallerySection("clip", d.clip_samples);
      } else html = homePage(d);
      main.innerHTML = html;
      if (pageId === "samples_blur" && d.blur_samples) wireTabs("blur", d.blur_samples);
      if (pageId === "samples_clip" && d.clip_samples) wireTabs("clip", d.clip_samples);
    }).catch(function (e) {
      main.innerHTML = "<div class='err'>Failed to load distribution: " + esc(e.message) + "</div>";
    });
  };
})(typeof window !== "undefined" ? window : this);
"""


def _handler_class(export_root: Path):
    root = export_root.resolve()

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(parsed.path)
            try:
                if path in ("/", "/index.html"):
                    self._send_html(_html_shell(title="OmniCorpus stats — Home", page_id="home"))
                elif path == "/blur":
                    self._send_html(_html_shell(title="OmniCorpus stats — Blur", page_id="blur"))
                elif path == "/clip":
                    self._send_html(_html_shell(title="OmniCorpus stats — CLIP", page_id="clip"))
                elif path == "/ratio":
                    self._send_html(_html_shell(title="OmniCorpus stats — Image / words", page_id="ratio"))
                elif path == "/samples/blur":
                    self._send_html(_html_shell(title="OmniCorpus stats — Blur samples", page_id="samples_blur"))
                elif path == "/samples/clip":
                    self._send_html(_html_shell(title="OmniCorpus stats — CLIP samples", page_id="samples_clip"))
                elif path == "/samples/clip/item":
                    self._send_html(
                        _html_shell(title="OmniCorpus stats — CLIP sample (image + text)", page_id="samples_clip_item")
                    )
                elif path == "/style.css":
                    self._send_text(STYLE_CSS.lstrip(), "text/css; charset=utf-8")
                elif path == "/app.js":
                    self._send_text(APP_JS.lstrip(), "application/javascript; charset=utf-8")
                elif path == "/api/distribution.json":
                    self._send_json_file(root / "distribution.json")
                elif path.startswith("/media/"):
                    rel = _safe_media_relpath(path[len("/media/") :])
                    if rel is None:
                        self.send_error(400, "Bad path")
                        return
                    fp = (root / rel).resolve()
                    try:
                        fp.relative_to(root)
                    except ValueError:
                        self.send_error(400, "Path escapes export root")
                        return
                    if not fp.is_file():
                        self.send_error(404, "Not found")
                        return
                    data = fp.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", _guess_type(fp))
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_error(404, "Not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError as e:
                self.send_error(500, str(e))

        def _send_html(self, body: str) -> None:
            self._send_text(body, "text/html; charset=utf-8")

        def _send_text(self, body: str, content_type: str) -> None:
            b = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _send_json_file(self, fp: Path) -> None:
            if not fp.is_file():
                self.send_error(404, "distribution.json not found")
                return
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _H


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Web UI for omnicorpus_blur_statistics.py --export-dir output (distribution.json + samples).",
    )
    parser.add_argument(
        "export_dir",
        type=str,
        help="Directory containing distribution.json (and optional blur/, clip/ sample trees)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()
    root = Path(args.export_dir)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    if not (root / "distribution.json").is_file():
        raise SystemExit(f"Missing distribution.json under: {root.resolve()}")

    handler = _handler_class(root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {root.resolve()} at http://{args.host}:{args.port}/")
    print("Pages: /  /blur  /clip  /ratio  /samples/blur  /samples/clip  /samples/clip/item?bin=…&i=…")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
