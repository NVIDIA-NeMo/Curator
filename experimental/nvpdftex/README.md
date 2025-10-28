# NVpdftex - Custom LaTeX compiler toolchain to generate annotated ground truth

NVpdftex is a toolchain that contains a modified version of the TeX and LaTeX engine.
It can be used to compile a document and render pages, while for each page obtaining

* Boxes with
    * Text content in markdown format (alternatively HTML)
    * A semantic class of the box
    * A PNG image of the rendered page
* Some metadata like page size in pixels

In contrast to other approaches, we directly hook into the core engine of TeX, avoiding detours through other formats like HTML and enabling us to obtain pixel-accurate locations of every glyph printed on the page.

## A bit of History

This toolchain has been used internally at NVIDIA to create large-scale OCR training data.
It was first publicly described in February 2025 in our [ÉCLAIR](https://arxiv.org/abs/2502.04223) paper, where we called the generated data the "arXiv-5M Dataset", later it was mentioned in the [blog post](https://huggingface.co/blog/nvidia/llama-nemotron-nano-vl) about our first Nano VL models.

In October 2025, we finally decided to publicly release the source code, so that everyone can generate their own datasets 🥳.

## The Augmentation and Translation Framework

In addition to the compilation tools, the toolchain also includes a LaTeX source code augmentation tool, that you can apply before the compilation process.

It can be used to transform LaTeX to LaTeX while changing the document appearance 🛠️ or changing the text content by translation 🗨️🌍.

Check out the augmentation tools 👉 [here](../latex-augment/).

## Installation and How to Use It

Required components:

* nvtexlive - The modified version of TeX Live which includes TeX and LaTeX
* nvtexpy - The engine which receives signals from the TeX compiler and translates those into boxes with labels and markdown content
* The launcher script in this repository (`process_latex.py`)
* Optionally: latex-augment - The augmentation tools

### Set up All Repos

```sh
# First clone this repository:
git clone -b experimental https://github.com/NVIDIA-NeMo/Curator.git nemo-curator

# Go to our base folder where we will check out the other repos
cd nemo-curator/experimental
# and let's give it a name
export EXPDIR=$PWD

# Check out the other repos
git clone git@github.com:NVIDIA/nvtexlive.git
git clone git@github.com:NVIDIA/nvtexpy.git
```

### Build the Container
```sh
cd $EXPDIR/nvtexlive
docker build -t nvtexlive:latest .
```

### Download some example LaTeX document
```sh
mkdir -p $EXPDIR/exampledoc && curl -L https://arxiv.org/src/1505.04597 | tar -xz -C $EXPDIR/exampledoc
```

### Launch the Container and Run nvpdftex Inside

```sh
# Launch the container
docker run --rm -it -v $EXPDIR:/expdir --net=host nvtexlive:latest bash

# Inside the container
cd /expdir

# nvtexpy must be installed for nvtexlive to find it
pip install ./nvtexpy

# Use the example script to launch the compilation
python nvpdftex/process_latex.py exampledoc/paper459.tex
```

When running `process_latex.py` you should see several processing stages and conversion of pages.
Out of 8 pages, 7 should succeed and one should fail (explanation below).

After that you will find the output files next to the `.tex` file like this:
```
nvpdftex_page_1.flat_layout.json
nvpdftex_page_1.pkl
nvpdftex_page_1.png
nvpdftex_page_1.txt
nvpdftex_page_2.flat_layout.json
nvpdftex_page_2.pkl
nvpdftex_page_2.png
nvpdftex_page_2.txt
etc.
```

* `.png`: An image of the page (generated from the PDF)
* `.flat_layout.json`: The bounding boxes with labels and content (see format below)
* `.txt`: A very detailed intermediate format (glyph-level) that we use to generate the flat layout (Likely you won't need it)
* `.pkl`: A pickled version of the intermediate representation. (You won't need it)

## Why Do Some Pages Fail?

TeX is made to produce visually appealing outputs, not to retain semantics. All the LaTeX packages on top of it add to its complexity and there are infinitely many ways to reach the same visual output.

NVtexpy contains a large set of rules to deal with this complexity and parse the latex modes, groups, boxes, and nodes to obtain an interpretable structure.

Those rules simply don't cover all cases, so feel free to add rules to nvtexpy to improve it!
In the example above, page 6 fails, because the tabular on the page doesn't have the same number of columns in every row, which works for compilation to a PDF but in general the semantics are not well defined.

## Notes on the Output Format

The output in the `.flat_layout.json` files will look like this:

```
{
    "metadata": {
        "coord_scale_factor": [
            6.334075991075136e-05,
            6.334075991075136e-05
        ],
        "page_width_px": 2550,
        "page_height_px": 3300
    },
    "ann": [
        {
            "category_id": "Page-header",
            "bbox": [
                1983,
                392,
                19,
                24
            ],
            "content": "5"
        },
        {
            "category_id": "Caption",
            "bbox": [
                593,
                493,
                18,
                16
            ],
            "content": "a"
        },
        ...
    ]
}
```

In `metadata` you can find the page size in pixels (according to the PNG) and the scale factors between pixel coordinates and internal TeX coordinates. You would only need those if you work with the intermediate format.

Every box in the `ann` list of boxes has a `category_id`, a bounding box (`bbox`) and the `content`.


* `category_id`
    * Can currently be one of
        * `Inline-Text`
        * `Text`
        * `Title`
        * `Section-header`
        * `List-item`
        * `Caption`
        * `Table`
        * `Picture`
        * `Formula`
        * `Code`
        * `Footnote`
        * `Page-header`
        * `Page-footer`
        * `Bibliography`
        * `TOC`
* `bbox`
    * Pixel coordinates as `(x, y, width, height)` with the origin being the top-left corner of the page
    * Note that the actual glyphs are not guaranteed to be 100% contained in the box (some pixels can be outside)
* `content`
    * Text content is in a custom hybrid format:
    * Normal and formatted text are in Markdown
    * Some special formatting is represented as HTML style tags:
        * `<sup></sup>` for superscripts
        * `<sub></sub>` for subscripts
        * `<del></del>` for strikethrough text
        * `<u></u>` for underlined text
        * `<o></o>` for overlined text
    * Inline math is formatted in LaTeX like `\(...\)`
    * Tables are formatted in LaTeX like `\begin{tabular}{...} ... \end{tabular}`
    * Special semantic markers
        * `<tbc>` at the end of a text block to indicate that the text is truncated
        * `<tbc>` at the beginning of a text block to indicate that the text is continued from the previous page or block
        * `<graphics>` for graphics (images)
