# nvpdftex - Custom LaTeX compiler to generate annotated ground truth

This folder contains launcher scripts that are used together with nvtexlive and nvtexpy to
compile latex sources and obtain rendered page images, annotated with text boxes.
Each text box will have a position in the image, a semantic class, and content given as markdown formatted text.

These scripts depend on

* nvtexlive
    * A customized fork of [TeX Live](https://www.tug.org/texlive/)
* nvtexpy
    * A Python-based engine that nvtexlive imports like a plugin
    * It contains hooks and rules and is the core of the nvpdftex pipeline

# Usage

This section is work in progress.