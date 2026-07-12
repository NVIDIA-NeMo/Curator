# Redacting Personally Identifiable Information with NVIDIA's GLiNER-PII Model

This tutorial demonstrates the use of NVIDIA's [GLiNER PII model](https://huggingface.co/nvidia/gliner-PII) in NeMo Curator. The GLiNER PII model detects and classifies a broad range of Personally Identifiable Information (PII) and Protected Health Information (PHI) in structured and unstructured text. It is non-generative and produces span-level entity annotations with confidence scores across 55+ categories.

This tutorial requires at least 1 NVIDIA GPU with:
  - Volta™ or higher (compute capability 7.0+)
  - CUDA 12.x

Before running this tutorial, see this [Installation Guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html#admin-installation) page for instructions on how to install NeMo Curator. Be sure to use an installation method which includes GPU dependencies.

Additionally, the tutorial uses the [GLiNER](https://github.com/urchade/GLiNER) library. It can be installed with:

```bash
uv pip install gliner
```

For reference, this tutorial was tested with `gliner==0.2.24`.

Since the `gliner_pii_redaction.ipynb` notebook uses the `GlinerPiiRedactor` class from `gliner_pii_redactor.py`, the user should `cd /path/to/Curator/tutorials/text/gliner-pii-redaction` to ensure that the module is importable.
