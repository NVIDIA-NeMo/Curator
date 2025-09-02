# Getting Started with Image Curation

The Python scripts in this directory contain examples for how to run typical image curation workflows with NeMo Curator. In particular:

- `image_curation_example.py` implements a pipeline to read images, generate their [CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) embeddings, score and filter the images by aesthetic and NSFW content, and save the results
- `image_dedup_example.py` implements a pipeline to read images, generate their [CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) embeddings, filter semantic duplicates (i.e., images which look similar in content to other images within the dataset), and save the results
- `helper.py` contains functions for downloading and saving image data used by the above scripts
