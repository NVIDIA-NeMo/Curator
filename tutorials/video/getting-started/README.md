# Getting Started with Video Curation

The Python scripts in this directory contain examples for how to run video curation workflows with NeMo Curator. In particular:

- `video_read_example.py` contains a simple pipeline which reads video data
- `video_split_clip_example.py` contains a more complex pipeline which reads video data, applies a splitting algorithm, transcodes the clips, applies a motion filter, scores and filters by aesthetic content, generates embeddings, generates captions, generates previews, and saves the results
