# Random OCR synthetic data generation

Pipeline:

1. Download Webster's dictionary
2. Generate English word list
3. Generate OCR document dataset

## Download Webster's dictionary

Available at [Project Gutenberg](https://www.gutenberg.org/ebooks/29765)

## Generate English word list

Run `webster.sh`.

```sh
bash webster.sh
```

## Generate OCR document dataset

Generate webdataset separately for each script with `gen_randomocr_wds.py`.

```sh
for script in ascii english latin chinese japanese korean; do
docker run --rm \
  -v $PWD:/workspace \
  -v path/to/data:/data \
  -e PYTHONPATH=/workspace/src \
  latex-augment python3 scripts/randomocr/gen_randomocr_wds.py ${script} /data/randomocr_${script}"
done
```
