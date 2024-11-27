# Bitext Cleaning

This tutorial demonstrates and highlights the bitext-specific functionalities within NeMo Curator's API to load and filter the English-German split of the [Multi-Target TED Talks](https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/) dataset. These files are based on [TED Talks](https://www.ted.com/).

The dataset contains a small training set of ~150k sentences, and ~2k dev and test sets. This tutorial only downloads and filters the training set.

## Walkthrough

This tutorial highlights several bitext-specific functionalities within NeMo Curator's API, including:
1. The ParallelDataset Class
2. Length Ratio Filtering via the LengthRatioFilter and the JointScoreFilter classes
3. Histogram-based Language ID Filtering via the HistogramFilter and the ParallelScoreFilter class.
4. (Optional and only on GPU) Model-based filtering using quality estimation model (`Unbabel/wmt20-comet-qe-da`) and the JointScoreFilter class.

## Usage

After installing the NeMo Curator package, you can simply run the following command:
```
python tutorials/bitext_cleaning/main.py
```
This will download the English-German training data, and run both length ratio filtering and histogram-based language ID filtering. The filtered data will be saved in the `data/` directory here.
