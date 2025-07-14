# Fuzzy Deduplication Stages

If you are limited by dataset size, compute resources, and/or time, we recommend running each stage of fuzzy deduplication as its own job on Slurm.
This directory contains all the scripts needed to do that.
By default, we skip the false positive check for fuzzy deduplication in favor of a more performant pipeline, see more information [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/gpudeduplication.html#id4).

Note that this pipeline is meant to replacing running the `5_fuzzy_deduplication.py` script.
