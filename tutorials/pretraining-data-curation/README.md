# RedPajama-Data-v2 Datasets Curation for LLM Pretraining

This tutorial demonstrates the usage of NeMo Curator to curate the RedPajama-Data-v2 dataset for LLM pretraining in a distributed environment.

## RedPajama-Data-v2
RedPajama-V2 (RPV2) is an open dataset for training large language models. The dataset includes over 100B text documents coming from 84 CommonCrawl snapshots and processed using the CCNet pipeline. In this tutorial, we will be perform data curation on two raw snapshots from RPV2 for demonstration purposes.

## Getting Started
This tutorial is designed to run in multi-node environment due to the pre-training dataset scale. To start the tutorial, run the slurm script `start-distributed-notebook.sh` in this directory which will start the Jupyter notebook that demonstrates the step by step walkthrough of the end to end curation pipeline. To access the Jupyter notebook running on the scheduler node from your local machine, you can establish an SSH tunnel by running the following command:

`ssh -L <local_port>:localhost:8888 <user>@<scheduler_address>`
