# NeMo Curator

NeMo Curator is a Python library that consists of a collection of scalable data-mining modules for curating natural language processing (NLP) data for training large language models (LLMs). The modules within NeMo Curator enable NLP researchers to mine high-quality text at scale from massive uncurated web corpora. For a demonstration of how each of the modules in NeMo Curator improves downstream performance, check out the [module ablation](#module-ablation).

NeMo Curator is built on [Dask](https://www.dask.org/) and [RAPIDS](https://developer.nvidia.com/rapids) to scale data curation and provide GPU acceleration. The Python interface provides easy methods to expand the functionality of your curation pipeline without worrying about how it will scale. More information can be found in the [usage section](#usage). There are many ways to integrate NeMo Curator in your pipeline. Check out the [installation instructions](#installation) for how to get started using it.

## Features
We currently support the following data-curation modules. For more details on each module, visit its documentation page in the [NeMo framework user guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html).
 - [Data download and text extraction](docs/user-guide/Download.rst)
   - Default implementations of download and extraction of Common Crawl, Wikipedia, and ArXiv data
   - Users can easily customize the download and extraction and extend to other datasets
 - [Language identification and separation](docs/user-guide/LanguageIdentificationUnicodeFormatting.rst)
   - Language identification with [fastText](https://fasttext.cc/docs/en/language-identification.html) and [pycld2](https://pypi.org/project/pycld2/)
 - [Text reformatting and cleaning](docs/user-guide/LanguageIdentificationUnicodeFormatting.rst)
   - Fix unicode decoding errors via [ftfy](https://ftfy.readthedocs.io/en/latest/)
 - [Quality filtering](docs/user-guide/QualityFiltering.rst)
   - Multilingual heuristic-based filtering
   - Classifier-based filtering via [fastText](https://fasttext.cc/)
 - [Document-level deduplication](docs/user-guide/GpuDeduplication.rst)
   - Both exact and fuzzy deduplication are accelerated using cuDF and Dask.
   - For fuzzy deduplication, our implementation follows the method described in [Microsoft Turing NLG 530B](https://arxiv.org/abs/2201.11990).
  - [Multilingual downstream-task decontamination](docs/user-guide/TaskDecontamination.rst)
    -  Our implementation follows the approach of [OpenAI GPT3](https://arxiv.org/pdf/2005.14165.pdf) and [Microsoft Turing NLG 530B](https://arxiv.org/abs/2201.11990)
  - [Distributed data classification](docs/user-guide/DistributedDataClassification.rst)
    - Multi-node multi-GPU classifier inference
    - Allows for sophisticated domain and quality classification
    - Flexible interface for extending to your own classifier network
  - [Personal identifiable information (PII) redaction](docs/user-guide/PersonalIdentifiableInformationIdentificationAndRemoval.rst)
    - Idenficiation tools for removing addresses, credit card numbers, social security numbers and more.

These modules are designed to be flexible and allow for reordering with few exceptions. The [NeMo Framework Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) includes prebuilt pipelines for you to start with and modify as needed.

## Learn More
- [Documentation](docs/)
- [Examples](examples/)
- [Module Ablation and Compute Performance](#module-ablation-and-compute-performance)

## Installation

NeMo Curator currently requires Python 3.10 and the GPU accelerated modules require CUDA 12 or above installed in order to be used.

NeMo Curator can be installed manually by cloning the repository and installing as follows -

For CPU only modules:
```
pip install .
```

For CPU + CUDA accelerated modules
```
pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"
```

### NeMo Framework Container

NeMo Curator is available in the [NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo). The NeMo Framework Container provides an end-to-end platform for development of custom generative AI models anywhere. The latest release of NeMo Curator comes preinstalled in the container.

## Usage

### Python Library

```Python
# Download your dataset
dataset = download_common_crawl("/datasets/common_crawl/", "2021-04", "2021-10", url_limit=10)
# Build your pipeline
curation_pipeline = Sequential([
  Modify(UnicodeReformatter()),
  ScoreFilter(WordCountFilter(min_words=80)),
  ScoreFilter(FastTextQualityFilter(model_path="model.bin")),
  TaskDecontamination([Winogrande(), Squad(), TriviaQA()])
])
# Curate your dataset
curated_dataset = curation_pipeline(dataset)
```

NeMo Curator provides a collection of robust python modules that can be chained together to construct your entire data curation pipeline. These modules can be run on your local machine or in a distributed compute environment like SLURM with no modifications. NeMo Curator provides simple base classes that you can inherit from to create your own filters, document modifiers, and other extensions without needing to worry about how they scale. The [examples](examples/) directory contains a bunch of scripts showcasing each of these modules. The data curation section of the [NeMo framework user guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html) provides in-depth documentation on how each of the modules work. If you need more information to modify the NeMo Curator for your usecase, the [implementation section](#implementation) provides a good starting point.

### Scripts

We provide CLI scripts to use as well in case those are more convienent. The scripts under `nemo_curator/scripts` map closely with each of the created python modules. Visit the [documentation](docs) for each of the python modules for more information about the scripts associated with it.


### NeMo Framework Launcher
[NeMo Megatron Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) is another way to interface with NeMo Curator. The launcher allows
for easy parameter and cluster configuration and will automatically generate the SLURM batch scripts that wrap around the CLI scripts required to run your pipeline.
Note: This is not the only way to run NeMo Curator on SLURM. There are example scripts in [`examples/slurm`](examples/slurm/) for running NeMo Curator on SLURM without the launcher.

## Module Ablation and Compute Performance

The modules within NeMo Curator were in large part designed to curate high-quality documents from Common Crawl snapshots and to be able to do so
in a scalable manner. In order to assess the quality of the Common Crawl documents curated by the modules in NeMo Curator, we performed a series
of ablation experiments in which we trained a 357M-parameter GPT-style model on the datasets resulting from the different stages of our data curation
pipeline implemented in NeMo Curator. The figure below demonstrates that the different data curation modules implemented within NeMo Curator
lead to improved model zero-shot downstream task performance.

<p align="center">
  <img src="./docs/user-guide/images/zeroshot_ablations.png" alt="drawing" width="700"/>
</p>

In terms of scalability and compute performance, using the RAPIDS + Dask fuzzy deduplication, we are able to deduplicate the 1.1 Trillion token Red Pajama dataset in 1.8 hours using 64 A100s.

Additionally, using the CPU-based modules the table below shows the time required and resulting data size reduction of each step of processing the [Common Crawl snapshot from November/December of 2020](https://commoncrawl.org/2020/12/nov-dec-2020-crawl-archive-now-available/) using 30 CPU nodes (with hardware similar to the `c5.24xlarge` [Amazon AWS C5 instance](https://aws.amazon.com/ec2/instance-types/c5/)):

<table>
  <thead>
    <tr>
      <th style="text-align:center">Dataset </th>
      <th colspan="2"> Download and text extraction</th>
      <th colspan="2">Text cleaning </th>
      <th colspan="2">Quality filtering</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td>Time  </td>
      <td> Output Size </td>
      <td>Time </td>
      <td> Output Size </td>
      <td>Time </td>
      <td> Output Size </td>
    </tr>
    <tr>
      <td>Common Crawl 2020-50</td>
      <td> 36 hrs</td>
      <td>2.8 TB</td>
      <td> 1 hr </td>
      <td> 2.8 TB </td>
      <td> 0.2 hr </td>
      <td> 0.52 TB </td>
    </tr>
  </tbody>
</table>

## Implementation

As mentioned above, the modules within NeMo Curator enable users to scale data-mining and NLP processing tasks to many nodes within a compute cluster.
The modules accomplish this using [Dask](https://www.dask.org/) with [cuDF](https://docs.rapids.ai/api/cudf/nightly/user_guide/10min/) (for the GPU-accelerated modules).
At the core of the NeMo Curator, `DocumentDataset` (the main dataset class) is just a simple wrapper around a Dask dataframe. Dask allows NeMo Curator to scale to arbitrary cluster sizes, and it supports a variety of distributed computing platforms. It supports reading and writing to different file formats, and it can balance these operations among nodes in the cluster. Importantly, Dask also supports the RAPIDS cuDF library for GPU-acclerated exact and fuzzy deduplication.
