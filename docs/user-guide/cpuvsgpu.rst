
.. _data-curator-cpuvsgpu:

======================================
CPU and GPU Modules with Dask
======================================

NeMo Curator provides GPU-accelerated modules alongside its CPU modules.
These modules are based on RAPIDS to enable scaling workflows to massive dataset sizes.
The remaining modules are CPU based and rely on Dask to scale to multi-node clusters.
When working with these different modules, it's important to understand how to properly set up your Dask cluster and how to manage where your dataset is stored in memory.

-----------------------------------------
Initializing the Dask Cluster
-----------------------------------------

NeMo Curator provides a simple function ``get_client`` that can be used to start a local Dask cluster or connect to an existing one.
All of the ``examples/`` use it to set up a Dask cluster.

.. code-block:: python

    from nemo_curator.utils.distributed_utils import get_client

    # Set up Dask client
    client = get_client(cluster_type="cpu")

    # Perform some computation...

``get_client`` takes a bunch of arguments that allow you to initialize or connect to an existing Dask cluster.

* ``cluster_type`` controls what type of Dask cluster to create. "cpu" will create a CPU based local Dask cluster, while "gpu" will create a GPU based local cluster.
  If "cpu" is specified, the number of processes started with the cluster can be specified with the ``n_workers`` argument.
  By default, this argument is set to ``os.cpu_count()``.
  If "gpu" is specified, one worker is started per GPU.
  It is possible to run entirely CPU-based workflows on a GPU cluster, though the process count (and therefore the number of parallel tasks) will be limited by the number of GPUs on your machine.

* ``scheduler_address`` and ``scheduler_file`` are used for connecting to an existing Dask cluster.
  Supplying one of these is essential if you are running a Dask cluster on Slurm or Kubernetes.
  All other arguments are ignored if either of these are passed, as the cluster configuration will be done when you create the schduler and works on your cluster.

* The remaining arguments can be modified `here <https://github.com/NVIDIA/NeMo-Curator/blob/main/nemo_curator/utils/distributed_utils.py>`_.

-----------------------------------------
CPU Modules
-----------------------------------------

As mentioned in the ``DocumentDataset`` documentation, the underlying storage format for datasets in NeMo Curator is just a Dask dataframe.
For the CPU modules, Dask uses pandas dataframes to hold dataframe partitions.
Most modules in NeMo Curator are CPU based.
Therefore, the default behavior for reading and writing datasets is to operate on them in CPU memory with a pandas backend.
The following two functions calls are equivalent.

.. code-block:: python

    books = DocumentDataset.read_json(files, add_filename=True)
    books = DocumentDataset.read_json(files, add_filename=True, backend="pandas")


-----------------------------------------
GPU Modules
-----------------------------------------

The following NeMo Curator modules are GPU based.

* Exact Deduplication
* Fuzzy Deduplication
* Semantic Deduplication
* Distributed Data Classification

  * Domain Classification (English and multilingual)
  * Quality Classification
  * AEGIS and Instruction Data Guard Safety Models
  * FineWeb Educational Content Classification
  * FineWeb Mixtral and FineWeb Nemotron-4 Educational Models
  * Content Type Classification
  * Prompt Task and Complexity Classification

GPU modules store the ``DocumentDataset`` using a ``cudf`` backend instead of a ``pandas`` one.
To read a dataset into GPU memory, one could use the following function call.

.. code-block:: python

    gpu_books = DocumentDataset.read_json(files, add_filename=True, backend="cudf")


Even if you start a GPU dask cluster, you can't operate on datasets that use a ``pandas`` backend.
The ``DocuemntDataset`` must either have been originally read in with a ``cudf`` backend, or it must be transferred during the script.

-----------------------------------------
Moving data between CPU and GPU
-----------------------------------------

The ``ToBackend`` module provides a way to move data between CPU memory and GPU memory by swapping between pandas and cuDF backends for your dataset.
To see how it works, take a look at this example.

.. code-block:: python

  from nemo_curator import Sequential, ToBackend, ScoreFilter, get_client
  from nemo_curator.datasets import DocumentDataset
  from nemo_curator.classifiers import DomainClassifier
  from nemo_curator.filters import RepeatingTopNGramsFilter, NonAlphaNumericFilter

  def main():
      client = get_client(cluster_type="gpu")

      dataset = DocumentDataset.read_json("books.jsonl")
      curation_pipeline = Sequential([
          ScoreFilter(RepeatingTopNGramsFilter(n=5)),
          ToBackend("cudf"),
          DomainClassifier(),
          ToBackend("pandas"),
          ScoreFilter(NonAlphaNumericFilter()),
      ])

      curated_dataset = curation_pipeline(dataset)

      curated_dataset.to_json("curated_books.jsonl")

  if __name__ == "__main__":
      main()

Let's highlight some of the important parts of this example.

* ``client = get_client(cluster_type="gpu")``: Creates a local Dask cluster with access to the GPUs. In order to use/swap to a cuDF dataframe backend, you need to make sure you are running on a GPU Dask cluster.
* ``dataset = DocumentDataset.read_json("books.jsonl")``: Reads in the dataset to a pandas (CPU) backend by default.
* ``curation_pipeline = ...``: Defines a curation pipeline consisting of a CPU filtering step, a GPU classifier step, and another CPU filtering step. The ``ToBackend("cudf")`` moves the dataset from CPU to GPU for the classifier, and the ``ToBackend("pandas")`` moves the dataset back to the CPU from the GPU for the last filter.
* ``curated_dataset.to_json("curated_books.jsonl")``: Writes the dataset directly to disk from the GPU. There is no need to transfer back to the CPU before writing to disk.

-----------------------------------------
Dask with Slurm
-----------------------------------------

We provide an example Slurm script pipeline in ``examples/slurm``.
This pipeline has a script ``start-slurm.sh`` that provides configuration options similar to what ``get_client`` provides.
Every Slurm cluster is different, so make sure you understand how your Slurm cluster works so the scripts can be easily adapted.
``start-slurm.sh`` calls ``containter-entrypoint.sh``, which sets up a Dask scheduler and workers across the cluster.

Our Python examples are designed to work such that they can be run locally on their own, or easily substituted into the ``start-slurm.sh`` script to run on multiple nodes.
You can adapt your scripts easily too by simply following the pattern of adding ``get_client`` with ``add_distributed_args``.

-----------------------------------------
Dask with K8s
-----------------------------------------

We also provide an example guide for how to get started with NeMo Curator on a Kubernetes cluster.

Please visit :ref:`curator_kubernetes` for more information.
