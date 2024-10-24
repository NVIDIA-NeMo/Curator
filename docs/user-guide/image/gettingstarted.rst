
.. _data-curator-image-getting-started:

================
Get Started
================

NeMo Curator provides many tools for curating large scale text-image pair datasets for training generative image models.

---------------------
Install NeMo Curator
---------------------
To install the image curation modules of NeMo Curator, ensure you meet the following requirements:

* Python 3.10
* Ubuntu 22.04/20.04
* NVIDIA GPU
  * Volta™ or higher (compute capability 7.0+)
  * CUDA 12 (or above)

Note: While some of the text-based NeMo Curator modules do not require a GPU, all image curation modules require a GPU.

You can get NeMo Curator in 3 ways.

1. PyPi
2. Source
3. NeMo Framework Container

#####################
PyPi
#####################
NeMo Curator's PyPi page can be found `here <https://pypi.org/project/nemo-curator/>`_.

.. code-block:: bash

    pip install cython
    pip install nemo-curator[image]

#####################
Source
#####################
NeMo Curator's GitHub can be found `here <https://github.com/NVIDIA/NeMo-Curator>`_.

.. code-block:: bash

    git clone https://github.com/NVIDIA/NeMo-Curator.git
    pip install cython
    pip install ./NeMo-Curator[image]

############################
NeMo Framework Container
############################
NeMo Curator comes preinstalled in the NeMo Framework container. You can find a list of all the NeMo Framework container tags `here <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`_.

---------------------
Use NeMo Curator
---------------------

NeMo Curator can be run locally, or on a variety of compute platforms (Slurm, k8s, and more).

To get started using the image modules in NeMo Curator, we recommend you check out the following resources:

* `Image Curation Tutorial <https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image-curation/image-curation.ipynb>`_
* `API Reference <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/api/image/index.html>`_