.. _data-curator-nemo-run:

======================================
NeMo Curator with NeMo Run
======================================
-----------------------------------------
NeMo Run
-----------------------------------------

The NeMo Run is a general purpose tool for configuring and executing Python functions and scripts acrosss various computing environments.
It is used across the NeMo Framework for managing machine learning experiments.
One of the key features of the NeMo Run is the ability to run code locally or on platforms like SLURM with minimal changes.

-----------------------------------------
Usage
-----------------------------------------

We recommend getting slightly familiar with NeMo Run before jumping into this. The documentation can be found here.

Let's walk through the example usage for how you can launch a slurm job using `examples/launch_slurm.py <https://github.com/NVIDIA/NeMo-Curator/blob/main/examples/nemo_run/launch_slurm.py>`_.

.. code-block:: python


    import nemo_run as run
    from nemo_run.core.execution import SlurmExecutor

    from nemo_curator.nemo_run import SlurmJobConfig

    @run.factory
    def nemo_curator_slurm_executor() -> SlurmExecutor:
        """
        Configure the following function with the details of your SLURM cluster
        """
        return SlurmExecutor(
            job_name_prefix="nemo-curator",
            account="my-account",
            nodes=2,
            exclusive=True,
            time="04:00:00",
            container_image="nvcr.io/nvidia/nemo:dev",
            container_mounts=["/path/on/machine:/path/in/container"],
        )

First, we need to define a factory that can produce a ``SlurmExecutor``.
This exectuor is where you define all your cluster parameters. Note: NeMo Run only supports running on SLURM clusters with `Pyxis <https://github.com/NVIDIA/pyxis>`_ right now.
After this, there is the main function

.. code-block:: python

    # Path to NeMo-Curator/examples/slurm/container_entrypoint.sh on the SLURM cluster
    container_entrypoint = "/cluster/path/slurm/container_entrypoint.sh"
    # The NeMo Curator command to run
    curator_command = "text_cleaning --input-data-dir=/path/to/data --output-clean-dir=/path/to/output"
    curator_job = SlurmJobConfig(
        job_dir="/home/user/jobs",
        container_entrypoint=container_entrypoint,
        script_command=curator_command,
    )

First, we need to specify the path to `examples/slurm/container-entrypoint.sh <https://github.com/NVIDIA/NeMo-Curator/blob/main/examples/slurm/container-entrypoint.sh>`_ on the cluster.
This shell script is responsible for setting up the Dask cluster on Slurm and will be the main script run.
Therefore, we need to define the path to it.

Second, we need to establish the NeMo Curator script we want to run.
This can be a command line utility like ``text_cleaning`` we have above, or it can be your own custom script ran with ``python path/to/script.py``


Finally, we combine all of these into a ``SlurmJobConfig``. This config has many options for configuring the Dask cluster.
We'll highlight a couple of important ones:

* ``device="cpu"`` determines the type of Dask cluster to initialize. If you are using GPU modules, please set this equal to ``"gpu"``.
* ``interface="etho0"`` specifies the network interface to use for communication within the Dask cluster. It will likely be different for your Slurm cluster, so please modify as needed. You can determine what interfaces are available by running the following function on your cluster.

  .. code-block:: python

    from nemo_curator import get_network_interfaces

    print(get_network_interfaces())

.. code-block:: python

    executor = run.resolve(SlurmExecutor, "nemo_curator_slurm_executor")
    with run.Experiment("example_nemo_curator_exp", executor=executor) as exp:
        exp.add(curator_job.to_script(), tail_logs=True)
        exp.run(detach=False)

After configuring the job, we can finally run it.
First, we use the run to resolve our custom factory.
Next, we use it to begin an experiment named "example_nemo_curator_exp" running on our Slurm exectuor.

``exp.add(curator_job.to_script(), tail_logs=True)`` adds the NeMo Curator script to be part of the experiment.
It converts the ``SlurmJobConfig`` to a ``run.Script``.
This ``curator_job.to_script()`` has two important parameters.
* ``add_scheduler_file=True``
* ``add_device=True``

Both of these modify the command specified in ``curator_command``.
Setting both to ``True`` (the default) transforms the original command from:

.. code-block:: bash

    # Original command
    text_cleaning \
        --input-data-dir=/path/to/data \
        --output-clean-dir=/path/to/output

to:

.. code-block:: bash

    # Modified commmand
    text_cleaning \
        --input-data-dir=/path/to/data \
        --output-clean-dir=/path/to/output \
        --scheduler-file=/path/to/scheduler/file \
        --device="cpu"


As you can see, ``add_scheduler_file=True`` causes ``--scheduler-file=/path/to/scheduer/file`` to be appended to the command, and ``add_device=True`` causes ``--device="cpu"`` (or whatever the device is set to) to be appended.
``/path/to/scheduer/file`` is determined by ``SlurmJobConfig``, and ``device`` is what the user specified in the ``device`` parameter previously.

The scheduler file argument is necessary to connect to the Dask cluster on Slurm.
All NeMo Curator scripts accept both arguments, so the default is to automatically add them.
If your script is configured differently, feel free to turn these off.

The final line ``exp.run(detach=False)`` starts the experiment on the Slurm cluster.