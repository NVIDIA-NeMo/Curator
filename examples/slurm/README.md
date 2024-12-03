# Dask with Slurm

This directory provides an example Slurm script pipeline.
This pipeline has a script `start-slurm.sh` that provides configuration options similar to what `get_client` provides.
Every Slurm cluster is different, so make sure you understand how your Slurm cluster works so the scripts can be easily adapted.
`start-slurm.sh` calls `containter-entrypoint.sh`, which sets up a Dask scheduler and workers across the cluster.

Our Python examples are designed to work such that they can be run locally on their own, or easily substituted into the `start-slurm.sh` script to run on multiple nodes.
You can adapt your scripts easily too by simply following the pattern of adding `get_client` with `add_distributed_args`.
