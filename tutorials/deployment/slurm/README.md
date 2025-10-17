The ray-sbatch-job.sh script is an example script that can be adapted to be used for single and multi-node slurm deployments in clusters that support container images in allocations.

It is recommneded to update the following variables/options before running in your own environment:
- `CONTAINER_MOUNTS` - Specify the mounts needed for the job. If no mounts are needed remove the `--container-mounts` flag from the `srun` commands.
- `IMAGE` - Update to use the latest Curator Image or any Image of choice
- `RUN_COMMAND` - Point to a script/python file that executes the main curation workflow.
- `SBATCH DIRECTIVES` - Set the relevant SBATCH directives in the file or pass as flags when submitting the job.

All of the options above can be modified in the script or set as environment variables that override the defaults in the script. For example:
```bash
RUN_COMMAND="python curation_script.py" IMAGE="my/image" CONTAINER_MOUNTS="/path/to/dataset:/data-dir" sbatch --nodes=2 -J=my-curation-job -A=my-account ray-batch-job.sh
```

For slurm environments that do not support or use containers, the script can be modified to call `module load` and source a venv for every `srun` command instead of using a container.
