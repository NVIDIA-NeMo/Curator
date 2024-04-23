import argparse

from dask_kubernetes.operator.kubecluster import KubeCluster, make_cluster_spec


def create_cluster(
    name: str,
    n_workers: int,
    n_gpus_per_worker: int,
    n_cpus_per_worker: int,
    image: str,
    image_pull_secret: str,
    pvcs: dict[str, str],
):
    dask_worker_command = "dask-worker"
    if n_gpus_per_worker and n_gpus_per_worker > 0:
        dask_worker_command = "dask-cuda-worker"

    custom_cluster_spec = make_cluster_spec(
        name=name,
        worker_command=dask_worker_command,
        n_workers=n_workers,
        image=image,
    )
    scheduler_spec = custom_cluster_spec["spec"]["scheduler"]["spec"]
    worker_spec = custom_cluster_spec["spec"]["worker"]["spec"]
    if image_pull_secret:
        scheduler_spec["imagePullSecrets"] = [{"name": image_pull_secret}]
        worker_spec["imagePullSecrets"] = [{"name": image_pull_secret}]

    obj_vols = []
    obj_vol_mounts = []
    for pvc_name, mount_path in pvcs.items():
        obj_vols.append(
            {
                "name": pvc_name,
                "persistentVolumeClaim": {
                    "claimName": pvc_name,
                },
            }
        )
        obj_vol_mounts.append(
            {
                "name": pvc_name,
                "mountPath": mount_path,
            }
        )

    scheduler_spec["volumes"] = obj_vols
    for ctr in scheduler_spec["containers"]:
        ctr["volumeMounts"] = obj_vol_mounts

    worker_spec["volumes"] = obj_vols
    for ctr in worker_spec["containers"]:
        ctr["volumeMounts"] = obj_vol_mounts
        # Resources are added to only the worker, since the scheduler doesn't need GPUs
        if n_gpus_per_worker or n_cpus_per_worker:
            if not ctr["resources"]:
                ctr["resources"] = {"limits": {}}
            if n_gpus_per_worker:
                ctr["resources"]["limits"]["nvidia.com/gpu"] = str(n_gpus_per_worker)
            if n_cpus_per_worker:
                ctr["resources"]["limits"]["cpu"] = str(n_cpus_per_worker)

    cluster = KubeCluster(
        custom_cluster_spec=custom_cluster_spec, shutdown_on_close=False
    )
    print(f"{cluster = }")


if __name__ == "__main__":

    def parse_pvcs(specs: str) -> dict[str, str]:
        name_to_path = {}
        for pvc in specs.split(","):
            # Can be empty
            if not pvc:
                continue
            name, _, path = pvc.partition(":")
            name_to_path[name] = path
        return name_to_path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="rapids-dask",
        help="The name of the DaskCluster which you would be able to inspect via `kubectl describe daskcluster <name>`.",
    )
    parser.add_argument(
        "-w", "--n_workers", type=int, default=2, help="Number of workers"
    )
    parser.add_argument(
        "-g",
        "--n_gpus_per_worker",
        type=int,
        default=None,
        help="Number of GPUs per worker. If not specified, the Dask Cluster defaults to a CPU cluster.",
    )
    parser.add_argument(
        "-c",
        "--n_cpus_per_worker",
        type=int,
        default=None,
        help="Number of CPUs per worker. Provide this flag if you want to limit your CPU resources and K8s will throttle the workers to make sure this limit is satisfied.",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="nvcr.io/nvidia/nemo:24.03.framework",
        help="The image used for the Dask Cluster scheduler and workers.",
    )
    parser.add_argument(
        "-s",
        "--image_pull_secret",
        type=str,
        default=None,
        help="If --image is from a private registry, specify the appropriate pull secret you created to allow these to be pulled.",
    )
    parser.add_argument(
        "-p",
        "--pvcs",
        type=parse_pvcs,
        default="",
        help="Comma sep PVC specificiation of $pvc_name_1:$mount_path_1,$pvc_name_2:$mount_path_2. Example: foo:/foo,bar:/bar mounts pvcs named foo and bar to /foo and /bar respectively.",
    )

    args = parser.parse_args()

    create_cluster(
        **vars(args),
    )
