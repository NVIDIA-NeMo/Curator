---
description: "Deploy NeMo Curator on Kubernetes clusters using Dask Operator, GPU Operator, and PVC storage with complete setup and management guide"
categories: ["how-to-guides"]
tags: ["kubernetes", "dask-operator", "gpu-operator", "pvc-storage", "cluster-management", "deployment", "container"]
personas: ["admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "universal"
---

(admin-deployment-kubernetes)=
# Running NeMo Curator on Kubernetes

## Prerequisites

* Kubernetes cluster
    * [GPU operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html)
    * [Dask Operator](https://kubernetes.dask.org/en/latest/operator_installation.html)
* [kubectl](https://kubernetes.io/docs/tasks/tools): the Kubernetes Cluster CLI
    * Please reach out to your Kubernetes cluster admin for how to set up your `kubectl` KUBECONFIG
* [ReadWriteMany](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes) [StorageClass](https://kubernetes.io/docs/concepts/storage/storage-classes/) (set up by Kubernetes cluster admin)

---

## Storage

To run NeMo Curator, we need to set up storage to upload and store the input files and processed outputs.

Here's an example of how to create a dynamic PV from a StorageClass set up by your cluster admin. Replace `STORAGE_CLASS=<...>` with the name of your StorageClass.

This example requests `150Gi` of space. Adjust that number for your workloads and be aware that not all storage provisioners support volume resizing.

```bash
STORAGE_CLASS=<...>
PVC_NAME=nemo-workspace

kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ${STORAGE_CLASS}
  resources:
    requests:
      # Requesting enough storage for a few experiments
      storage: 150Gi
EOF
```

```{admonition} Note
The storage class must support `ReadWriteMany` because multiple Pods may need to access the PVC to concurrently read and write.
```

## Set Up PVC Busybox Helper Pod

Inspecting the PVC and copying to and from it's facilitated with a Busybox container. Some examples below assume you have this Pod running to copy to and from the PVC.

```bash
PVC_NAME=nemo-workspace
MOUNT_PATH=/nemo-workspace

kubectl create -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: nemo-workspace-busybox
spec:
  containers:
  - name: busybox
    image: busybox
    command: ["sleep", "infinity"]
    volumeMounts:
    - name: workspace
      mountPath: ${MOUNT_PATH}
  volumes:
  - name: workspace
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF
```

Feel free to delete this container if no longer needed, but it should use very little resources when idle.

```bash
kubectl delete pod nemo-workspace-busybox
```

## Set Up Docker Secrets

A Kubernetes Secret needs to be created on the Kubernetes cluster to authenticate with the NGC private registry. If not done already, get an NGC key from ngc.nvidia.com. Create a secret key on the Kubernetes cluster with (replace `<NGC KEY HERE>` with your NGC secret key. Note that if you have any special characters in your key you might need to wrap the key in single quotes (`'`) so it can be parsed correctly by Kubernetes):

```bash
kubectl create secret docker-registry ngc-registry --docker-server=nvcr.io --docker-username=\$oauthtoken --docker-password=<NGC KEY HERE>
```

## Set Up Python Environment

The environment to run the provided scripts in this example doesn't need the full `nemo_curator` package, so you can create a virtual environment with just the required packages as follows:

```bash
python3 -m venv venv
source venv/bin/activate

pip install 'dask_kubernetes>=2024.4.1'
```

```{seealso}
For details on NeMo Curator container environments and configurations, see [Container Environments](reference-infrastructure-container-environments).

**Configuration**: After setting up your cluster, see {doc}`Deployment Environment Configuration <../config/deployment-environments>` for Kubernetes-specific environment variables and settings.
```

## Upload Data to PVC

# Running NeMo Curator on Kubernetes

## Prerequisites

* Kubernetes cluster
    * [GPU operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html)
    * [Dask Operator](https://kubernetes.dask.org/en/latest/operator_installation.html)
* [kubectl](https://kubernetes.io/docs/tasks/tools): the Kubernetes Cluster CLI
    * Please reach out to your Kubernetes cluster admin for how to set up your `kubectl` KUBECONFIG
* [ReadWriteMany](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes) [StorageClass](https://kubernetes.io/docs/concepts/storage/storage-classes/) (set up by Kubernetes cluster admin)

---

## Storage

To run NeMo Curator, we need to set up storage to upload and store the input files and processed outputs.

Here's an example of how to create a dynamic PV from a StorageClass set up by your cluster admin. Replace `STORAGE_CLASS=<...>` with the name of your StorageClass.

This example requests `150Gi` of space. Adjust that number for your workloads and be aware that not all storage provisioners support volume resizing.

```bash
STORAGE_CLASS=<...>
PVC_NAME=nemo-workspace

kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ${STORAGE_CLASS}
  resources:
    requests:
      # Requesting enough storage for a few experiments
      storage: 150Gi
EOF
```

```{admonition} Note
The storage class must support `ReadWriteMany` because multiple Pods may need to access the PVC to concurrently read and write.
```

## Set Up PVC Busybox Helper Pod

Inspecting the PVC and copying to and from it's facilitated with a Busybox container. Some examples below assume you have this Pod running to copy to and from the PVC.

```bash
PVC_NAME=nemo-workspace
MOUNT_PATH=/nemo-workspace

kubectl create -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: nemo-workspace-busybox
spec:
  containers:
  - name: busybox
    image: busybox
    command: ["sleep", "infinity"]
    volumeMounts:
    - name: workspace
      mountPath: ${MOUNT_PATH}
  volumes:
  - name: workspace
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF
```

Feel free to delete this container if no longer needed, but it should use very little resources when idle.

```bash
kubectl delete pod nemo-workspace-busybox
```

## Set Up Docker Secrets

A Kubernetes Secret needs to be created on the Kubernetes cluster to authenticate with the NGC private registry. If not done already, get an NGC key from ngc.nvidia.com. Create a secret key on the Kubernetes cluster with (replace `<NGC KEY HERE>` with your NGC secret key. Note that if you have any special characters in your key you might need to wrap the key in single quotes (`'`) so it can be parsed correctly by Kubernetes):

```bash
kubectl create secret docker-registry ngc-registry --docker-server=nvcr.io --docker-username=\$oauthtoken --docker-password=<NGC KEY HERE>
```

## Set Up Python Environment

The environment to run the provided scripts in this example doesn't need the full `nemo_curator` package, so you can create a virtual environment with just the required packages as follows:

```bash
python3 -m venv venv
source venv/bin/activate

pip install 'dask_kubernetes>=2024.4.1'
```

```{seealso}
For details on NeMo Curator container environments and configurations, see [Container Environments](reference-infrastructure-container-environments).

**Configuration**: After setting up your cluster, see {doc}`Deployment Environment Configuration <../config/deployment-environments>` for Kubernetes-specific environment variables and settings.
```

## Upload Data to PVC

To copy into the `nemo-workspace` PVC, we'll do so with `kubectl exec`. You may also use `kubectl cp`, but `exec` has fewer surprises regarding compressed files:

```bash
# Replace <...> with a path on your local machine
LOCAL_WORKSPACE=<...>

# This copies $LOCAL_WORKSPACE/my_dataset to /my_dataset within the PVC.
# Change foobar to the directory or file you wish to upload.
( cd $LOCAL_WORKSPACE; tar cf - my_dataset | kubectl exec -i nemo-workspace-busybox -- tar xf - -C /nemo-workspace )
```

```{admonition} Note
See [Data Curator Download](text-load-data) for an example of how to download local data that can be uploaded to the PVC with the above instruction.

**Storage Configuration**: For cloud storage access (S3, Azure, GCS), see {doc}`Storage & Credentials Configuration <../config/storage-credentials>` to set up the necessary credentials and access patterns.
```

## Create a Dask Cluster

Use the `create_dask_cluster.py` to create a CPU or GPU Dask cluster.

:::{admonition} Note
If you're creating another Dask cluster with the same `--name <name>`, first delete it via:

```bash
kubectl delete daskcluster <name>
```
:::

```bash
# Creates a CPU Dask cluster with 1 worker
python create_dask_cluster.py \
    --name rapids-dask \
    --n_workers 1 \
    --image nvcr.io/nvidia/nemo-curator:latest \
    --image_pull_secret ngc-registry \
    --pvcs nemo-workspace:/nemo-workspace

#╭───────────────────── Creating KubeCluster 'rapids-dask' ─────────────────────╮
#│                                                                              │
#│   DaskCluster                                                      Running   │
#│   Scheduler Pod                                                    Running   │
#│   Scheduler Service                                                Created   │
#│   Default Worker Group                                             Created   │
#│                                                                              │
#│ ⠧ Getting dashboard URL                                                      │
#╰──────────────────────────────────────────────────────────────────────────────╯
#cluster = KubeCluster(rapids-dask, 'tcp://localhost:61757', workers=2, threads=510, memory=3.94 TiB)

# Creates a GPU Dask cluster with 2 workers with 1 GPU each
python create_dask_cluster.py \
    --name rapids-dask \
    --n_workers 2 \
    --n_gpus_per_worker 1 \
    --image nvcr.io/nvidia/nemo-curator:latest \
    --image_pull_secret ngc-registry \
    --pvcs nemo-workspace:/nemo-workspace
```

After creating a cluster, you should be able to proceed after confirming the scheduler and the workers are all `Running`:

```bash
# Set DASK_CLUSTER_NAME to the value of --name
DASK_CLUSTER_NAME=rapids-dask
kubectl get pods -l "dask.org/cluster-name=$DASK_CLUSTER_NAME"

# NAME                                                     READY   STATUS    RESTARTS      AGE
# rapids-dask-default-worker-587238cf2c-7d685f4d75-k6rnq   1/1     Running   0             57m
# rapids-dask-default-worker-f8ff963886-5577fff76b-qmvcd   1/1     Running   3 (52m ago)   57m
# rapids-dask-scheduler-654799869d-9bw4z                   1/1     Running   0             57m
```

## (Option 1) Running Existing Module

Here's an example of running the existing `gpu_exact_dups` Curator module. The arguments and script name will need to be changed according to the module you wish to run:

```bash
# Set DASK_CLUSTER_NAME to the value of --name
DASK_CLUSTER_NAME=rapids-dask
SCHEDULER_POD=$(kubectl get pods -l "dask.org/cluster-name=$DASK_CLUSTER_NAME,dask.org/component=scheduler" -o name)
# Starts an interactive shell session in the scheduler pod
kubectl exec -it $SCHEDULER_POD -- bash

########################
# Inside SCHEDULER_POD #
########################
# Run the following inside the interactive shell to launch script in the background and
# tee the logs to the /nemo-workspace PVC that was mounted in for persistence.
# The command line flags will need to be replaced with whatever the module script accepts.
# Recall that the PVC is mounted at /nemo-workspace, so any outputs should be written
# to somewhere under /nemo-workspace.

mkdir -p /nemo-workspace/curator/{output,log,profile}
# Write logs to script.log and to a log file with a date suffix
LOGS="/nemo-workspace/curator/script.log /nemo-workspace/curator/script.log.$(date +%y_%m_%d-%H-%M-%S)"
(
echo "Writing to: $LOGS"
gpu_exact_dups \
    --input-data-dirs /nemo-workspace/my_dataset \
    --output-dir /nemo-workspace/curator/output \
    --hash-method md5 \
    --log-dir /nemo-workspace/curator/log \
    --num-files -1 \
    --files-per-partition 1 \
    --profile-path /nemo-workspace/curator/profile \
    --log-frequency 250 \
    --scheduler-address localhost:8786 \
    2>&1
echo "Finished!"
) | tee $LOGS &

# At this point, feel free to disconnect the shell via Ctrl+D or simply
exit
```

At this point you can tail the logs and look for `Finished!` in `/nemo-workspace/curator/script.log`:

```bash
# Command will follow the logs of the running module (Press ctrl+C to close)
kubectl exec -it $SCHEDULER_POD -- tail -f /nemo-workspace/curator/script.log

# Writing to: /nemo-workspace/curator/script.log /nemo-workspace/curator/script.log.24_03_27-15-52-31
# Computing hashes for /nemo-workspace/my_dataset
#                       id                           _hashes
# 0  cc-2023-14-0397113620  91b77eae49c10a65d485ac8ca18d6c43
# 1  cc-2023-14-0397113621  a266f0794cc8ffbd431823e6930e4f80
# 2  cc-2023-14-0397113622  baee533e2eddae764de2cd6faaa1286c
# 3  cc-2023-14-0397113623  87dd52a468448b99078f97e76f528eab
# 4  cc-2023-14-0397113624  a17664daf4f24be58e0e3a3dcf81124a
# Finished!
```

## (Option 2) Running Custom Module

In this example, we'll demonstrate how to run a NeMo Curator module that you've defined locally.

Since your curator module may depend on version of the Curator that differs from what's in the container, we'll need to build a custom image with your code installed:

```bash
# Clone your repo. This example uses the official repo
git clone https://github.com/NVIDIA/NeMo-Curator.git NeMo-Curator-dev

# Checkout specific ref. This example uses a commit in the main branch
git -C NeMo-Curator-dev checkout fc167a6edffd38a55c333742972a5a25b901cb26

# Example NeMo Curator base image. Change it according to your requirements
BASE_IMAGE=nvcr.io/nvidia/nemo-curator:latest
docker build -t nemo-curator-custom ./NeMo-Curator-dev -f - <<EOF
FROM ${BASE_IMAGE}

COPY ./ /NeMo-Curator-dev/
RUN pip install -e /NeMo-Curator-dev
EOF

# Then push this image to your registry: Change <private-registry>/<image>:<tag> accordingly
docker tag nemo-curator-custom <private-registry>/<image>:<tag>
docker push <private-registry>/<image>:<tag>
```

:::{admonition} Note
When using a custom image, you'll likely need to create a different secret unless you pushed to a public registry:

```bash
# Fill in <private-registry>/<username>/<password>
kubectl create secret docker-registry my-private-registry --docker-server=<private-registry> --docker-username=<username> --docker-password=<password>
```

And with this new secret, you create your new Dask cluster:

```bash
# Fill in <private-registry>/<username>/<password>
python create_dask_cluster.py \
    --name rapids-dask \
    --n_workers 2 \
    --n_gpus_per_worker 1 \
    --image <private-registry>/<image>:<tag> \
    --image_pull_secret my-private-registry \
    --pvcs nemo-workspace:/nemo-workspace
```
:::

After the Dask cluster is deployed, you can proceed to run your module. In this example we'll use the `NeMo-Curator/nemo_curator/scripts/find_exact_duplicates.py` module, but you can find other templates in [NeMo-Curator/examples](https://github.com/NVIDIA/NeMo-Curator/tree/main/examples):

```bash
# Set DASK_CLUSTER_NAME to the value of --name
DASK_CLUSTER_NAME=rapids-dask
SCHEDULER_POD=$(kubectl get pods -l "dask.org/cluster-name=$DASK_CLUSTER_NAME,dask.org/component=scheduler" -o name)
# Starts an interactive shell session in the scheduler pod
kubectl exec -it $SCHEDULER_POD -- bash

########################
# Inside SCHEDULER_POD #
########################
# Run the following inside the interactive shell to launch script in the background and
# tee the logs to the /nemo-workspace PVC that was mounted in for persistence.
# The command line flags will need to be replaced with whatever the module script accepts.
# Recall that the PVC is mounted at /nemo-workspace, so any outputs should be written
# to somewhere under /nemo-workspace.

mkdir -p /nemo-workspace/curator/{output,log,profile}
# Append logs to script.log and write to a log file with a date suffix
LOGS="/nemo-workspace/curator/script.log /nemo-workspace/curator/script.log.$(date +%y_%m_%d-%H-%M-%S)"
(
echo "Writing to: $LOGS"
# Recall that /NeMo-Curator-dev was copied and installed in the Dockerfile above
python3 -u /NeMo-Curator-dev/nemo_curator/scripts/find_exact_duplicates.py \
    --input-data-dirs /nemo-workspace/my_dataset \
    --output-dir /nemo-workspace/curator/output \
    --hash-method md5 \
    --log-dir /nemo-workspace/curator/log \
    --files-per-partition 1 \
    --profile-path /nemo-workspace/curator/profile \
    --log-frequency 250 \
    --scheduler-address localhost:8786 \
    2>&1
echo "Finished!"
) | tee $LOGS &

# At this point, feel free to disconnect the shell via Ctrl+D or simply
exit
```

At this point you can tail the logs and look for `Finished!` in `/nemo-workspace/curator/script.log`:

```bash
# Command will follow the logs of the running module (Press ctrl+C to close)
kubectl exec -it $SCHEDULER_POD -- tail -f /nemo-workspace/curator/script.log

# Writing to: /nemo-workspace/curator/script.log /nemo-workspace/curator/script.log.24_03_27-20-52-07
# Reading 2 files
# /NeMo-Curator-dev/nemo_curator/modules/exact_dedup.py:157: UserWarning: Output path f/nemo-workspace/curator/output/_exact_duplicates.parquet already exists and will be overwritten
#   warnings.warn(
# Finished!
```

## Deleting Cluster

After you've finished using the created Dask cluster, you can delete it to release the resources:

```bash
# Where <name> is the flag passed to create_dask_cluster.py. Example: `--name <name>`
kubectl delete daskcluster <name>
```

## Download Data from PVC

To download data from your PVC, you can use the `nemo-workspace-busybox` Pod created earlier:

```bash
# Replace <...> with a path on your local machine
LOCAL_WORKSPACE=<...>

# Tar will fail if LOCAL_WORKSPACE doesn't exist
mkdir -p $LOCAL_WORKSPACE

# Copy file in PVC at /nemo-workspace/foobar.txt to local file-system at $LOCAL_WORKSPACE/nemo-workspace/foobar.txt
kubectl exec nemo-workspace-busybox -- tar cf - /nemo-workspace/foobar.txt | tar xf - -C $LOCAL_WORKSPACE

# Copy directory in PVC /nemo-workspace/fizzbuzz to local file-system at $LOCAL_WORKSPACE/fizzbuzz
kubectl exec nemo-workspace-busybox -- tar cf - /nemo-workspace/fizzbuzz | tar xf - -C $LOCAL_WORKSPACE
``` 