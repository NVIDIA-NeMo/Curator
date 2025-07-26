======================================
Running NeMo Curator on AWS EKS
======================================

--------------------------------------
Background
--------------------------------------
AWS EKS is a fully managed service that makes it easier to run Kubernetes on AWS without needing to install, operate, and maintain your own Kubernetes control plane.

Running NeMo Curator on AWS EKS offers streamlined Kubernetes management integrated with AWS services like CloudWatch for enhanced monitoring and logging, and native auto-scaling capabilities.

For more details, refer to `EKS documentation <https://docs.aws.amazon.com/eks/latest/userguide/what-is-eks.html>`__

This guide covers all essential prerequisites. It includes an example demonstrating how to create an EFS storage class and offers step-by-step instructions for setting up an EFS Persistent Volume Claim to dynamically provisioning Kubernetes Persistent Volume. Furthermore, it outlines the required steps to deploy a Dask cluster and delves into utilizing the Kubernetes Python client library to assign NeMo-Curator tasks to the Dask scheduler.


Prerequisites:
----------------

* EKS Cluster:
    * `Dask Operator <https://kubernetes.dask.org/en/latest/installing.html>`__
    * If self managed node group is created with ubuntu worker nodes then install GPU operator. When setting up a self-managed node group with Ubuntu worker nodes in Amazon EKS, it's advantageous to install the GPU Operator. The GPU Operator is highly recommended as it simplifies the deployment and management of NVIDIA GPU resources within Kubernetes clusters. This operator automates the installation of NVIDIA drivers, integrates with container runtimes like containerd through the NVIDIA Container Toolkit, manages device plugins, and provides monitoring capabilities.
        `GPU operator <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html>`__
    * If EKS managed node group is created with Amazon Linux 2 worker nodes then install Nvidia device plugin. This approach has a limitation, the pre-installed NVIDIA GPU driver version and NVIDIA container runtime version lags the release schedule from NVIDIA requires you to assume the responsibility ofr upgrading the NVIDIA devicw plugin version.
        `Nvidia Device Plugin installation <https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html>`__
     For more details, please refer `NVIDIA GPU Operator with Amazon EKS <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/amazon-eks.html>`__
* Storage:
    * `EFS for EKS <https://github.com/kubernetes-sigs/aws-efs-csi-driver/blob/master/docs/efs-create-filesystem.md>`__ (setup by Kubernetes cluster admin)

Create a Storage Class for AWS EFS
----------------------------------

.. code-block:: yaml

   cat <<EOF | kubectl apply -f -
   kind: StorageClass
   apiVersion: storage.k8s.io/v1
   metadata:
     name: efs-sc
   provisioner: efs.csi.aws.com
   parameters:
     provisioningMode: efs-ap
     fileSystemId: ${FileSystemId}  # Replace with your actual FileSystemId
     directoryPerms: "700"
   EOF

In the above YAML:

- Replace `${FileSystemId}` with the actual EFS FileSystemId from AWS.
- This definition sets up a StorageClass named `efs-sc` that provisions PersistentVolumes using the AWS EFS CSI Driver.


PVC Definition
--------------------------------
Now, we can use the storage class created in the previous step to create a PVC.

.. code-block:: yaml

   cat <<EOF | kubectl apply -f -
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: nemo-workspace
   spec:
     accessModes:
       - ReadWriteMany
     storageClassName: efs-sc
     resources:
       requests:
         storage: 150Gi
   EOF

This PVC requests 150GiB of storage with ReadWriteMany access mode from the efs-sc StorageClass.

Dask cluster creation:
----------------------

Please refer index.rst for instructions on creating a Docker secret to utilize the NeMo image and upload data to the PVC created in the previous step.

Python environment can be setup by executing the following commands:

.. code-block:: bash

    python3 -m venv venv
    source venv/bin/activate

    pip install dask_kubernetes
    pip install kubernetes

The environment to run the provided scripts needs only dask-kubernetes and kubernetes packages.

.. code-block:: bash

  python3 examples/k8s/create_dask_cluster.py \
          --name dask-gpu-cluster \
          --n_workers 2 \
          --image nvcr.io/nvidian/nemo:nightly \
          --image_pull_secret ngc-registry \
          --pvcs nemo-workspace:/nemo-workspace

The above command uses the create_dask_cluster python code to create 2 GPU dask workers with PVCs attached to the dask-gpu-cluster.

After the cluster is created, you can check if the scheduler and worker pods are running by executing:

.. code-block:: bash

    kubectl get pods

The output will look as follows:

+---------------------------------------------------------+-------+---------+----------+------+
| NAME                                                    | READY | STATUS  | RESTARTS | AGE  |
+---------------------------------------------------------+-------+---------+----------+------+
| dask-kubernetes-operator-1720671237-6f8c579d4d-gk8pg    | 1/1   | Running | 0        | 27h  |
+---------------------------------------------------------+-------+---------+----------+------+
| rapids-dask-default-worker-be7c9e6b19-668b8cc459-cxcwg  | 1/1   | Running | 0        | 21h  |
+---------------------------------------------------------+-------+---------+----------+------+
| rapids-dask-default-worker-f4b5c0ff1a-66db8c4cb5-w68gd  | 1/1   | Running | 0        | 21h  |
+---------------------------------------------------------+-------+---------+----------+------+
| rapids-dask-scheduler-5dfc446f-9tw2t                    | 1/1   | Running | 0        | 21h  |
+---------------------------------------------------------+-------+---------+----------+------+



Use Kubernetes Python client library to submit NeMo-Curator jobs to the Dask scheduler:
------------------------------------------------------

In this method, we programmatically connect to the scheduler pod using the Kubernetes Python client library to execute the existing NeMo curator modules.

This approach can be used when employing another wrapper or service to submit jobs to Dask cluster in a distributed manner.

1) To execute existing NeMo curator modules in a scheduler pod from outside the EKS cluster, run the following:

.. code-block:: bash

    python3 examples/k8s/kubeclient.py --command "add_id --scheduler-address localhost:8786 --input-data-dir=/nemo-workspace/arxiv --output-data-dir=/nemo-workspace/arxiv-addid/" --kubeconfig "~/.kube/config"

In this context, the --kubeconfig parameter is utilized to enable the Kubernetes Python client library to automatically load configuration settings from "~/.kube/config".

Note: The default location of kubeconfig is $HOME/.kube/.config. You can verify this by running:

.. code-block:: bash

    kubectl get pod   -v6 2>&1 |awk  '/Config loaded from file:/{print $NF}'

`v6` sets the verbose level to see the kubeconfig file in use.


2) To execute existing NeMo curator modules in a scheduler pod from another pod within the EKS cluster, add necessary permissions, such as pods/exec, and spin up a client pod.

This approach is allows the execution of NeMo Curator modules within the scheduler pod from a separate client pod. This separation ensures that the client pod can be provisioned with specific permissions tailored for executing commands and accessing resources within the Kubernetes environment.

Moreover, deploying this client pod can be orchestrated by another service such as AWS Batch, facilitating scalable and efficient management of computational tasks within Kubernetes clusters.


.. code-block:: yaml

    cat <<EOF | kubectl apply -f -
    apiVersion: rbac.authorization.k8s.io/v1
    kind: ClusterRole
    metadata:
      name: pod-exec
    rules:
    - apiGroups:
      - ""
      resources:
      - pods
      - pods/exec
      verbs:
      - list
      - get
      - watch
      - create
    ---
    apiVersion: rbac.authorization.k8s.io/v1
    kind: ClusterRoleBinding
    metadata:
      name: allow-pods-exec
    subjects:
    - kind: ServiceAccount
      name: default
      namespace: default
    roleRef:
      kind: ClusterRole
      name: pod-exec
      apiGroup: rbac.authorization.k8s.io
    EOF

The above yaml file creates a ClusterRole and a ClusterRoleBinding.

ClusterRole Definition:
  * Specifies permissions (rules) for interacting with Kubernetes pods.
  * `resources`: `["pods", "pods/exec"]` specifies the resources pods and pods/exec.
  * `verbs`: `["list", "get", "watch", "create"]` lists the actions allowed on these resources (`list`, `get`, `watch`, `create`).


ClusterRoleBinding Definition:
  * Binds the `pod-exec` ClusterRole to a specific ServiceAccount (`default` in the `default` namespace).
  * This means that any pods using the `default` ServiceAccount in the `default` namespace will have the permissions specified in the `pod-exec` ClusterRole.


Now, we can spin up a client pod.

.. code-block:: yaml

    cat <<EOF | kubectl apply -f -
    apiVersion: v1
    kind: Pod
    metadata:
      name: client-pod
      labels:
        app: client
    spec:
      containers:
        - name: client
          image: python:3.10-slim-bullseye
          command: ["sh", "-c", "pip install kubernetes && sleep infinity"]
    EOF

Here, we are using a light-weight public python docker image and installing kubernetes Python client package so that we can run kubeclient.py from this client pod and connect to the scheduler pod to run existing NeMo Curator modules.

Once the client-pod is up and running we can copy the kubeclient.py script into the client pod and and run the the script.

.. code-block:: bash

    kubectl cp examples/k8s/kubeclient.py client-pod:kubeclient.py
    kubectl exec client-pod -- python3 kubeclient.py --command "add_id --scheduler-address localhost:8786 --input-data-dir=/nemo-workspace/arxiv --output-data-dir=/nemo-workspace/arxiv-addid/"


