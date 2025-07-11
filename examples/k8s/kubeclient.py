import argparse

from kubernetes import client, config
from kubernetes.stream import stream


def execute_command_in_scheduler_pod(api_instance, pod_name, namespace, command):
    # Construct command to execute
    exec_command = ["/bin/sh", "-c", command]

    # Execute the command in the pod
    resp = stream(
        api_instance.connect_get_namespaced_pod_exec,
        pod_name,
        namespace,
        command=exec_command,
        stderr=True,
        stdin=False,
        stdout=True,
        tty=False,
    )
    print("Response: " + resp)


def get_scheduler_pod(api_instance, label_selector):
    scheduler_pods = api_instance.list_pod_for_all_namespaces(
        watch=False, label_selector=label_selector
    )
    # This returns the name of the first scheduler pod from the list
    return scheduler_pods.items[0].metadata.name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True)
    parser.add_argument("--kubeconfig", type=str)
    args = parser.parse_args()

    # Load kube config using either the provided kubeconfig or the service account
    if args.kubeconfig:  # Check if args.kubeconfig is not None
        config.load_kube_config(args.kubeconfig)
    else:
        config.load_incluster_config()

    # Create Kubernetes API client
    api_instance = client.CoreV1Api()

    pod_name = get_scheduler_pod(api_instance, "dask.org/component=scheduler")
    namespace = "default"
    execute_command_in_scheduler_pod(api_instance, pod_name, namespace, args.command)
