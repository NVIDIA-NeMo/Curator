# Ray start

RAY_MAX_LIMIT_FROM_API_SERVER=40000 RAY_MAX_LIMIT_FROM_DATA_SOURCE=40000 ray start --head --node-ip-address 10.57.203.156 --port 6379 --metrics-export-port 8080 --dashboard-port 8265 --dashboard-host 127.0.0.1 --ray-client-server-port 10001 --temp-dir /tmp/ray --disable-usage-stats --include-dashboard=True

# Run tutorial

RAY_ADDRESS=10.57.203.156:6379 DASHBOARD_METRIC_PORT=44227 AUTOSCALER_METRIC_PORT=44217 XENNA_RAY_METRICS_PORT=8080 XENNA_RESPECT_CUDA_VISIBLE_DEVICES=1 python3 main.py --input-path /localhome/local-asolergibert/Curator/tutorials/text/megatron-tokenizer/datasets/tinystories --output-path /localhome/local-asolergibert/Curator/tutorials/text/megatron-tokenizer/datasets/tinystories-tokens

# Check Ray

RAY_ADDRESS=10.57.203.156:6379 ray status
