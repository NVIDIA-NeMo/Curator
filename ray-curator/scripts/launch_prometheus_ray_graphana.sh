set -e
set -x

# Get the absolute path to the grafana config directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAFANA_CONFIG_DIR="$(cd "${SCRIPT_DIR}/../configs/metrics/grafana" && pwd)"
PROVISIONING_PATH="${GRAFANA_CONFIG_DIR}/provisioning"
DASHBOARDS_PATH="${GRAFANA_CONFIG_DIR}/dashboards"

# Update grafana.ini with the dynamic provisioning path
sed -i "s|provisioning = .*|provisioning = ${PROVISIONING_PATH}|g" "${GRAFANA_CONFIG_DIR}/grafana.ini"

# Update dashboards configuration with the dynamic dashboards path
sed -i "s|path: .*|path: ${DASHBOARDS_PATH}|g" "${GRAFANA_CONFIG_DIR}/provisioning/dashboards/default.yml"

mkdir -p .metrics

cd .metrics
# Check if prometheus is already runnings
if ! pgrep -f "prometheus .*" > /dev/null; then
    # Install prometheus
    ray metrics launch-prometheus
else
    echo "Prometheus is already running, skipping installation"
fi

# Install and start grafana
# This is only for amd64 architecture,
# If you are on a different architecture, please install grafana manually
if [[ $(uname -m) == "x86_64" ]]; then
    # Download and extract grafana only if it does not exist
    if [ ! -d "grafana-v12.0.2" ]; then
        wget https://dl.grafana.com/enterprise/release/grafana-enterprise-12.0.2.linux-amd64.tar.gz
        tar -zxvf grafana-enterprise-12.0.2.linux-amd64.tar.gz
    fi

    cd grafana-v12.0.2
    # Start grafana with nohup and redirect logs if grafana is not running
    if ! pgrep -f "grafana server" > /dev/null; then
        nohup ./bin/grafana-server --config "${GRAFANA_CONFIG_DIR}/grafana.ini" web > ../grafana.log 2>&1 &
    fi
fi

echo "If you are running using Xenna, please remember to export XENNA_RAY_METRICS_PORT=8080"
echo "You can access the grafana dashboard at http://localhost:3000, username: admin, password: admin"
echo "You can access the prometheus dashboard at http://localhost:9090"
echo "Currently, we only provide a xenna dashboard,"
echo "but you can add more dashboards by adding json files"
echo "in ${GRAFANA_CONFIG_DIR}/dashboards"
echo "from /tmp/ray/session_latest/metrics/grafana/dashboards"
echo "To kill prometheus and grafana, run: pkill -f 'prometheus .*' && pkill -f 'grafana server'"
echo "Prometheus stores tha persistant data inside .metrics/data, if you want to delete the data, run: rm -rf .metrics/data"
