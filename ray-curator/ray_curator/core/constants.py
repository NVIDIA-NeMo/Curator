PROMETHEUS_YAML_TEMPLATE = """
global:
  scrape_interval: 10s # Set the scrape interval to every 10 seconds. Default is every 1 minute.
  evaluation_interval: 10s # Evaluate rules every 10 seconds. The default is every 1 minute.
  # scrape_timeout is set to the global default (10s).

scrape_configs:
# Scrape from each Ray node as defined in the service_discovery.json provided by Ray.
- job_name: 'ray'
  file_sd_configs:
  - files:
    - '{service_discovery_path}'
"""

GRAPHANA_INI_TEMPLATE = """
[security]
allow_embedding = true

[auth.anonymous]
enabled = true
org_name = Main Org.
org_role = Viewer

[paths]
provisioning = {provisioning_path}

[server]
http_port = {grafana_web_port}
"""

GRAPHANA_DASHBOARD_YAML_TEMPLATE = """

apiVersion: 1

providers:
  - name: Ray    # Default dashboards provided by OSS Ray
    folder: Ray
    type: file
    options:
      path: {dashboards_path}
"""

GRAPHANA_DATASOURCE_YAML_TEMPLATE = """
apiVersion: 1
datasources:
- access: proxy
  isDefault: true
  jsonData: {{}}
  name: Prometheus
  secureJsonData: {{}}
  type: prometheus
  url: {prometheus_url}
"""

DEFAULT_EXECUTOR = "Xenna"

DEFAULT_RAY_PORT = 6379

DEFAULT_RAY_DASHBOARD_PORT = 8265

DEFAULT_RAY_TEMP_DIR = "/tmp/ray"  # noqa: S108

DEFAULT_PROMETHEUS_WEB_PORT = 9090

DEFAULT_GRAFANA_WEB_PORT = 3000

DEFAULT_RAY_METRICS_PORT = 8080

DEFAULT_RAY_DASHBOARD_HOST = "127.0.0.1"

DEFAULT_NEMO_CURATOR_METRICS_PATH = "/tmp/nemo_curator_metrics"  # noqa: S108

DEFAULT_RAY_AUTOSCALER_METRIC_PORT = 44217
DEFAULT_RAY_DASHBOARD_METRIC_PORT = 44227
