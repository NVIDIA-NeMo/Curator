# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DEFAULT_NEMO_CURATOR_METRICS_PATH = "/tmp/nemo_curator_metrics"  # noqa: S108
GRAFANA_VERSION = "12.0.2"

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
    - /tmp/ray/prom_metrics_service_discovery.json
"""

GRAFANA_INI_TEMPLATE = """
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

GRAFANA_DASHBOARD_YAML_TEMPLATE = """

apiVersion: 1

providers:
  - name: Ray    # Default dashboards provided by OSS Ray
    folder: Ray
    type: file
    options:
      path: {dashboards_path}
"""

GRAFANA_DATASOURCE_YAML_TEMPLATE = """
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

DEFAULT_PROMETHEUS_WEB_PORT = 9090

DEFAULT_GRAFANA_WEB_PORT = 3000
