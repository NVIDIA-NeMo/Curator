---
description: "Technical reference for NeMo Curator monitoring infrastructure including Prometheus configuration, Grafana dashboards, and metrics collection"
categories: ["reference"]
tags: ["monitoring", "prometheus", "grafana", "metrics", "infrastructure", "observability", "performance"]
personas: ["admin-focused", "devops-focused", "mle-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(reference-infra-monitoring)=
# Monitoring Infrastructure

Technical reference for NeMo Curator's monitoring and observability infrastructure. This section covers the implementation details, configuration options, and technical specifications for production monitoring deployments.

```{note}
For setup and usage instructions, see the [Admin Monitoring Guide](admin-monitoring). This reference focuses on technical implementation details.
```

---

## Architecture Overview

### Monitoring Stack Components

```{mermaid}
graph TB
    A[NeMo Curator Pipeline] --> B[Ray Metrics Exporter]
    B --> C[Prometheus Server]
    C --> D[Grafana Dashboard]
    
    E[System Metrics] --> C
    F[Application Logs] --> G[Log Aggregation]
    
    C --> H[Time Series Database]
    D --> I[Alert Manager]
    
    subgraph "Monitoring Infrastructure"
        C
        D
        H
        I
    end
```

### Data Flow

1. **Metrics Collection**: Ray workers export metrics to Prometheus endpoints
2. **Storage**: Prometheus stores time-series data locally
3. **Visualization**: Grafana queries Prometheus for dashboard rendering
4. **Alerting**: Alert Manager processes threshold-based notifications

---

## Prometheus Configuration

### Service Discovery

NeMo Curator uses file-based service discovery for Ray cluster monitoring:

```yaml
# /tmp/nemo_curator_metrics/prometheus.yml
global:
  scrape_interval: 10s
  evaluation_interval: 10s

scrape_configs:
- job_name: 'ray'
  file_sd_configs:
  - files:
    - /tmp/ray/prom_metrics_service_discovery.json
```

### Metrics Endpoints

```{list-table} Prometheus Scrape Targets
:header-rows: 1
:widths: 30 25 45

* - Component
  - Default Port
  - Metrics Endpoint
* - Ray Head Node
  - 8080
  - `/metrics`
* - Ray Worker Nodes
  - 8080
  - `/metrics`
* - Prometheus Server
  - 9090
  - `/metrics` (self-monitoring)
```

### Configuration Parameters

```{list-table} Prometheus Configuration Options
:header-rows: 1
:widths: 35 20 45

* - Parameter
  - Default Value
  - Description
* - `scrape_interval`
  - 10s
  - Frequency of metrics collection
* - `evaluation_interval`
  - 10s
  - Rule evaluation frequency
* - `retention.time`
  - 15d
  - Data retention period
* - `web.listen-address`
  - :9090
  - HTTP server bind address
```

---

## Grafana Configuration

### Provisioning Structure

```bash
/tmp/nemo_curator_metrics/grafana/
├── grafana.ini                 # Main configuration
├── provisioning/
│   ├── datasources/
│   │   └── default.yml        # Prometheus datasource
│   └── dashboards/
│       └── default.yml        # Dashboard provider
└── dashboards/
    └── xenna_grafana_dashboard.json  # Pre-built dashboard
```

### Datasource Configuration

```yaml
# provisioning/datasources/default.yml
apiVersion: 1
datasources:
- access: proxy
  isDefault: true
  jsonData: {}
  name: Prometheus
  secureJsonData: {}
  type: prometheus
  url: http://localhost:9090
```

### Dashboard Provisioning

```yaml
# provisioning/dashboards/default.yml
apiVersion: 1
providers:
  - name: Ray
    folder: Ray
    type: file
    options:
      path: /tmp/nemo_curator_metrics/grafana/dashboards
```

---

## Metrics Specification

### Ray Cluster Metrics

```{list-table} Core Ray Metrics
:header-rows: 1
:widths: 40 20 40

* - Metric Name
  - Type
  - Description
* - `ray_cluster_active_nodes`
  - Gauge
  - Number of active cluster nodes
* - `ray_tasks_total`
  - Counter
  - Total tasks executed
* - `ray_tasks_running`
  - Gauge
  - Currently running tasks
* - `ray_memory_used_bytes`
  - Gauge
  - Memory usage per node
* - `ray_cpu_utilization`
  - Gauge
  - CPU utilization percentage
```

### NeMo Curator Metrics

```{list-table} Application-Specific Metrics
:header-rows: 1
:widths: 40 20 40

* - Metric Name
  - Type
  - Description
* - `nemo_curator_pipeline_duration_seconds`
  - Histogram
  - Pipeline execution time
* - `nemo_curator_records_processed_total`
  - Counter
  - Total records processed
* - `nemo_curator_errors_total`
  - Counter
  - Processing errors encountered
* - `nemo_curator_stage_duration_seconds`
  - Histogram
  - Individual stage execution time
```

---

## Performance Considerations

### Resource Requirements

```{list-table} Monitoring Resource Usage
:header-rows: 1
:widths: 25 25 25 25

* - Component
  - CPU
  - Memory
  - Storage
* - Prometheus
  - 0.5-1 CPU
  - 2-4 GB
  - 1-10 GB/day
* - Grafana
  - 0.1-0.5 CPU
  - 512 MB-1 GB
  - 100 MB
* - Ray Metrics
  - 0.1 CPU
  - 256 MB
  - N/A
```

### Scaling Considerations

**Small Deployments (< 10 nodes)**:

- Default configuration sufficient
- Local storage acceptable
- Basic alerting rules

**Medium Deployments (10-100 nodes)**:

- Increase Prometheus retention
- Consider remote storage
- Enhanced alerting configuration

**Large Deployments (> 100 nodes)**:

- Prometheus federation or sharding
- Remote storage (e.g., Thanos)
- Advanced alerting and routing

---

## Security Configuration

### Authentication

```ini
# grafana.ini security section
[security]
allow_embedding = true

[auth.anonymous]
enabled = true
org_name = Main Org.
org_role = Viewer
```

### Network Security

```{list-table} Security Recommendations
:header-rows: 1
:widths: 30 70

* - Component
  - Security Measures
* - Prometheus
  - Bind to localhost only, use reverse proxy for external access
* - Grafana
  - Enable authentication, configure HTTPS, restrict admin access
* - Ray Metrics
  - Internal network only, firewall external access
```

---

## API Reference

### Prometheus HTTP API

```bash
# Query current metrics
curl "http://localhost:9090/api/v1/query?query=ray_cluster_active_nodes"

# Query time range
curl "http://localhost:9090/api/v1/query_range?query=ray_tasks_total&start=2025-01-01T00:00:00Z&end=2025-01-01T01:00:00Z&step=15s"

# Get targets status
curl "http://localhost:9090/api/v1/targets"
```

### Grafana HTTP API

```bash
# Get dashboard list
curl -H "Authorization: Bearer <token>" "http://localhost:3000/api/dashboards/home"

# Create dashboard
curl -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d @dashboard.json \
  "http://localhost:3000/api/dashboards/db"
```

---

## Troubleshooting Reference

### Common Configuration Issues

```{list-table} Configuration Troubleshooting
:header-rows: 1
:widths: 40 60

* - Issue
  - Solution
* - Prometheus not discovering Ray targets
  - Check `/tmp/ray/prom_metrics_service_discovery.json` exists
* - Grafana datasource connection failed
  - Verify Prometheus URL in datasource configuration
* - Missing dashboard panels
  - Confirm dashboard JSON is valid and properly provisioned
* - High memory usage
  - Reduce retention period or increase scrape interval
```

### Log Analysis

```bash
# Prometheus startup issues
grep -i "error\|fatal" /tmp/nemo_curator_metrics/prometheus.log

# Grafana plugin issues  
grep -i "plugin\|error" /tmp/nemo_curator_metrics/grafana.log

# Ray metrics connectivity
curl -s http://localhost:8080/metrics | head -20
```

---

## Extension Points

### Custom Metrics

Add application-specific metrics to your NeMo Curator stages:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define custom metrics
RECORDS_PROCESSED = Counter('nemo_curator_records_processed_total', 
                           'Total records processed', ['stage_name'])
PROCESSING_TIME = Histogram('nemo_curator_processing_duration_seconds',
                           'Time spent processing', ['stage_name'])

# Use in your stages
class CustomStage(ProcessingStage):
    def process(self, data):
        with PROCESSING_TIME.labels(stage_name=self.__class__.__name__).time():
            result = self._process_data(data)
            RECORDS_PROCESSED.labels(stage_name=self.__class__.__name__).inc(len(result))
            return result
```

### Custom Dashboards

Create domain-specific monitoring dashboards:

```json
{
  "dashboard": {
    "title": "NeMo Curator Text Processing",
    "panels": [
      {
        "title": "Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(nemo_curator_records_processed_total[5m])",
            "legendFormat": "Records/sec"
          }
        ]
      }
    ]
  }
}
```

---

## Related Topics

- **[Admin Monitoring Guide](admin-monitoring)** - Setup and usage instructions
- **[Memory Management](reference-infra-memory-management)** - Resource optimization
- **[GPU Processing](reference-infra-gpu-processing)** - GPU monitoring considerations
- **[Container Environments](reference-infrastructure-container-environments)** - Containerized monitoring
