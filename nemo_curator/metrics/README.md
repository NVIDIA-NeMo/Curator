# NeMo Curator Metrics

This module provides monitoring and visualization capabilities for NeMo Curator operations using Prometheus and Grafana.

## Overview

The metrics module enables real-time monitoring of data curation workloads by:
- Setting up Prometheus for metrics collection
- Configuring Grafana for visualization dashboards
- Providing pre-built dashboards for Ray/Xenna workloads

## Quick Start

### Start Monitoring Services

```bash
# Start with default ports (Prometheus: 9090, Grafana: 3000)
python -m nemo_curator.metrics.start_prometheus_grafana --yes

# Or specify custom ports
python -m nemo_curator.metrics.start_prometheus_grafana --prometheus_web_port 9091 --grafana_web_port 3001
```

### Run a Pipeline

```bash
python examples/quickstart.py
```

### Access Dashboards

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus Dashboard**: http://localhost:9090

### For Ray/Xenna Users

```bash
export XENNA_RAY_METRICS_PORT=8080
```

## Components

- `start_prometheus_grafana.py` - Main script to launch monitoring services
- `utils.py` - Helper functions for downloading and configuring services
- `constants.py` - Configuration constants and templates
- `xenna_grafana_dashboard.json` - Pre-configured Grafana dashboard for Ray workloads

## Cleanup

```bash
# Stop services
pkill -f 'prometheus .*'
pkill -f 'grafana server'

# Remove persistent data
rm -rf data/
```

## Custom Dashboards

Add custom Grafana dashboards by placing JSON files in:
`/tmp/nemo_curator_metrics/grafana/dashboards/`
