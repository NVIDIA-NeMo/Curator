---
description: "Set up monitoring and observability for NeMo Curator operations using Prometheus and Grafana dashboards"
categories: ["workflows"]
tags: ["monitoring", "observability", "prometheus", "grafana", "metrics", "performance", "troubleshooting"]
personas: ["admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "universal"
---

(admin-monitoring)=
# Monitoring & Observability

Set up comprehensive monitoring and observability for NeMo Curator operations using Prometheus and Grafana. Monitor pipeline performance, resource utilization, and system health across development and production environments.

## Overview

NeMo Curator provides built-in monitoring capabilities that enable:

- **Real-time Metrics**: Monitor pipeline execution, resource usage, and performance
- **Visual Dashboards**: Pre-configured Grafana dashboards for Ray/Xenna workloads  
- **Automated Setup**: One-command deployment of monitoring infrastructure
- **Custom Dashboards**: Extensible framework for domain-specific monitoring

---

## Quick Setup

### Start Monitoring Services

Launch Prometheus and Grafana with default configuration:

```bash
# Start with default ports (Prometheus: 9090, Grafana: 3000)
python -m nemo_curator.metrics.start_prometheus_grafana --yes

# Or specify custom ports to avoid conflicts
python -m nemo_curator.metrics.start_prometheus_grafana \
  --prometheus_web_port 9091 \
  --grafana_web_port 3001
```

### Access Monitoring Dashboards

After successful startup, access your monitoring interfaces:

- **Grafana Dashboard**: <http://localhost:3000> (admin/admin)
- **Prometheus Dashboard**: <http://localhost:9090>

### Run Monitored Pipeline

Execute any NeMo Curator pipeline to see metrics in action:

```bash
# Example: Run quickstart tutorial with monitoring
python tutorials/quickstart.py
```

---

## Production Configuration

### Environment-Specific Setup

::::{tab-set}

:::{tab-item} Development Environment
:sync: env-dev

```bash
# Development setup with debug logging
export NEMO_CURATOR_LOG_LEVEL="DEBUG"
python -m nemo_curator.metrics.start_prometheus_grafana --yes
```
:::

:::{tab-item} Production Environment
:sync: env-prod

```bash
# Production setup with custom ports and logging
export NEMO_CURATOR_LOG_LEVEL="WARNING"
python -m nemo_curator.metrics.start_prometheus_grafana \
  --prometheus_web_port 9090 \
  --grafana_web_port 3000 \
  --yes
```
:::

::::

### Ray/Xenna Integration

For distributed processing with Ray or Xenna, enable metrics collection:

```bash
# Enable Ray metrics collection
export XENNA_RAY_METRICS_PORT=8080

# Start monitoring services
python -m nemo_curator.metrics.start_prometheus_grafana --yes

# Run distributed pipeline
python your_distributed_pipeline.py
```

---

## Monitoring Components

### Core Services

```{list-table} Monitoring Stack Components
:header-rows: 1
:widths: 25 25 50

* - Component
  - Default Port
  - Purpose
* - Prometheus
  - 9090
  - Metrics collection and storage
* - Grafana
  - 3000
  - Visualization and dashboards
* - Ray Metrics
  - 8080
  - Distributed system metrics
```

### Data Storage

Monitoring data is stored in `/tmp/nemo_curator_metrics/` with the following structure:

```bash
/tmp/nemo_curator_metrics/
├── grafana/
│   ├── dashboards/          # Custom dashboard JSON files
│   ├── provisioning/        # Grafana configuration
│   └── grafana.ini         # Grafana settings
├── prometheus.yml          # Prometheus configuration
├── prometheus.log          # Prometheus logs
└── grafana.log            # Grafana logs
```

---

## Dashboard Configuration

### Pre-built Dashboards

NeMo Curator includes a pre-configured Grafana dashboard for Ray workloads:

- **Xenna Dashboard**: Monitors Ray cluster performance, task execution, and resource utilization
- **Auto-provisioned**: Automatically loaded when Grafana starts
- **Customizable**: Modify or extend based on your requirements

### Adding Custom Dashboards

Create custom monitoring dashboards for specific use cases:

1. **Export Dashboard**: Create dashboard in Grafana UI and export as JSON
2. **Save to Directory**: Place JSON files in `/tmp/nemo_curator_metrics/grafana/dashboards/`
3. **Restart Grafana**: Dashboards are automatically loaded on service restart

```bash
# Example: Add custom dashboard
cp my_custom_dashboard.json /tmp/nemo_curator_metrics/grafana/dashboards/
# Restart monitoring services to load new dashboard
```

---

## Performance Monitoring

### Key Metrics to Monitor

Monitor these critical metrics for optimal NeMo Curator performance:

```{list-table} Essential Performance Metrics
:header-rows: 1
:widths: 30 35 35

* - Metric Category
  - Key Indicators
  - Monitoring Focus
* - Pipeline Execution
  - Task completion rate, processing time
  - Throughput and latency
* - Resource Utilization
  - CPU, memory, GPU usage
  - Resource efficiency
* - Data Processing
  - Records processed, error rates
  - Data quality and reliability
* - System Health
  - Service availability, response times
  - Infrastructure stability
```

### Setting Up Alerts

Configure Grafana alerts for critical thresholds:

1. **Memory Usage**: Alert when memory usage exceeds 80%
2. **Processing Errors**: Alert on error rate increases
3. **Service Health**: Alert when services become unavailable
4. **Performance Degradation**: Alert on significant slowdowns

---

## Troubleshooting

### Common Issues

::::{tab-set}

:::{tab-item} Port Conflicts
:sync: troubleshoot-ports

**Problem**: Services fail to start due to port conflicts

**Solution**:
```bash
# Check port usage
netstat -tulpn | grep :3000
netstat -tulpn | grep :9090

# Use custom ports
python -m nemo_curator.metrics.start_prometheus_grafana \
  --prometheus_web_port 9091 \
  --grafana_web_port 3001
```
:::

:::{tab-item} Service Startup Failures
:sync: troubleshoot-startup

**Problem**: Prometheus or Grafana fails to start

**Solution**:
```bash
# Check logs for errors
tail -f /tmp/nemo_curator_metrics/prometheus.log
tail -f /tmp/nemo_curator_metrics/grafana.log

# Verify permissions
ls -la /tmp/nemo_curator_metrics/
chmod -R 755 /tmp/nemo_curator_metrics/
```
:::

:::{tab-item} Missing Metrics
:sync: troubleshoot-metrics

**Problem**: No metrics appearing in dashboards

**Solution**:
```bash
# Verify Ray metrics port
echo $XENNA_RAY_METRICS_PORT

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Restart services if needed
pkill -f 'prometheus .*'
pkill -f 'grafana server'
python -m nemo_curator.metrics.start_prometheus_grafana --yes
```
:::

::::

### Log Analysis

Monitor service logs for troubleshooting:

```bash
# Real-time log monitoring
tail -f /tmp/nemo_curator_metrics/prometheus.log
tail -f /tmp/nemo_curator_metrics/grafana.log

# Search for specific errors
grep -i error /tmp/nemo_curator_metrics/*.log
grep -i warning /tmp/nemo_curator_metrics/*.log
```

---

## Cleanup and Maintenance

### Stop Monitoring Services

```bash
# Stop all monitoring services
pkill -f 'prometheus .*'
pkill -f 'grafana server'
```

### Remove Persistent Data

```bash
# Remove all monitoring data and configuration
rm -rf /tmp/nemo_curator_metrics/

# Remove only data, keep configuration
rm -rf /tmp/nemo_curator_metrics/data/
```

### Maintenance Tasks

Regular maintenance for production environments:

1. **Log Rotation**: Set up log rotation for monitoring service logs
2. **Data Retention**: Configure Prometheus data retention policies  
3. **Backup Dashboards**: Export and backup custom dashboard configurations
4. **Security Updates**: Keep Prometheus and Grafana versions current

---

## Integration with Deployment

### Kubernetes Deployment

For Kubernetes deployments, integrate monitoring with your cluster setup:

```yaml
# Example: Add monitoring to your Kubernetes deployment
apiVersion: v1
kind: ConfigMap
metadata:
  name: nemo-curator-monitoring
data:
  setup.sh: |
    #!/bin/bash
    python -m nemo_curator.metrics.start_prometheus_grafana --yes
```

### Slurm Integration

For Slurm clusters, include monitoring in your job scripts:

```bash
#!/bin/bash
#SBATCH --job-name=nemo-curator-with-monitoring

# Start monitoring services
python -m nemo_curator.metrics.start_prometheus_grafana --yes

# Run your curation pipeline
python your_pipeline.py

# Cleanup (optional)
pkill -f 'prometheus .*'
pkill -f 'grafana server'
```

---

## Related Topics

- **[Configuration Guide](admin-config)** - Environment and deployment configuration
- **[Kubernetes Deployment](admin-deployment-kubernetes)** - Container orchestration setup
- **[Slurm Deployment](admin-deployment-slurm)** - HPC cluster deployment
- **[Infrastructure References](reference-infra-monitoring)** - Technical monitoring details
