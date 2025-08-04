---
description: "Configure storage access, API keys, and security credentials for NeMo Curator deployments across cloud and on-premises environments"
categories: ["how-to-guides"]
tags: ["storage-credentials", "cloud-storage", "api-keys", "security", "aws", "azure", "gcs", "file-systems"]
personas: ["admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "universal"
---

(admin-config-storage-credentials)=
# Storage & Credentials Configuration

Configure storage access, API keys, and security credentials for NeMo Curator deployments. This guide focuses on operational setup for different storage backends and credential management.

```{tip}
**Using These Credentials**: After configuring storage access, you can use these credentials in your deployments:
- {doc}`Kubernetes Deployment <../deployment/kubernetes>`: Apply credentials via Kubernetes secrets
- {doc}`Slurm Deployment <../deployment/slurm/index>`: Mount credential files in Slurm containers
- {doc}`Deployment Environment Configuration <deployment-environments>`: Environment-specific credential patterns
```

---

## Cloud Storage Configuration

### Amazon S3 Configuration

#### AWS Credentials Setup

::::{tab-set}

:::{tab-item} AWS Credentials File
:sync: aws-creds-file

```ini
# ~/.aws/credentials
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY

[production]
aws_access_key_id = PROD_ACCESS_KEY_ID  
aws_secret_access_key = PROD_SECRET_ACCESS_KEY
```
:::

:::{tab-item} Environment Variables
:sync: aws-creds-env

```bash
# AWS credentials via environment variables
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-west-2"
export AWS_PROFILE="production"  # Optional: use specific profile
```
:::

:::{tab-item} IAM Roles (Recommended for Production)
:sync: aws-creds-iam

```bash
# Use IAM roles for EC2/EKS deployments
export AWS_ROLE_ARN="arn:aws:iam::123456789012:role/NemoCuratorRole"
export AWS_WEB_IDENTITY_TOKEN_FILE="/var/run/secrets/eks.amazonaws.com/serviceaccount/token"
```
:::

::::

#### S3 Configuration Options

```bash
# S3-specific settings
export AWS_S3_ENDPOINT_URL="https://s3.amazonaws.com"  # Custom endpoint
export AWS_S3_USE_SSL="true"
export AWS_S3_VERIFY_SSL="true"
export AWS_S3_ADDRESSING_STYLE="virtual"  # or "path"

# Performance tuning
export AWS_S3_MAX_CONCURRENT_REQUESTS="10"
export AWS_S3_MAX_BANDWIDTH="100MB/s"
export AWS_S3_MULTIPART_THRESHOLD="64MB"
export AWS_S3_MULTIPART_CHUNKSIZE="16MB"
```

### Azure Blob Storage Configuration

#### Azure Credentials Setup

::::{tab-set}

:::{tab-item} Service Principal
:sync: azure-creds-sp

```bash
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
```
:::

:::{tab-item} Managed Identity (Recommended for Azure VMs)
:sync: azure-creds-msi

```bash
export AZURE_USE_MSI="true"
export AZURE_CLIENT_ID="managed-identity-client-id"  # Optional
```
:::

:::{tab-item} Connection String
:sync: azure-creds-conn

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=myaccount;AccountKey=mykey;EndpointSuffix=core.windows.net"
```
:::

::::

#### Azure Storage Configuration

```bash
# Azure Blob Storage settings
export AZURE_STORAGE_ACCOUNT="your-storage-account"
export AZURE_STORAGE_CONTAINER="nemo-curator-data"
export AZURE_STORAGE_ENDPOINT="https://myaccount.blob.core.windows.net/"

# Performance settings
export AZURE_STORAGE_MAX_CONCURRENCY="10"
export AZURE_STORAGE_BLOCK_SIZE="4MB"
```

### Google Cloud Storage Configuration

#### GCS Credentials Setup

::::{tab-set}

:::{tab-item} Service Account Key File
:sync: gcs-creds-key

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```
:::

:::{tab-item} Workload Identity (Recommended for GKE)
:sync: gcs-creds-workload

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
# Workload Identity automatically handles authentication
```
:::

::::

#### GCS Configuration Options

```bash
# GCS-specific settings
export GCS_PROJECT_ID="your-project-id"
export GCS_BUCKET="nemo-curator-bucket"
export GCS_DEFAULT_LOCATION="US"

# Performance tuning
export GCS_MAX_RETRY_DELAY="60"
export GCS_TOTAL_TIMEOUT="300"
```

---

## API Keys and Model Access

### Hugging Face Configuration

```bash
# Hugging Face Hub authentication
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
export HF_HOME="/shared/cache/huggingface"  # Cache directory
export HF_HUB_CACHE="/shared/cache/huggingface/hub"
export HF_DATASETS_CACHE="/shared/cache/huggingface/datasets"

# Offline mode (for air-gapped environments)
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
```

### OpenAI API Configuration

```bash
# OpenAI API credentials
export OPENAI_API_KEY="sk-your-openai-api-key"
export OPENAI_ORGANIZATION="org-your-organization-id"  # Optional
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Custom endpoint

# Rate limiting and timeouts
export OPENAI_MAX_RETRIES="3"
export OPENAI_TIMEOUT="60"
```

### NVIDIA API Configuration

```bash
# NVIDIA NGC API
export NGC_API_KEY="your-ngc-api-key"
export NGC_ORG="your-organization"
export NGC_TEAM="your-team"  # Optional

# NVIDIA NIM (NVIDIA Inference Microservices)
export NVIDIA_API_KEY="nvapi-your-api-key"
export NIM_BASE_URL="https://integrate.api.nvidia.com/v1"
```

### Anthropic API Configuration

```bash
# Anthropic Claude API
export ANTHROPIC_API_KEY="sk-ant-your-api-key"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
```

---

## File System Configuration

### Shared File Systems

#### NFS Configuration

```bash
# NFS mount settings
export NFS_SERVER="nfs-server.example.com"
export NFS_PATH="/shared/nemo-curator"
export NFS_MOUNT_OPTIONS="rw,hard,intr,timeo=60,retrans=2"

# Local mount point
export SHARED_DATA_DIR="/mnt/shared/data"
export SHARED_CACHE_DIR="/mnt/shared/cache"
export SHARED_LOGS_DIR="/mnt/shared/logs"
```

#### Lustre Configuration

```bash
# Lustre parallel file system
export LUSTRE_MOUNT="/lustre/nemo-curator"
export LUSTRE_STRIPE_COUNT="4"  # Number of OSTs to stripe across
export LUSTRE_STRIPE_SIZE="1MB"  # Stripe size

# Lustre-optimized paths
export DATA_DIR="${LUSTRE_MOUNT}/data"
export CACHE_DIR="${LUSTRE_MOUNT}/cache"
export OUTPUT_DIR="${LUSTRE_MOUNT}/output"
```

#### GPFS (IBM Spectrum Scale) Configuration

```bash
# GPFS configuration
export GPFS_MOUNT="/gpfs/nemo-curator"
export GPFS_BLOCK_SIZE="256KB"
export GPFS_PREFETCH_THREADS="8"

# GPFS-optimized settings
export DATA_DIR="${GPFS_MOUNT}/data"
export TEMP_DIR="${GPFS_MOUNT}/tmp"
```

### Local Storage Optimization

```bash
# Local SSD optimization
export LOCAL_SSD_DIR="/mnt/local-ssd"
export TEMP_DIR="${LOCAL_SSD_DIR}/tmp"
export SCRATCH_DIR="${LOCAL_SSD_DIR}/scratch"

# I/O optimization
export IO_THREADS="8"
export BUFFER_SIZE="64MB"
```

---

## Security Configuration

### SSL/TLS Configuration

```bash
# SSL certificate paths
export SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
export SSL_CERT_DIR="/etc/ssl/certs"
export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"

# Disable SSL verification (NOT recommended for production)
export PYTHONHTTPSVERIFY="0"
export CURL_INSECURE="1"
```

### Proxy Configuration

```bash
# HTTP/HTTPS proxy settings
export HTTP_PROXY="http://proxy.company.com:8080"
export HTTPS_PROXY="http://proxy.company.com:8080"
export NO_PROXY="localhost,127.0.0.1,.company.com"

# Proxy authentication
export HTTP_PROXY="http://username:password@proxy.company.com:8080"
export HTTPS_PROXY="http://username:password@proxy.company.com:8080"
```

### Secrets Management

#### Kubernetes Secrets

```yaml
# kubernetes-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: nemo-curator-secrets
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: "your-access-key"
  AWS_SECRET_ACCESS_KEY: "your-secret-key"
  HUGGINGFACE_HUB_TOKEN: "hf_your_token"
  OPENAI_API_KEY: "sk-your-openai-key"
```

#### HashiCorp Vault Integration

```bash
# Vault configuration
export VAULT_ADDR="https://vault.company.com:8200"
export VAULT_TOKEN="your-vault-token"
export VAULT_NAMESPACE="nemo-curator"

# Vault secret paths
export VAULT_AWS_PATH="secret/nemo-curator/aws"
export VAULT_OPENAI_PATH="secret/nemo-curator/openai"
```

---

## Credential Validation

### Storage Access Validation

```python
# Validate S3 access
import boto3
try:
    s3 = boto3.client('s3')
    buckets = s3.list_buckets()
    print("✓ S3 access configured correctly")
    print(f"Available buckets: {[b['Name'] for b in buckets['Buckets']]}")
except Exception as e:
    print(f"✗ S3 access failed: {e}")

# Validate Azure access
try:
    from azure.storage.blob import BlobServiceClient
    blob_service = BlobServiceClient.from_connection_string(
        os.environ['AZURE_STORAGE_CONNECTION_STRING']
    )
    containers = list(blob_service.list_containers())
    print("✓ Azure Blob Storage access configured correctly")
except Exception as e:
    print(f"✗ Azure access failed: {e}")

# Validate GCS access
try:
    from google.cloud import storage
    client = storage.Client()
    buckets = list(client.list_buckets())
    print("✓ GCS access configured correctly")
except Exception as e:
    print(f"✗ GCS access failed: {e}")
```

### API Key Validation

```python
# Validate Hugging Face access
try:
    from huggingface_hub import whoami
    user_info = whoami()
    print(f"✓ Hugging Face authenticated as: {user_info['name']}")
except Exception as e:
    print(f"✗ Hugging Face authentication failed: {e}")

# Validate OpenAI access
try:
    import openai
    client = openai.OpenAI()
    models = client.models.list()
    print("✓ OpenAI API access configured correctly")
except Exception as e:
    print(f"✗ OpenAI API access failed: {e}")
```

---

## Deployment-Specific Configurations

### Development Environment

```bash
# Development storage configuration
export AWS_PROFILE="development"
export NEMO_CURATOR_CACHE_DIR="./cache"
export NEMO_CURATOR_DATA_DIR="./data"
export NEMO_CURATOR_OUTPUT_DIR="./output"

# Use local storage for development
export USE_LOCAL_STORAGE="true"
```

### Staging Environment

```bash
# Staging environment configuration
export AWS_PROFILE="staging"
export NEMO_CURATOR_CACHE_DIR="/shared/staging/cache"
export NEMO_CURATOR_DATA_DIR="s3://staging-bucket/data"
export NEMO_CURATOR_OUTPUT_DIR="s3://staging-bucket/output"

# Reduced performance settings for cost optimization
export AWS_S3_MAX_CONCURRENT_REQUESTS="5"
```

### Production Environment

```bash
# Production storage configuration
export AWS_PROFILE="production"
export NEMO_CURATOR_CACHE_DIR="/shared/prod/cache"
export NEMO_CURATOR_DATA_DIR="s3://prod-data-bucket/input"
export NEMO_CURATOR_OUTPUT_DIR="s3://prod-data-bucket/output"

# Optimized performance settings
export AWS_S3_MAX_CONCURRENT_REQUESTS="20"
export AWS_S3_MULTIPART_THRESHOLD="128MB"
export AWS_S3_MULTIPART_CHUNKSIZE="32MB"

# Security settings
export SSL_VERIFY="true"
export ENABLE_AUDIT_LOGGING="true"
```
