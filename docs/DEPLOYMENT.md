# Documentation Deployment to S3

This guide explains how to deploy the NeMo Curator documentation to the S3 hosting environment using both automated GitHub Actions and manual deployment methods.

## 🤖 Automated Deployment (Recommended)

The repository includes a GitHub Action that automatically deploys documentation:

### **Automatic Triggers:**
- **📦 Releases**: Automatically deploys when a new release is published
- **📝 Documentation Updates**: Deploys latest docs when changes are pushed to `main` branch (docs/ files, *.md, CHANGELOG.md)

### **Manual Triggers:**
Go to **Actions** → **Deploy Documentation** → **Run workflow** with options:
- **Version**: Override auto-detected version
- **Dry Run**: Preview changes without deploying  
- **Skip Backup**: Skip backing up current latest version
- **Skip Akamai**: Skip CDN cache purging (changes may take up to 1 hour to appear)

### **Required Setup:**
1. **AWS IAM Role**: Configure `AWS_ROLE_ARN` secret with S3 permissions
2. **GitHub Secrets**: Add the role ARN to repository secrets
3. **Akamai Credentials** (Optional): For automatic CDN cache purging
   - `AKAMAI_CLIENT_TOKEN`
   - `AKAMAI_CLIENT_SECRET` 
   - `AKAMAI_ACCESS_TOKEN`
   - `AKAMAI_HOST`

## 📋 Manual Deployment

For local testing or manual deployments, you can use the deployment script directly.

### Prerequisites

1. **AWS CLI**: Install and configure AWS CLI with appropriate credentials
   ```bash
   pip install awscli
   aws configure
   ```

2. **S3 Permissions**: Ensure your AWS credentials have the following permissions for the bucket `s3://brightspot-assets-prod/developer/docs/nemo/curator/`:
   - `s3:ListBucket`
   - `s3:GetObject`
   - `s3:PutObject`
   - `s3:DeleteObject`

3. **Documentation Environment**: Set up the docs environment
   ```bash
   make docs-env
   source .venv-docs/bin/activate
   ```

## Deployment Commands

### Quick Deployment

Build and deploy the latest GA documentation to S3:

```bash
make docs-deploy
```

This command:
1. Runs `make docs-publish-ga` to build the documentation
2. Auto-detects the version from `project.json`
3. Backs up the current `/latest/` to a versioned directory
4. Deploys the new build to `/latest/`
5. Updates version management files

### Dry Run

See what would be deployed without making changes:

```bash
make docs-deploy-dry-run
```

### Deploy Specific Version

Deploy with a specific version override:

```bash
make docs-deploy-version VERSION=1.2.3
```

### Deploy Without Cache Purging

Deploy to S3 but skip Akamai CDN cache purging:

```bash
make docs-deploy-no-cache
```

### Direct Script Usage

You can also use the deployment script directly for more control:

```bash
# Basic deployment
./scripts/deploy-docs.sh

# Dry run
./scripts/deploy-docs.sh --dry-run

# Specific version
./scripts/deploy-docs.sh --version 1.2.3

# Skip backing up current latest
./scripts/deploy-docs.sh --no-backup

# Help
./scripts/deploy-docs.sh --help
```

## S3 Structure

The deployment creates the following structure in S3:

```
s3://brightspot-assets-prod/developer/docs/nemo/curator/
├── latest/                    # Current release (always points to newest)
│   ├── index.html
│   ├── versions1.json         # Version switcher config
│   └── ...                    # All documentation files
├── 25.7/                      # Previous version (backed up)
│   ├── index.html
│   └── ...
├── 25.6/                      # Older version
│   └── ...
└── versions1.json             # Root-level version config
```

## Version Management

The system uses `versions1.json` for version switching in the documentation:

- **`/latest/versions1.json`**: Controls the version switcher widget
- **`/versions1.json`**: Root-level version index

When deploying:
1. The current `/latest/` is backed up to `/{previous-version}/`
2. The new build is deployed to `/latest/`
3. `versions1.json` is updated to include all versions
4. The current version is marked as `preferred: true` and points to `../latest`
5. Previous versions point to their respective directories (`../25.7`, etc.)

## Access URLs

After deployment, documentation is available at:

- **Latest**: https://docs.nvidia.com/nemo/curator/latest/
- **Specific version**: https://docs.nvidia.com/nemo/curator/25.7/
- **Version switcher**: Available in the top navigation bar

## Troubleshooting

### AWS Permissions Issues

If you get permission errors:

1. Verify your AWS credentials:
   ```bash
   aws sts get-caller-identity
   ```

2. Test S3 access:
   ```bash
   aws s3 ls s3://brightspot-assets-prod/developer/docs/nemo/curator/
   ```

### Build Issues

If the deployment fails due to build issues:

1. Ensure the documentation builds successfully:
   ```bash
   make docs-clean
   make docs-publish-ga
   ```

2. Check for build warnings/errors in the output

### Version Detection Issues

If version auto-detection fails:

1. Check that `docs/_build/html/project.json` exists
2. Manually specify version: `./scripts/deploy-docs.sh --version X.Y.Z`

## Development Workflow

Recommended workflow for documentation updates:

1. **Make changes** to documentation files
2. **Test locally**: `make docs-live-ga`
3. **Build for production**: `make docs-publish-ga`
4. **Review build output** in `docs/_build/html/`
5. **Dry run deployment**: `make docs-deploy-dry-run`
6. **Deploy**: `make docs-deploy`

## Cache Control

The deployment script sets appropriate cache headers:

- **Documentation files**: `public, max-age=3600` (1 hour)
- **Version files**: `public, max-age=300` (5 minutes)

This ensures reasonable caching while allowing version updates to propagate quickly.

## 🚀 Akamai CDN Cache Purging

After deploying to S3, the deployment system automatically purges the Akamai CDN cache to ensure users see updated content immediately.

### Setup Options

#### Option 1: Akamai CLI (Recommended)

Install and configure the Akamai CLI:

```bash
# Install Akamai CLI
npm install -g akamai-cli

# Configure credentials (creates ~/.edgerc)
akamai configure

# Test configuration
akamai purge --help
```

#### Option 2: API Credentials

Set environment variables for Fast Purge API:

```bash
export AKAMAI_CLIENT_TOKEN="your-client-token"
export AKAMAI_CLIENT_SECRET="your-client-secret"  
export AKAMAI_ACCESS_TOKEN="your-access-token"
export AKAMAI_HOST="your-akamai-host.luna.akamaiapis.net"
```

### What Gets Purged

The deployment automatically purges these URLs:

- `https://docs.nvidia.com/nemo/curator/latest/`
- `https://docs.nvidia.com/nemo/curator/latest/*` (all latest content)
- `https://docs.nvidia.com/nemo/curator/versions1.json` (version switcher)
- `https://docs.nvidia.com/nemo/curator/{VERSION}/` (specific version)
- `https://docs.nvidia.com/nemo/curator/{VERSION}/*` (all version content)

### Manual Cache Purging

If automatic purging fails or you need to purge manually:

```bash
# Using Akamai CLI
akamai purge invalidate "https://docs.nvidia.com/nemo/curator/latest/*"

# Using the deployment script without S3 update
./scripts/deploy-docs.sh --skip-backup --dry-run  # See what would be purged
```

### Troubleshooting Cache Issues

**Cache not clearing:**
1. Check Akamai CLI configuration: `akamai configure --list`
2. Verify API credentials have Fast Purge permissions
3. Check for rate limiting (max 1000 URLs per 5 minutes)

**Deployment works but cache purging fails:**
```bash
# Deploy without cache purging first
make docs-deploy-no-cache

# Then manually purge
akamai purge invalidate "https://docs.nvidia.com/nemo/curator/latest/*"
``` 