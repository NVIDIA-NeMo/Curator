# GitHub Actions Setup for Documentation Deployment

This guide helps you configure the required AWS permissions and GitHub secrets for automated documentation deployment.

## 🔐 AWS IAM Setup

### 1. Create IAM Role for GitHub Actions

Create an IAM role with the following trust policy (replace `YOUR_ORG` and `YOUR_REPO`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/YOUR_REPO:ref:refs/heads/main"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": [
            "repo:YOUR_ORG/YOUR_REPO:ref:refs/heads/main",
            "repo:YOUR_ORG/YOUR_REPO:ref:refs/tags/*"
          ]
        }
      }
    }
  ]
}
```

### 2. Attach S3 Permissions Policy

Create and attach this policy to the role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:PutObjectAcl"
      ],
      "Resource": [
        "arn:aws:s3:::brightspot-assets-prod",
        "arn:aws:s3:::brightspot-assets-prod/developer/docs/nemo/curator/*"
      ]
    }
  ]
}
```

### 3. Note the Role ARN

Copy the role ARN (format: `arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME`)

## 🌐 Akamai API Setup (Optional)

For automatic CDN cache purging, you'll need Akamai API credentials.

### 1. Create API Client in Akamai Control Center

1. Log in to [Akamai Control Center](https://control.akamai.com)
2. Go to **Identity & Access Management** → **API users**
3. Click **New API user for me**
4. Name: `github-docs-deployment`
5. Select APIs: **Fast Purge** (for cache invalidation)
6. Grant access to your property/group
7. Note the credentials:
   - Client Token
   - Client Secret  
   - Access Token
   - Host (your luna.akamaiapis.net endpoint)

### 2. Test API Access

```bash
# Install Akamai CLI locally to test
npm install -g akamai-cli

# Configure with your credentials
akamai configure

# Test Fast Purge access
akamai purge --help
```

## 🔑 GitHub Repository Secrets

### Add Required Secrets

1. Go to your repository **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add these secrets:

**Required:**
- **Name**: `AWS_ROLE_ARN`
  - **Value**: The role ARN from step 3 above

**Optional (for Akamai CDN cache purging):**
- **Name**: `AKAMAI_CLIENT_TOKEN`
  - **Value**: Your Akamai API client token
- **Name**: `AKAMAI_CLIENT_SECRET`
  - **Value**: Your Akamai API client secret  
- **Name**: `AKAMAI_ACCESS_TOKEN`
  - **Value**: Your Akamai API access token
- **Name**: `AKAMAI_HOST`
  - **Value**: Your Akamai API host (e.g., `akaa-xxxxx.luna.akamaiapis.net`)

## 🏗️ GitHub Environment (Optional)

For additional security, you can set up a `production` environment:

1. Go to **Settings** → **Environments** → **New environment**
2. Name: `production`
3. Add protection rules:
   - ✅ Required reviewers (recommended for production deployments)
   - ✅ Restrict pushes to protected branches
4. Add environment secret `AWS_ROLE_ARN` (same value as repository secret)

If you set up the environment, uncomment this line in `.github/workflows/deploy-docs.yml`:
```yaml
environment: production
```

## 🧪 Testing the Setup

### Test with Dry Run

1. Go to **Actions** → **Deploy Documentation**
2. Click **Run workflow**
3. Check **dry-run** option
4. Click **Run workflow**

This will test the build and AWS permissions without making changes.

### Test Automatic Triggers

1. **Release trigger**: Create a new release to test automatic deployment
2. **Main branch trigger**: Push documentation changes to main branch

## 🔧 Troubleshooting

### Common Issues

**❌ "Error: Could not assume role"**
- Check that the IAM role trust policy includes your repository
- Verify the role ARN in GitHub secrets is correct
- Ensure the repository has the correct OIDC provider configured

**❌ "Access Denied" S3 errors**
- Verify the IAM role has the correct S3 permissions
- Check that the bucket name and paths are correct in the policy

**❌ Documentation build fails**
- Check that all required dependencies are available
- Review the build logs for specific error messages
- Test the build locally with `make docs-publish-ga`

**❌ Akamai cache purging fails**
- Verify Akamai API credentials are correctly set in GitHub secrets
- Check that the API client has Fast Purge permissions
- Review rate limiting (max 1000 URLs per 5 minutes)
- Test credentials locally with `akamai purge --help`

### Debug Steps

1. **Check AWS credentials in the workflow**:
   ```bash
   aws sts get-caller-identity
   ```

2. **Test S3 access**:
   ```bash
   aws s3 ls s3://brightspot-assets-prod/developer/docs/nemo/curator/
   ```

3. **Verify documentation build**:
   ```bash
   make docs-env
   make docs-publish-ga
   ls -la docs/_build/html/
   ```

## 🔄 Integration with Release Workflow

To automatically deploy documentation as part of your release process, you can modify `.github/workflows/release.yml` to trigger the documentation deployment after a successful release.

Add this job to the release workflow:

```yaml
deploy-docs:
  needs: release
  if: success() && !inputs.dry-run
  uses: ./.github/workflows/deploy-docs.yml
  secrets: inherit
```

This ensures documentation is always updated when you publish a new release. 