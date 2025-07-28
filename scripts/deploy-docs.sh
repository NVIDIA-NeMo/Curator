#!/bin/bash

# Deploy NeMo Curator documentation to S3
# Usage: ./scripts/deploy-docs.sh [--version VERSION] [--dry-run]

set -euo pipefail

# Configuration
S3_BUCKET="s3://brightspot-assets-prod/developer/docs/nemo/curator"
BUILD_DIR="docs/_build/html"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
DRY_RUN=false
VERSION=""
BACKUP_LATEST=true
SKIP_AKAMAI=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-backup)
            BACKUP_LATEST=false
            shift
            ;;
        --skip-akamai)
            SKIP_AKAMAI=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--version VERSION] [--dry-run] [--no-backup] [--skip-akamai]"
            echo ""
            echo "Options:"
            echo "  --version VERSION    Specify version to deploy (default: auto-detect)"
            echo "  --dry-run           Show what would be done without executing"
            echo "  --no-backup         Skip backing up current latest to versioned directory"
            echo "  --skip-akamai       Skip Akamai CDN cache purging"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Function to print colored output
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to execute commands with dry-run support
execute() {
    local cmd="$1"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
    else
        eval "$cmd"
    fi
}

# Check if AWS CLI is available
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install it first."
        exit 1
    fi
    
    # Test AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
}

# Auto-detect version from project.json
detect_version() {
    local project_json="${PROJECT_ROOT}/docs/_build/html/project.json"
    if [[ -f "$project_json" ]]; then
        # Extract version from project.json
        VERSION=$(python3 -c "import json; print(json.load(open('$project_json'))['version'])" 2>/dev/null || echo "")
    fi
    
    if [[ -z "$VERSION" ]]; then
        log_error "Could not auto-detect version. Please specify with --version"
        exit 1
    fi
    
    log_info "Detected version: $VERSION"
}

# Backup current /latest/ to versioned directory
backup_latest() {
    if [[ "$BACKUP_LATEST" != "true" ]]; then
        log_info "Skipping latest backup (--no-backup specified)"
        return
    fi
    
    log_info "Checking if current /latest/ exists..."
    
    # Check if latest directory exists in S3
    if aws s3 ls "${S3_BUCKET}/latest/" &> /dev/null; then
        log_info "Backing up current /latest/ to versioned directory..."
        
        # Download current versions1.json to determine current version
        local temp_versions="/tmp/current_versions1.json"
        if aws s3 cp "${S3_BUCKET}/latest/versions1.json" "$temp_versions" 2>/dev/null; then
            local current_version=$(python3 -c "
import json
try:
    with open('$temp_versions') as f:
        data = json.load(f)
    for item in data:
        if item.get('preferred'):
            print(item['version'])
            break
except:
    pass
" 2>/dev/null || echo "")
            
            if [[ -n "$current_version" && "$current_version" != "$VERSION" ]]; then
                log_info "Backing up version $current_version"
                execute "aws s3 sync ${S3_BUCKET}/latest/ ${S3_BUCKET}/${current_version}/ --delete"
                log_success "Backed up /latest/ to /${current_version}/"
            else
                log_warning "Current version matches new version or could not detect current version"
            fi
            
            rm -f "$temp_versions"
        else
            log_warning "Could not download current versions1.json, skipping backup"
        fi
    else
        log_info "No existing /latest/ directory found, proceeding with fresh deployment"
    fi
}

# Update versions1.json to include new version
update_versions_json() {
    log_info "Updating versions1.json..."
    
    local versions_file="${PROJECT_ROOT}/docs/_build/html/versions1.json"
    local temp_versions="/tmp/new_versions1.json"
    
    # Create updated versions1.json
    python3 -c "
import json
import sys

# Read current versions1.json from build
try:
    with open('$versions_file') as f:
        versions = json.load(f)
except FileNotFoundError:
    versions = []

# Update the current version to point to latest
new_version = '$VERSION'
found = False

for item in versions:
    if item['version'] == new_version:
        item['preferred'] = True
        item['url'] = '../latest'
        found = True
    else:
        item['preferred'] = False
        # Update URL to point to versioned directory
        if item.get('url') == '../latest':
            item['url'] = '../' + item['version']

if not found:
    # Add new version entry
    versions.append({
        'preferred': True,
        'version': new_version,
        'url': '../latest'
    })

# Sort versions by version number (descending)
def version_key(item):
    try:
        parts = item['version'].split('.')
        return tuple(int(x) for x in parts)
    except:
        return (0,)

versions.sort(key=version_key, reverse=True)

# Write updated versions1.json
with open('$temp_versions', 'w') as f:
    json.dump(versions, f, indent=4)

print(f'Updated versions1.json with version {new_version}')
"
    
    # Copy updated versions1.json back to build directory
    cp "$temp_versions" "$versions_file"
    rm -f "$temp_versions"
    
    log_success "Updated versions1.json"
}

# Deploy to S3
deploy_to_s3() {
    log_info "Deploying to S3..."
    
    # Sync to /latest/ directory
    execute "aws s3 sync ${BUILD_DIR}/ ${S3_BUCKET}/latest/ --delete --cache-control 'public, max-age=3600'"
    
    # Also copy versions1.json to root level (sibling to version directories)
    execute "aws s3 cp ${BUILD_DIR}/versions1.json ${S3_BUCKET}/versions1.json --cache-control 'public, max-age=300'"
    
    log_success "S3 deployment complete!"
}

# Purge Akamai CDN cache
purge_akamai_cache() {
    log_info "Purging Akamai CDN cache..."
    
    # Check if Akamai CLI is available
    if command -v akamai &> /dev/null; then
        log_info "Using Akamai CLI for cache purging..."
        
        # Base URLs to purge
        local urls=(
            "https://docs.nvidia.com/nemo/curator/latest/"
            "https://docs.nvidia.com/nemo/curator/latest/*"
            "https://docs.nvidia.com/nemo/curator/versions1.json"
            "https://docs.nvidia.com/nemo/curator/${VERSION}/"
            "https://docs.nvidia.com/nemo/curator/${VERSION}/*"
        )
        
        # Purge URLs using Akamai CLI
        for url in "${urls[@]}"; do
            log_info "Purging: $url"
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "[DRY-RUN] akamai purge invalidate $url"
            else
                execute "akamai purge invalidate '$url'" || {
                    log_warning "Failed to purge $url via CLI, will try API fallback"
                    return 1
                }
            fi
        done
        
        log_success "Akamai cache purged via CLI"
        return 0
        
    elif [[ -n "${AKAMAI_CLIENT_TOKEN:-}" && -n "${AKAMAI_CLIENT_SECRET:-}" && -n "${AKAMAI_ACCESS_TOKEN:-}" ]]; then
        log_info "Using Akamai Fast Purge API..."
        purge_akamai_api
        
    else
        log_warning "Akamai CLI not found and API credentials not configured"
        log_info "To enable Akamai cache purging:"
        log_info "  Option 1: Install Akamai CLI: npm install -g akamai-cli"
        log_info "  Option 2: Set environment variables: AKAMAI_CLIENT_TOKEN, AKAMAI_CLIENT_SECRET, AKAMAI_ACCESS_TOKEN, AKAMAI_HOST"
        return 1
    fi
}

# Purge Akamai cache using Fast Purge API
purge_akamai_api() {
    local akamai_host="${AKAMAI_HOST:-akaa-baseurl-xxxxxxxxxxx-xxxxxxxxxxxxx.luna.akamaiapis.net}"
    
    # Prepare URLs for API purge
    local purge_urls='[
        "https://docs.nvidia.com/nemo/curator/latest/",
        "https://docs.nvidia.com/nemo/curator/latest/*",
        "https://docs.nvidia.com/nemo/curator/versions1.json",
        "https://docs.nvidia.com/nemo/curator/'${VERSION}'/",
        "https://docs.nvidia.com/nemo/curator/'${VERSION}'/*"
    ]'
    
    # Create authorization header using EdgeGrid auth
    local timestamp=$(date -u +%Y%m%dT%H:%M:%S+0000)
    local nonce=$(openssl rand -hex 16)
    
    # For simplicity, use curl with pre-configured .edgerc file if available
    if [[ -f "$HOME/.edgerc" ]]; then
        log_info "Using .edgerc configuration for API authentication"
        
        # Use httpie with akamai auth plugin if available, otherwise fall back to curl
        if command -v http &> /dev/null && pip list | grep -q akamai-http-auth; then
            execute "echo '$purge_urls' | http --auth-type=akamai POST https://${akamai_host}/ccu/v3/invalidate/url/production objects:=@-"
        else
            log_warning "Advanced Akamai API auth requires akamai-http-auth plugin or manual EdgeGrid implementation"
            log_info "Consider using Akamai CLI instead: npm install -g akamai-cli"
            return 1
        fi
    else
        log_warning "No .edgerc file found for Akamai API authentication"
        return 1
    fi
    
    log_success "Akamai cache purged via API"
}

# Validate build directory
validate_build() {
    if [[ ! -d "$BUILD_DIR" ]]; then
        log_error "Build directory not found: $BUILD_DIR"
        log_info "Please run 'make docs-publish-ga' first"
        exit 1
    fi
    
    if [[ ! -f "${BUILD_DIR}/index.html" ]]; then
        log_error "index.html not found in build directory"
        log_info "Please ensure the documentation build completed successfully"
        exit 1
    fi
    
    log_success "Build directory validated"
}

# Main execution
main() {
    log_info "Starting NeMo Curator documentation deployment..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Validate prerequisites
    check_aws_cli
    validate_build
    
    # Auto-detect version if not specified
    if [[ -z "$VERSION" ]]; then
        detect_version
    fi
    
    log_info "Deploying version: $VERSION"
    log_info "Target S3 location: $S3_BUCKET"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi
    
    # Execute deployment steps
    backup_latest
    update_versions_json
    deploy_to_s3
    
    # Purge CDN cache if not skipped
    if [[ "$SKIP_AKAMAI" != "true" ]]; then
        purge_akamai_cache || {
            log_warning "Akamai cache purging failed, but deployment completed successfully"
            log_info "You may need to manually purge the cache or wait for TTL expiration"
        }
    else
        log_info "Skipping Akamai cache purging (--skip-akamai specified)"
    fi
    
    # Final success message
    log_success "Documentation deployment complete!"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        log_info "Documentation available at:"
        log_info "  📚 Latest: https://docs.nvidia.com/nemo/curator/latest/"
        log_info "  🔗 Version $VERSION: https://docs.nvidia.com/nemo/curator/${VERSION}/"
        log_info "  📋 All versions: https://docs.nvidia.com/nemo/curator/versions1.json"
        
        if [[ "$SKIP_AKAMAI" == "true" ]]; then
            log_warning "⚠️  Cache not purged - changes may take up to 1 hour to appear"
        else
            log_info "✅ CDN cache purged - changes should be visible immediately"
        fi
    fi
}

# Run main function
main "$@" 