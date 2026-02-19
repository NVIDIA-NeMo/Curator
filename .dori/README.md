# Nemo Curator Local

Custom skill pack for Nemo Curator Local.

## Skills

Run `dori create my-skill` to add your first skill.

## Installation

```bash
# From GitHub (after publishing)
dori install github:owner/nemo-curator-local

# Or install locally
dori install ./path/to/nemo-curator-local
```

## Usage

```bash
# List available skills
dori list

# Use a skill
dori ::shortcut file.md
```

## Development

```bash
# Create a new skill
dori create my-skill --template audit

# Validate the pack
dori pack check

# Build REGISTRY.json
dori pack build
```

## License

MIT
