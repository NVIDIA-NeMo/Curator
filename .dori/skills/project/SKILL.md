---
name: project
description: Set up and organize a data curation project. Use when the user is starting a new curation project, asks about directory structure, file organization, naming conventions, or how to organize their data files.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  type: setup
---

# Project Setup Skill

Help users organize their data curation project with proper directory structure, file formats, and naming conventions.

## When This Skill Applies

- User is starting a new curation project
- User asks "how should I organize my files?"
- User asks about directory structure
- User asks about file formats (JSONL vs Parquet)
- User asks about naming conventions
- User has multiple datasets to curate

## Skill Workflow

### Step 1: Understand the Project

Ask the user:
1. What data are you curating? (web crawl, internal docs, etc.)
2. How many datasets? (one vs multiple sources)
3. What's your end goal? (LLM training, RAG, etc.)
4. Are you on a single machine or cluster?

### Step 2: Recommend Project Structure

#### Basic Single-Dataset Project

```
my-curation-project/
├── data/
│   ├── raw/                    # Original input data
│   │   └── crawl_2024.jsonl
│   ├── filtered/               # After heuristic filtering
│   │   └── crawl_2024_filtered.jsonl
│   ├── classified/             # After ML classification
│   │   └── crawl_2024_classified.jsonl
│   └── final/                  # Final curated output
│       └── crawl_2024_curated.jsonl
├── cache/                      # Intermediate files (dedup, etc.)
│   └── dedup_cache/
├── logs/                       # Processing logs
├── configs/                    # Pipeline configurations
│   └── filter_config.yaml
├── scripts/                    # Custom pipeline scripts
│   └── run_curation.py
└── README.md                   # Project documentation
```

#### Multi-Dataset Project

```
my-curation-project/
├── datasets/
│   ├── common-crawl/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── config.yaml
│   ├── wikipedia/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── config.yaml
│   └── internal-docs/
│       ├── raw/
│       ├── processed/
│       └── config.yaml
├── cache/                      # Shared cache (dedup)
├── merged/                     # Combined final output
├── logs/
└── README.md
```

#### Cluster/Large-Scale Project

```
my-curation-project/
├── data/
│   ├── raw/                    # Can be remote (S3, GCS, etc.)
│   │   ├── shard_0000.parquet
│   │   ├── shard_0001.parquet
│   │   └── ...
│   └── processed/
│       ├── shard_0000.parquet
│       └── ...
├── cache/                      # Local or distributed storage
│   ├── minhash/                # Dedup intermediate files
│   └── embeddings/             # Embedding cache
├── configs/
│   └── cluster_config.yaml
└── scripts/
    ├── submit_job.sh
    └── run_stage.py
```

### Step 3: Explain File Formats

#### JSONL (JSON Lines)

Best for:
- Smaller datasets (< 100GB)
- Human-readable debugging
- Simple pipelines

```json
{"id": "doc_001", "text": "Document content here...", "url": "https://..."}
{"id": "doc_002", "text": "Another document...", "url": "https://..."}
```

Required fields:
- `text` (or custom field name): The content to process
- `id` (optional but recommended): Unique document identifier

Optional metadata:
- `url`: Source URL
- `timestamp`: Collection time
- `language`: Detected language
- Any custom fields you want to preserve

#### Parquet

Best for:
- Large datasets (> 100GB)
- Columnar queries
- Cluster processing
- Preserving data types

Same schema as JSONL, but binary columnar format.

### Step 4: Naming Conventions

#### Dataset Naming
```
{source}_{version}_{date}
```
Examples:
- `commoncrawl_cc2024_20240115`
- `wikipedia_en_20240101`
- `internal_docs_v2`

#### Stage Output Naming
```
{dataset}_{stage}.{format}
```
Examples:
- `commoncrawl_filtered.jsonl`
- `commoncrawl_classified.jsonl`
- `commoncrawl_deduped.jsonl`
- `commoncrawl_final.parquet`

#### Version Tracking
```
{dataset}_{stage}_v{N}.{format}
```
For iterative refinement:
- `commoncrawl_filtered_v1.jsonl`
- `commoncrawl_filtered_v2.jsonl` (adjusted thresholds)

### Step 5: Scaffold the Project

Generate the directory structure:

```bash
# Create project structure
mkdir -p my-curation-project/{data/{raw,filtered,classified,final},cache,logs,configs,scripts}

# Create README
cat > my-curation-project/README.md << 'EOF'
# Data Curation Project

## Overview
[Describe your curation goals]

## Data Sources
- Source 1: [description]

## Pipeline Stages
1. Heuristic filtering (CPU)
2. Quality classification (GPU)
3. Fuzzy deduplication (GPU)

## Running
```bash
# Stage 1: Filter
python scripts/run_filter.py --input data/raw/data.jsonl --output data/filtered/

# Stage 2: Classify
python scripts/run_classify.py --input data/filtered/ --output data/classified/

# Stage 3: Deduplicate
python scripts/run_dedup.py --input data/classified/ --output data/final/
```

## Configuration
See `configs/` for stage configurations.
EOF

echo "Project created: my-curation-project/"
```

### Step 6: Help with Data Preparation

If user's data isn't in the right format:

#### Convert CSV to JSONL
```python
import csv
import json

with open('data.csv') as f_in, open('data.jsonl', 'w') as f_out:
    reader = csv.DictReader(f_in)
    for row in reader:
        # Ensure 'text' field exists
        if 'content' in row and 'text' not in row:
            row['text'] = row.pop('content')
        f_out.write(json.dumps(row) + '\n')
```

#### Convert directory of text files to JSONL
```python
import json
from pathlib import Path

with open('data.jsonl', 'w') as f_out:
    for txt_file in Path('text_files/').glob('*.txt'):
        doc = {
            'id': txt_file.stem,
            'text': txt_file.read_text(),
        }
        f_out.write(json.dumps(doc) + '\n')
```

#### Add IDs to existing JSONL
```python
import json
import uuid

with open('data.jsonl') as f_in, open('data_with_ids.jsonl', 'w') as f_out:
    for line in f_in:
        doc = json.loads(line)
        if 'id' not in doc:
            doc['id'] = str(uuid.uuid4())
        f_out.write(json.dumps(doc) + '\n')
```

## Example Interactions

### Example 1: New user starting fresh
```
User: I want to curate some web data for LLM training. How should I set up my project?

Agent: I'll help you set up a curation project. A few questions:

1. How big is your data? (GB/TB)
2. Is it already in JSONL or Parquet format?
3. Single machine or cluster?

User: About 50GB of JSONL files, single machine with GPU

Agent: Perfect. Here's your recommended structure:

```
web-curation/
├── data/
│   ├── raw/          # Put your JSONL files here
│   ├── filtered/     # After heuristic filtering (CPU)
│   ├── classified/   # After quality scoring (GPU)
│   └── final/        # Deduplicated output
├── cache/            # For deduplication intermediate files
├── logs/
└── scripts/
```

Let me scaffold this for you:
[Creates directories and README]

Now copy your JSONL files to `data/raw/`. What's the text field called in your data?
```

### Example 2: Data format questions
```
User: My data is in CSV format with a 'content' column. Will that work?

Agent: NeMo Curator expects JSONL or Parquet with a `text` field by default.

I can help you convert. Here's a script:

```python
import csv
import json

with open('data.csv') as f_in, open('data.jsonl', 'w') as f_out:
    for row in csv.DictReader(f_in):
        row['text'] = row.pop('content')  # Rename to 'text'
        f_out.write(json.dumps(row) + '\n')
```

Or you can use `--text-field content` in most NeMo Curator stages to use your existing column name.
```

### Example 3: Multi-dataset project
```
User: I have data from Common Crawl, Wikipedia, and our internal docs. How should I organize?

Agent: For multiple datasets, I recommend keeping them separate until the final merge:

```
multi-source-curation/
├── datasets/
│   ├── common-crawl/
│   │   ├── raw/
│   │   └── processed/
│   ├── wikipedia/
│   │   ├── raw/
│   │   └── processed/
│   └── internal/
│       ├── raw/
│       └── processed/
├── merged/           # Combined final output
└── cache/            # Shared dedup cache
```

Benefits:
- Different configs per dataset (Wikipedia needs different filters than Common Crawl)
- Track provenance (which source each doc came from)
- Run dedup across all sources at the end

Want me to scaffold this structure?
```

## Best Practices

### 1. Always Keep Raw Data
Never overwrite your original files. Always write to a new location.

### 2. Use Meaningful IDs
If your data doesn't have IDs, add them before processing:
```python
doc['id'] = f"{source}_{hash(doc['text'])[:16]}"
```

### 3. Track Provenance
Add metadata about processing:
```python
doc['_curation'] = {
    'source': 'common-crawl',
    'filtered_at': '2024-01-15',
    'filter_version': 'v1',
}
```

### 4. Use Parquet for Large Data
For datasets > 100GB, Parquet is faster and smaller:
```python
import pandas as pd
df = pd.read_json('data.jsonl', lines=True)
df.to_parquet('data.parquet')
```

### 5. Shard Large Files
For cluster processing, shard into ~1GB files:
```bash
# Split 100GB file into 1GB shards
split -b 1G large_file.jsonl shard_
for f in shard_*; do mv "$f" "$f.jsonl"; done
```
