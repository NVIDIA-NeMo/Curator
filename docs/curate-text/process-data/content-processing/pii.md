---
description: "Identify and remove Personal Identifiable Information from text data using NeMo Curator's privacy-preserving tools"
categories: ["how-to-guides"]
tags: ["pii-removal", "privacy", "compliance", "masking", "regex", "data-protection"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-format-pii)=
# PII Identification and Removal

Remove Personal Identifiable Information (PII) from your text data using NeMo Curator's tools and utilities.

## How it Works

NeMo Curator utilizes Ray to parallelize PII detection and removal tasks, allowing it to scale to terabytes of data easily. Ray can be deployed on various distributed compute environments (HPC clusters, Kubernetes, cloud platforms like AWS EKS and Google Cloud), with the current implementation supporting Ray on HPC clusters that use Slurm as the resource manager.

### Available PII Entity Types

The PII de-identification tool helps you remove the following sensitive data from your datasets:

| Entity Type | Description |
|-------------|-------------|
| `"PERSON"` | Names of individuals |
| `"EMAIL_ADDRESS"` | email addresses |
| `"ADDRESS"` | Street addresses |
| `"PHONE_NUMBER"` | Telephone numbers |
| `"IP_ADDRESS"` | Internet Protocol addresses |
| `"CREDIT_CARD"` | Credit/debit card numbers |
| `"US_SSN"` | US Social Security Numbers |
| `"DATE_TIME"` | Date information |
| `"URL"` | Web addresses |
| `"US_DRIVER_LICENSE"` | US driver's licenses |
| `"US_PASSPORT"` | US passport numbers |
| `"LOCATION"` | Location information |

### Redaction Format

The format of PII replacement depends on the `anonymize_action` parameter:

**Replace Action** (`anonymize_action="replace"`): PII entities are replaced with the entity type surrounded by double curly braces:

```text
Original text: "My name is John Smith and my email is john.smith@example.com"
Redacted text: "My name is {{PERSON}} and my email is {{EMAIL_ADDRESS}}"
```

**Redact Action** (`anonymize_action="redact"`, default): PII entities are completely removed:

```text
Original text: "My name is John Smith and my email is john.smith@example.com"  
Redacted text: "My name is  and my email is "
```

This consistent formatting makes it easy to identify processed content and understand what type of information was handled.

---

## Usage

Here's how to read, de-identify, and write a dataset:

::::{tab-set}

:::{tab-item} Python

```python
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import read_data, write_to_disk, get_client
from nemo_curator.utils.file_utils import get_batched_files
from nemo_curator.stages.text.modules import Modify

# Create a PII modifier
modifier = PiiModifier(
    language="en",
    supported_entities=["PERSON", "EMAIL_ADDRESS"],
    anonymize_action="replace",  # Default is "redact"
    batch_size=1000,
    device="gpu")

# Process files in batches
for file_names in get_batched_files(
        "book_dataset",
        "output_directory",
        "jsonl",
        32
):
    # Read a batch of documents
    source_data = read_data(file_names, file_type="jsonl", backend='pandas', add_filename=True)
    dataset = DocumentDataset(source_data)
    print(f"Dataset has {source_data.npartitions} partitions")

    # De-identify PII
    modify = Modify(modifier)
    modified_dataset = modify(dataset)
    
    # Write de-identified documents
    write_to_disk(modified_dataset.df,
                  "output_directory",
                  write_to_filename=True,
                  output_type="jsonl"
                  )
```
:::

:::{tab-item} CLI

The PII redaction module can also be invoked via a CLI interface:

```bash
deidentify --help
```

Run the command without arguments to see a complete list of supported options.

To launch the script from within a Slurm environment, you can modify and use the script:

```bash
examples/slurm/start-slurm.sh
```
:::

:::{tab-item} Async CLI

The `AsyncLLMPiiModifier` module can be invoked via the `nemo_curator/scripts/async_llm_pii_redaction.py` script, which provides a CLI-based interface. To see a complete list of options supported by the script, execute:

```bash
async_llm_pii_redaction --help
```

Here's an example of using the async CLI tool:

```bash
async_llm_pii_redaction \
  --input-data-dir /path/to/input \
  --output-data-dir /path/to/output \
  --base_url "http://0.0.0.0:8000/v1" \
  --api_key "your_api_key" \
  --max_concurrent_requests 20
```
:::

:::{tab-item} LLM CLI

The LLM-based PII redaction module can also be invoked via the `llm_pii_redaction` CLI tool. To see a complete list of options supported by the script, execute:

```bash
llm_pii_redaction --help
```

Here's an example of using the non-async CLI tool:

```bash
llm_pii_redaction \
  --input-data-dir /path/to/input \
  --output-data-dir /path/to/output \
  --base_url "http://0.0.0.0:8000/v1" \
  --api_key "your_api_key"
```

:::
::::

### Key Components Explained

| Component | Description |
|-----------|-------------|
| `PiiModifier` | The class responsible for PII de-identification with parameters: <br>- `language`: The language of the text (currently supports English)<br>- `supported_entities`: Types of PII to detect (see "Available PII Entity Types" section)<br>- `anonymize_action`: How to handle PII ("redact", "replace", "mask", "hash", or "custom"). Default is "redact"<br>- `batch_size`: Number of documents to process at once<br>- `device`: Processing device ("gpu" or "cpu") |
| `get_batched_files` | Retrieves batches of documents with parameters:<br>- First argument: Input directory<br>- Second argument: Output directory<br>- Third argument: File extension<br>- Fourth argument: Batch size (number of files) |
| `read_data` | Reads data from files using Ray with pandas data processing<br>- `add_filename=True`: Ensures output files have the same filename as input files |
| `DocumentDataset` | Creates the standard format for text datasets in NeMo Curator |
| `Modify` | Applies the PiiModifier to the dataset |
| `write_to_disk` | Writes the de-identified documents to disk |

### Anonymize Action Options

The `anonymize_action` parameter supports the following options:

- **"redact"** (default): Completely removes PII entities
- **"replace"**: Replaces PII with entity type labels in double curly braces (e.g., `{{PERSON}}`)
- **"mask"**: Masks PII with specified characters (additional parameters: `chars_to_mask`, `masking_char`)
- **"hash"**: Replaces PII with hash values (additional parameter: `hash_type` - "sha256", "sha512", or "md5")
- **"custom"**: Uses a custom function for anonymization (additional parameter: `lambda` function)

## Handling Interrupted Processing

The PII processing pipeline automatically supports resumable operations through the `get_batched_files` utility, which identifies and processes only files that haven't been processed yet.

## Custom System Prompts

When working with non-English text or when you want to customize how the LLM identifies PII entities, you can provide a custom system prompt. However, ensure that the JSON schema is included exactly as shown in the default system prompt.

```json
{
    "type": "array",
    "items": {
        "type": "object",
        "required": ["entity_type", "entity_text"],
        "properties": {
            "entity_type": {"type": "string"},
            "entity_text": {"type": "string"}
        }
    }
}
```

For reference, the default system prompt is:

```text
"You are an expert redactor. The user is going to provide you with some text.
Please find all personally identifying information from this text.
Return results according to this JSON schema: {JSON_SCHEMA}
Only return results for entities which actually appear in the text.
It is very important that you return the entity_text by copying it exactly from the input.
Do not perform any modification or normalization of the text.
The entity_type should be one of these: {PII_LABELS}"
```

`{PII_LABELS}` represents a comma-separated list of strings corresponding to the PII entity types you want to identify (for example, "name", "email", "ip_address", etc.).

When using a custom system prompt with non-English text, make sure to adapt the instructions while maintaining the exact JSON schema requirement. The LLM models will use this system prompt to guide their identification of PII entities.
