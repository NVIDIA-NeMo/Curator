.. _data-curator-pii:

======================================
PII Identification and Removal
======================================

--------------------------------------
Background
--------------------------------------

The purpose of a personal identifiable information (PII) de-identification tool is to help scrub sensitive data out of datasets.
Some examples of sensitive data types that are currently supported by the tools include:

- Name
- Email Address
- Physical Address
- Phone Numbers
- IP Address
- Credit/Debit Card Numbers
- US Social Security Numbers
- Dates

NeMo Curator provides several tools for PII identification and removal.

The ``PiiModifier`` class utilizes the `Presidio <https://microsoft.github.io/presidio/>`_ library to identify and redact PII in text.
`Dask <https://dask.org>`_ is used to parallelize tasks and hence it can be used to scale up to terabytes of data easily.

The ``LLMPiiModifier`` and ``AsyncLLMPiiModifier`` classes utilize LLM models to identify PII in text.
Using `NVIDIA NIM <https://developer.nvidia.com/nim>`_, the ``LLMPiiModifier`` and ``AsyncLLMPiiModifier`` classes can submit prompts to LLMs, such as Meta's `Llama 3.1 <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>`_, for PII identification.
The LLM's response is then parsed and used to redact the PII in the text.
We found that LLM-based PII redaction (using Llama-3.1-70B) outperformed Presidio by 26% on core PII categories.
Evaluations were run on the `Gretel PII masking <https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1>`_ dataset and the `Text Anonymization Benchmark <https://arxiv.org/abs/2202.00443>`_ dataset.
Please note that the LLM prompts were written in English, and the datasets used for evaluation were in English as well.

-----------------------------------------
Usage
-----------------------------------------

############################
PII redaction using Presidio
############################

Imagine we've a "books" dataset stored in the following structure:
::

    books_dataset/
        books_00.jsonl
        books_01.jsonl
        books_02.jsonl

You could read, de-identify the dataset, and write it to an output directory using the following approach:

.. code-block:: python

    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.distributed_utils import read_data, write_to_disk, get_client
    from nemo_curator.utils.file_utils import get_batched_files
    from nemo_curator.modules.modify import Modify
    from nemo_curator.modifiers.pii_modifier import PiiModifier

    client = get_client(cluster_type="cpu")

    modifier = PiiModifier(
        language="en",
        supported_entities=["PERSON", "EMAIL_ADDRESS"],
        anonymize_action="replace",
        batch_size=1000,
        device="gpu",
    )

    for file_names in get_batched_files(
        "book_dataset",
        "output_directory",
        "jsonl",
        32
    ):
        dataset = DocumentDataset.read_json(file_names, backend="pandas", add_filename=True)

        modify = Modify(modifier)
        modified_dataset = modify(dataset)

        modified_dataset.to_json("output_directory", write_to_filename=True)

Let's walk through this code line by line:

* ``modifier = PiiModifier(...)`` creates an instance of ``PiiModifier`` class that's responsible for PII de-identification.
* ``supported_entities=["PERSON", "EMAIL_ADDRESS"]`` specifies the PII entities that the ``PiiModifier`` will identify and redact. By default, the ``PiiModifier`` will identify and redact the following entities:
::

    [
        "ADDRESS",
        "CREDIT_CARD",
        "EMAIL_ADDRESS",
        "DATE_TIME",
        "IP_ADDRESS",
        "LOCATION",
        "PERSON",
        "URL",
        "US_SSN",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "PHONE_NUMBER",
    ]

* ``for file_names in get_batched_files`` retrieves a batch of 32 documents from the ``book_dataset`` directory.
* ``dataset = DocumentDataset.read_json(...)`` reads the data from the batch of files using Dask using Pandas as the backend. ``DocumentDataset`` is the standard format for text datasets in NeMo Curator. The ``add_filename`` argument ensures that the output files have the same filename as the input files.
* ``modify = Modify(modifier)`` creates an instance of the ``Modify`` class. This class can take any modifier as an argument.
* ``modified_dataset = modify(dataset)`` modifies the data in the dataset by performing the PII de-identification based upon the passed parameters.
* ``modified_dataset.to_json(...)`` writes the de-identified documents to disk.

The ``PiiModifier`` module can be invoked via the ``nemo_curator/scripts/find_pii_and_deidentify.py`` script which provides a CLI-based interface. To see a complete list of options supported by the script, execute:

``find_pii_and_deidentify --help``

To launch the script from within a Slurm environment, the script ``examples/slurm/start-slurm.sh`` can be modified and used.

############################
LLM-based PII redaction
############################

Let's again consider the "books" dataset stored in the following structure:
::

    books_dataset/
        books_00.jsonl
        books_01.jsonl
        books_02.jsonl

In order to use the ``AsyncLLMPiiModifier`` class, you will need to set up a NIM endpoint with a ``base_url`` and (optionally) an ``api_key``.
For instructions on how to set up a NIM endpoint, please refer to the `NIM Getting Started page <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html>`_.

After setting up a NIM endpoint, you can read, de-identify the dataset, and write it to an output directory with:

.. code-block:: python

    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.distributed_utils import read_data, write_to_disk, get_client
    from nemo_curator.utils.file_utils import get_batched_files
    from nemo_curator.modules.modify import Modify
    from nemo_curator.modifiers.async_llm_pii_modifier import AsyncLLMPiiModifier

    client = get_client(cluster_type="cpu")

    modifier = AsyncLLMPiiModifier(
        # Endpoint for the user's NIM
        base_url="http://0.0.0.0:8000/v1",
        api_key="API KEY (if needed)",
        model="meta/llama-3.1-70b-instruct",
        # The user may provide custom labels for PII entities if desired
        pii_labels=["name", "email"],
        language="en",
        max_concurrent_requests=10,
    )

    for file_names in get_batched_files(
        "book_dataset",
        "output_directory",
        "jsonl",
        32
    ):
        dataset = DocumentDataset.read_json(file_names, backend="pandas", add_filename=True)

        modify = Modify(modifier)
        modified_dataset = modify(dataset)

        modified_dataset.to_json("output_directory", write_to_filename=True)

Let's walk through this code line by line:

* ``modifier = AsyncLLMPiiModifier(...)`` creates an instance of ``AsyncLLMPiiModifier`` class that's responsible for PII de-identification.
* ``model="meta/llama-3.1-70b-instruct"`` specifies the LLM model to use. ``AsyncLLMPiiModifier`` requires LLMs that support the OpenAI chat message format (that is, system, user, and assistant roles). Examples include instruct-tuned versions of Meta's LLaMA ("meta/llama-3.1-70b-instruct" is the default model used by ``AsyncLLMPiiModifier``) and OpenAI's GPT models (such as "gpt-4o").
* ``pii_labels=["name", "email"]`` specifies the PII entities that the ``AsyncLLMPiiModifier`` will identify and redact. By default, the ``AsyncLLMPiiModifier`` will identify and redact the following entities:
::

    [
        "medical_record_number",
        "location",
        "address",
        "ssn",
        "date_of_birth",
        "date_time",
        "name",
        "email",
        "customer_id",
        "employee_id",
        "phone_number",
        "ip_address",
        "credit_card_number",
        "user_name",
        "device_identifier",
        "bank_routing_number",
        "company_name",
        "unique_identifier",
        "biometric_identifier",
        "account_number",
        "certificate_license_number",
        "license_plate",
        "vehicle_identifier",
        "api_key",
        "password",
        "health_plan_beneficiary_number",
        "national_id",
        "tax_id",
        "url",
        "swift_bic",
        "cvv",
        "pin",
    ]

* We recommend setting ``max_concurrent_requests=10`` to avoid overwhelming the NIM endpoint. However, the user can set this to a higher or lower value depending on their use case.
* ``for file_names in get_batched_files`` retrieves a batch of 32 documents from the ``book_dataset`` directory.
* ``dataset = DocumentDataset.read_json(...)`` reads the data from the batch of files using Dask using Pandas as the backend. ``DocumentDataset`` is the standard format for text datasets in NeMo Curator. The ``add_filename`` argument ensures that the output files have the same filename as the input files.
* ``modify = Modify(modifier)`` creates an instance of the ``Modify`` class. This class can take any modifier as an argument.
* ``modified_dataset = modify(dataset)`` modifies the data in the dataset by performing the PII de-identification based upon the passed parameters.
* ``modified_dataset.to_json(...)`` writes the de-identified documents to disk.

Redaction Format
~~~~~~~~~~~~~~~~~~~~~~

When PII entities are identified, they're replaced with the entity type surrounded by double curly braces. For example:

.. code-block:: text

    Original text: "My name is John Smith and my email is john.smith@example.com"
    Redacted text: "My name is {{name}} and my email is {{email}}"

This consistent formatting makes it easy to identify redacted content and understand what type of information was removed.

Command-Line Usage
~~~~~~~~~~~~~~~~~~~~~~

The ``AsyncLLMPiiModifier`` module can be invoked via the ``nemo_curator/scripts/async_llm_pii_redaction.py`` script which provides a CLI-based interface. To see a complete list of options supported by the script, execute:

.. code-block:: bash

    async_llm_pii_redaction --help

Here's an example of using the async CLI tool:

.. code-block:: bash

    async_llm_pii_redaction \
      --input-data-dir /path/to/input \
      --output-data-dir /path/to/output \
      --base_url "http://0.0.0.0:8000/v1" \
      --api_key "your_api_key" \
      --max_concurrent_requests 20

Above, we recommend using the ``AsyncLLMPiiModifier`` because it utilizes ``AsyncOpenAI`` to submit multiple concurrent requests to the NIM endpoint.
The higher the ``max_concurrent_requests`` is, the faster the ``AsyncLLMPiiModifier`` will be, but the user should be mindful to avoid overwhelming the NIM endpoint.
Alternatively, the user can use the ``LLMPiiModifier`` class which doesn't utilize ``AsyncOpenAI`` and hence submits requests serially.
Use of the ``LLMPiiModifier`` class is the same as the ``AsyncLLMPiiModifier`` class except that the ``max_concurrent_requests`` parameter isn't used.

For example:

.. code-block:: python

    from nemo_curator.modifiers.llm_pii_modifier import LLMPiiModifier

    modifier = LLMPiiModifier(
        # Endpoint for the user's NIM
        base_url="http://0.0.0.0:8000/v1",
        api_key="API KEY (if needed)",
        model="meta/llama-3.1-70b-instruct",
        # The user may provide custom labels for PII entities if desired
        pii_labels=["name", "email"],
        language="en",
    )

The ``LLMPiiModifier`` module can be invoked via the ``nemo_curator/scripts/llm_pii_redaction.py`` script which provides a CLI-based interface. To see a complete list of options supported by the script, execute:

.. code-block:: bash

    llm_pii_redaction --help

Example of using the non-async CLI tool:

.. code-block:: bash

    llm_pii_redaction \
      --input-data-dir /path/to/input \
      --output-data-dir /path/to/output \
      --base_url "http://0.0.0.0:8000/v1" \
      --api_key "your_api_key"

Custom System Prompts
~~~~~~~~~~~~~~~~~~~~~~

When working with non-English text or when you want to customize how the LLM identifies PII entities, you can provide a custom system prompt. However, ensure that the JSON schema is included exactly as shown in the default system prompt.

.. code-block:: json

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

For reference, the default system prompt is:

.. code-block:: text

    "You are an expert redactor. The user is going to provide you with some text.
    Please find all personally identifying information from this text.
    Return results according to this JSON schema: {JSON_SCHEMA}
    Only return results for entities which actually appear in the text.
    It is very important that you return the entity_text by copying it exactly from the input.
    Do not perform any modification or normalization of the text.
    The entity_type should be one of these: {PII_LABELS}"

``{PII_LABELS}`` represents a comma-separated list of strings corresponding to the PII entity types you want to identify (for example, "name", "email", "ip_address", etc.).

When using a custom system prompt with non-English text, make sure to adapt the instructions while maintaining the exact JSON schema requirement. The LLM models will use this system prompt to guide their identification of PII entities.

############################
Resuming from interruptions
############################

It can be helpful to track which documents in a dataset have already been processed so that long curation jobs can be resumed if they're interrupted.
NeMo Curator provides a utility for easily tracking which dataset shards have already been processed.
A call to ``get_batched_files`` will return an iterator over the files that have yet to be processed by a modifier such as ``PiiModifier``.
When you re-run the code example provided above, NeMo Curator ensures that only unprocessed files are processed by the PII module.
