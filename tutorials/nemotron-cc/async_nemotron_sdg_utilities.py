import asyncio
import logging

from openai import OpenAI
from transformers import AutoTokenizer

from nemo_curator import (
    DocumentJoiner,
    DocumentSplitter,
    Filter,
    Modify,
    ScoreFilter,
    Sequential,
)
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import SubstringFilter, TokenCountFilter
from nemo_curator.modifiers import (
    LineRemover,
    MarkdownRemover,
    QuotationRemover,
    Slicer,
)
from nemo_curator.services import AsyncOpenAIClient
from nemo_curator.synthetic import (
    NemotronCCDiverseQAPostprocessor,
    NemotronCCKnowledgeListPostprocessor,
)
from nemo_curator.synthetic.async_nemotron_cc import AsyncNemotronCCGenerator
from nemo_curator.synthetic.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)


def get_prefix_token_count(tokenizer: AutoTokenizer, system_prompt: str, user_prompt_template: str) -> int:
    user_prompt = user_prompt_template.format(document="placeholder")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prefix_tokens = tokenizer.apply_chat_template(messages)

    return len(prefix_tokens)


def build_preprocessing_pipeline(  # noqa: PLR0913
    tokenizer: AutoTokenizer,
    text_field: str,
    system_prompt: str,
    user_prompt_template: str,
    min_document_tokens: int,
    min_segment_tokens: int,
    max_input_tokens: int,
) -> Sequential:
    # Construct filters for document filtering
    document_token_count_filter = TokenCountFilter(tokenizer=tokenizer, min_tokens=min_document_tokens)

    # Construct filters for segment filtering
    prefix_token_count = get_prefix_token_count(tokenizer, system_prompt, user_prompt_template)
    max_segment_tokens = max_input_tokens - prefix_token_count - 2
    long_segment_token_count_filter = TokenCountFilter(tokenizer=tokenizer, max_tokens=max_segment_tokens)
    short_segment_token_count_filter = TokenCountFilter(tokenizer=tokenizer, min_tokens=min_segment_tokens)

    return Sequential(
        [
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=text_field,
                score_field="document_token_count",
                score_type=int,
            ),
            # Split documents into segments
            DocumentSplitter(separator="\n", text_field=text_field, segment_id_field="segment_id"),
            # Filter out segments that are too long
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=text_field,
                score_field="segment_token_count",
                score_type=int,
            ),
            # Join adjacent short segments
            DocumentJoiner(
                separator="\n",
                text_field=text_field,
                segment_id_field="segment_id",
                document_id_field="id",
                max_length=max_segment_tokens,
                length_field="segment_token_count",
                drop_segment_id_field=False,
            ),
            # Filter out segments that are too short even after joining
            Filter(
                short_segment_token_count_filter.keep_document,
                filter_field="segment_token_count",
            ),
        ]
    )


def build_wikipedia_postprocessing_pipeline(tokenizer: AutoTokenizer, rephrased_field: str) -> Sequential:
    max_rephrased_tokens = 510
    min_document_tokens = 50

    long_segment_token_count_filter = TokenCountFilter(tokenizer=tokenizer, max_tokens=max_rephrased_tokens)
    document_token_count_filter = TokenCountFilter(tokenizer=tokenizer, min_tokens=min_document_tokens)
    return Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=rephrased_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=rephrased_field),
            # Remove documents not starting with the specified prefix
            ScoreFilter(
                SubstringFilter(substring="Here is a paraphrased version:", position="prefix"),
                text_field=rephrased_field,
                score_field="substring",
                score_type=int,
            ),
            # Remove the paraphrase prefix
            Modify(
                Slicer(
                    left="Here is a paraphrased version:",
                    include_left=False,
                    strip=True,
                ),
                text_field=rephrased_field,
            ),
            # Remove quotation marks
            Modify(QuotationRemover(), text_field=rephrased_field),
            # Concat paragraphs belonging to the same document
            DocumentJoiner(
                separator="\n",
                text_field=rephrased_field,
                segment_id_field="segment_id",
                document_id_field="id",
                drop_segment_id_field=False,
            ),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=rephrased_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )


async def wikipedia_rephraser(
    dataset: DocumentDataset,
    text_field: str,
    openai_client: OpenAI,
    tokenizer: AutoTokenizer,
    api_model_name: str,
) -> DocumentDataset:
    client = AsyncOpenAIClient(openai_client)
    generator = AsyncNemotronCCGenerator(client)
    rephrased_field = "rephrased"
    config = {
        "MIN_DOCUMENT_TOKENS": 30,
        "MIN_SEGMENT_TOKENS": 10,
        "MAX_INPUT_TOKENS": 512,
        "MAX_OUTPUT_TOKENS": 512,
        "TOP_K": 0,
        "TOP_P": 0.9,
        "END_STRINGS": "['</s>']",
        "TEMPERATURE": 0.5,
    }

    preprocessing_pipeline = build_preprocessing_pipeline(
        tokenizer,
        text_field,
        NEMOTRON_CC_SYSTEM_PROMPT,
        WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
        config["MIN_DOCUMENT_TOKENS"],
        config["MIN_SEGMENT_TOKENS"],
        config["MAX_INPUT_TOKENS"],
    )

    print("Running Wikipedia rephraser preprocessing pipeline")
    dataset = preprocessing_pipeline(dataset)

    print("Computing pandas dataframe")
    pandas_df = dataset.df.compute()

    async def process_text(text: str) -> str | None:
        try:
            rewritten_text = await generator.rewrite_to_wikipedia_style(
                text,
                api_model_name,
                model_kwargs={
                    "top_p": config["TOP_P"],
                    "stop": config["END_STRINGS"],
                    "max_tokens": config["MAX_OUTPUT_TOKENS"],
                    "temperature": config["TEMPERATURE"],
                },
            )
            print(rewritten_text)
            return rewritten_text[0]
        except asyncio.TimeoutError:
            logging.exception("Request timed out after 30 seconds")  # noqa: LOG015
            return None
        except Exception:
            logging.exception("Error processing text")  # noqa: LOG015
            return None

    # Create tasks for all texts with progress tracking
    print("Creating Async tasks for all texts")
    tasks = [process_text(text) for text in pandas_df[text_field]]

    # Run all tasks concurrently with asyncio
    print("Starting concurrent processing of texts")
    try:
        # Remove return_exceptions from tqdm_asyncio.gather() since it's not supported
        from tqdm.asyncio import tqdm_asyncio

        rewritten_texts = await tqdm_asyncio.gather(*tasks, desc="Processing texts")
    except Exception as e:  # noqa: BLE001
        print(f"Error during async gathering: {e!s}")
        rewritten_texts = [None] * len(tasks)

    # Handle any exceptions that occurred during gathering
    rewritten_texts = [result if result is not None else None for result in rewritten_texts]
    print(f"Completed processing {len(rewritten_texts)} texts")

    pandas_df[rephrased_field] = rewritten_texts

    rephrased_dataset = DocumentDataset.from_pandas(pandas_df)

    print("Running Wikipedia rephraser postprocessing pipeline")
    postprocessing_pipeline = build_wikipedia_postprocessing_pipeline(tokenizer, rephrased_field)

    rephrased_dataset = postprocessing_pipeline(rephrased_dataset)
    print("Wikipedia rephraser postprocessing complete")
    return rephrased_dataset


def build_diverse_qa_postprocessing_pipeline(
    tokenizer: AutoTokenizer, text_field: str, llm_response_field: str
) -> Sequential:
    max_rephrased_tokens = 598
    min_document_tokens = 100

    long_segment_token_count_filter = TokenCountFilter(tokenizer=tokenizer, max_tokens=max_rephrased_tokens)
    document_token_count_filter = TokenCountFilter(tokenizer=tokenizer, min_tokens=min_document_tokens)
    return Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=llm_response_field),
            # Reformat QA pairs
            NemotronCCDiverseQAPostprocessor(tokenizer, text_field=text_field, response_field=llm_response_field),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )


def build_distill_postprocessing_pipeline(tokenizer: AutoTokenizer, llm_response_field: str) -> Sequential:
    max_rephrased_tokens = 1598
    min_document_tokens = 50

    long_segment_token_count_filter = TokenCountFilter(tokenizer=tokenizer, max_tokens=max_rephrased_tokens)
    document_token_count_filter = TokenCountFilter(tokenizer=tokenizer, min_tokens=min_document_tokens)
    return Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=llm_response_field),
            # Remove documents not starting with the specified prefix
            ScoreFilter(
                SubstringFilter(substring="Paraphrased Text:", position="prefix"),
                text_field=llm_response_field,
                score_field="substring",
                score_type=int,
            ),
            # Remove the paraphrase prefix
            Modify(
                Slicer(
                    left="Paraphrased Text:",
                    include_left=False,
                    strip=True,
                ),
                text_field=llm_response_field,
            ),
            # Remove quotation marks
            Modify(QuotationRemover(), text_field=llm_response_field),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )


def build_extract_knowledge_postprocessing_pipeline(tokenizer: AutoTokenizer, llm_response_field: str) -> Sequential:
    max_rephrased_tokens = 1398
    min_document_tokens = 50

    long_segment_token_count_filter = TokenCountFilter(tokenizer=tokenizer, max_tokens=max_rephrased_tokens)
    document_token_count_filter = TokenCountFilter(tokenizer=tokenizer, min_tokens=min_document_tokens)
    return Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=llm_response_field),
            # Remove passage lines
            Modify(
                LineRemover(patterns=["Passage:", "Passage 1:", "Passage 2:", "Passage 3:"]),
                text_field=llm_response_field,
            ),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )


def build_knowledge_list_postprocessing_pipeline(tokenizer: AutoTokenizer, llm_response_field: str) -> Sequential:
    max_rephrased_tokens = 598
    min_document_tokens = 50

    long_segment_token_count_filter = TokenCountFilter(tokenizer=tokenizer, max_tokens=max_rephrased_tokens)
    document_token_count_filter = TokenCountFilter(tokenizer=tokenizer, min_tokens=min_document_tokens)
    return Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=llm_response_field),
            NemotronCCKnowledgeListPostprocessor(text_field=llm_response_field),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )


async def generate_content(  # noqa: PLR0913
    dataset: DocumentDataset,
    text_field: str,
    openai_client: OpenAI,
    tokenizer: AutoTokenizer,
    api_model_name: str,
    task_type: str,
) -> DocumentDataset:
    """
    Generates content based on the specified task type.

    Args:
        dataset (DocumentDataset): The input dataset.
        text_field (str): The field containing the input text.
        openai_client (OpenAI): The OpenAI client.
        tokenizer (AutoTokenizer): The tokenizer.
        api_model_name (str): The name of the OpenAI model to use.
        task_type (str): The type of task to perform ('distill', 'diverse_qa', 'extract_knowledge', 'knowledge_list').
        n_entries (int): The number of entries to process.

    Returns:
        DocumentDataset: The processed dataset.
    """

    client = AsyncOpenAIClient(openai_client)
    nemotron_cc = AsyncNemotronCCGenerator(client)
    llm_response_field = task_type

    # Define configurations for different task types
    task_configs = {
        "distill": {
            "system_prompt": NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
            "prompt_template": DISTILL_PROMPT_TEMPLATE,
            "min_document_tokens": 30,
            "min_segment_tokens": 10,
            "max_input_tokens": 2000,
            "max_output_tokens": 1600,
            "postprocessing_pipeline_builder": build_distill_postprocessing_pipeline,
            "generation_function": nemotron_cc.distill,
        },
        "diverse_qa": {
            "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
            "prompt_template": DIVERSE_QA_PROMPT_TEMPLATE,
            "min_document_tokens": 30,
            "min_segment_tokens": 30,
            "max_input_tokens": 1000,
            "max_output_tokens": 600,
            "postprocessing_pipeline_builder": build_diverse_qa_postprocessing_pipeline,
            "generation_function": nemotron_cc.generate_diverse_qa,
        },
        "extract_knowledge": {
            "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
            "prompt_template": EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
            "min_document_tokens": 30,
            "min_segment_tokens": 30,
            "max_input_tokens": 1400,
            "max_output_tokens": 1400,
            "postprocessing_pipeline_builder": build_extract_knowledge_postprocessing_pipeline,
            "generation_function": nemotron_cc.extract_knowledge,
        },
        "knowledge_list": {
            "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
            "prompt_template": KNOWLEDGE_LIST_PROMPT_TEMPLATE,
            "min_document_tokens": 30,
            "min_segment_tokens": 30,
            "max_input_tokens": 1000,
            "max_output_tokens": 600,
            "postprocessing_pipeline_builder": build_knowledge_list_postprocessing_pipeline,
            "generation_function": nemotron_cc.generate_knowledge_list,
        },
    }

    config = {
        "MIN_DOCUMENT_TOKENS": 30,
        "MIN_SEGMENT_TOKENS": 10,
        "MAX_INPUT_TOKENS": 2000,
        "MAX_OUTPUT_TOKENS": 1600,
        "TOP_K": 0,
        "TOP_P": 0.9,
        "END_STRINGS": "['</s>']",
        "TEMPERATURE": 0.5,
    }

    task_config = task_configs.get(task_type)
    if not task_config:
        msg = f"Invalid task type: {task_type}"
        raise ValueError(msg)

    preprocessing_pipeline = build_preprocessing_pipeline(
        tokenizer,
        text_field,
        task_config["system_prompt"],
        task_config["prompt_template"],
        task_config["min_document_tokens"],
        task_config["min_segment_tokens"],
        task_config["max_input_tokens"],
    )

    print(f"Running {task_type} preprocessing pipeline")
    dataset = preprocessing_pipeline(dataset)

    pandas_df = dataset.df.compute()

    # Process with pandas
    rewritten_texts = []

    async def process_text(text: str) -> str | None:
        try:
            llm_response = await task_config["generation_function"](
                text,
                api_model_name,
                model_kwargs={
                    "top_k": config["TOP_K"],
                    "top_p": config["TOP_P"],
                    "stop": config["END_STRINGS"],
                    "max_tokens": config["MAX_OUTPUT_TOKENS"],
                    "temperature": config["TEMPERATURE"],
                },
            )
            return llm_response[0]
        except Exception as e:  # noqa: BLE001
            print(f"Error processing text: {e!s}")
            return None

    # Create tasks for all texts
    tasks = [process_text(text) for text in pandas_df[text_field]]

    # Run all tasks concurrently with asyncio
    try:
        rewritten_texts = await asyncio.gather(*tasks, return_exceptions=True)
        # Handle any exceptions that occurred during gathering
        rewritten_texts = [result if not isinstance(result, Exception) else None for result in rewritten_texts]
    except Exception as e:  # noqa: BLE001
        print(f"Error during async gathering: {e!s}")
        rewritten_texts = [None] * len(tasks)

    # Assign new column in pandas
    pandas_df[llm_response_field] = rewritten_texts

    # Convert back to Dask
    rephrased_dataset = DocumentDataset.from_pandas(pandas_df)

    if task_type == "diverse_qa":
        postprocessed_pipeline = task_config["postprocessing_pipeline_builder"](
            tokenizer, text_field, llm_response_field
        )
    else:
        postprocessed_pipeline = task_config["postprocessing_pipeline_builder"](tokenizer, llm_response_field)
    rephrased_dataset = postprocessed_pipeline(rephrased_dataset)
    print(f"{task_type} generation complete.")
    return rephrased_dataset
