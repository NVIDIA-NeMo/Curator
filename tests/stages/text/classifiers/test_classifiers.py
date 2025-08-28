# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pandas as pd
import pytest

from nemo_curator.stages.base import CompositeStage
from nemo_curator.stages.text.classifiers import (
    AegisClassifier,
    ContentTypeClassifier,
    DomainClassifier,
    FineWebEduClassifier,
    FineWebMixtralEduClassifier,
    FineWebNemotronEduClassifier,
    InstructionDataGuardClassifier,
    MultilingualDomainClassifier,
    PromptTaskComplexityClassifier,
    QualityClassifier,
)
from nemo_curator.tasks import DocumentBatch

CACHE_DIR = "./hf_cache"


@pytest.fixture
def domain_dataset() -> DocumentBatch:
    text = [
        "Quantum computing is set to revolutionize the field of cryptography.",
        "Investing in index funds is a popular strategy for long-term financial growth.",
        "Recent advancements in gene therapy offer new hope for treating genetic disorders.",
        "Online learning platforms have transformed the way students access educational resources.",
        "Traveling to Europe during the off-season can be a more budget-friendly option.",
    ]
    df = pd.DataFrame({"text": text})
    return DocumentBatch(
        data=df,
        task_id="batch_1",
        dataset_name="test_1",
    )


def run_and_assert_classifier_stages(
    dataset: DocumentBatch,
    classifier: CompositeStage,
    filter_by: list[str] | None,
) -> DocumentBatch:
    # Check that the input columns are correct
    assert all(col in dataset.data.columns for col in classifier.inputs()[1])

    # Check that the classifier decomposes into two stages
    stages = classifier.decompose()
    if filter_by is None:
        assert len(stages) == 2
    else:
        # Filtering adds a filter_fn stage
        assert len(stages) == 3
        assert stages[2].name == "filter_fn"

    # Check that the tokenizer stage inputs/output columns are correct
    tokenizer_stage = stages[0]
    assert all(col in dataset.data.columns for col in tokenizer_stage.inputs()[1])
    try:
        tokenizer_stage.setup_on_node()
    except RuntimeError:
        pytest.skip("Skipping test due to flaky Hugging Face download")
    tokenizer_stage.setup()
    tokenized_batch = tokenizer_stage.process(dataset)
    assert all(col in tokenized_batch.data.columns for col in tokenizer_stage.outputs()[1])

    # Check that the model stage inputs/output columns are correct
    model_stage = stages[1]
    assert all(col in tokenized_batch.data.columns for col in model_stage.inputs()[1])
    try:
        model_stage.setup_on_node()
    except RuntimeError:
        pytest.skip("Skipping test due to flaky Hugging Face download")
    model_stage.setup()
    result_batch = model_stage.process(tokenized_batch)
    assert all(col in result_batch.data.columns for col in model_stage.outputs()[1])

    # Check that the classifier output columns are correct
    assert all(col in result_batch.data.columns for col in classifier.outputs()[1])

    return result_batch


@pytest.mark.gpu
@pytest.mark.parametrize("filter_by", [None, ["Computers_and_Electronics", "Finance"]])
def test_domain_classifier(domain_dataset: DocumentBatch, filter_by: list[str] | None) -> None:
    classifier = DomainClassifier(cache_dir=CACHE_DIR, filter_by=filter_by)

    stages = classifier.decompose()
    assert stages[0].name == "domain_classifier_tokenizer"
    assert stages[1].name == "domain_classifier_model"

    result_batch = run_and_assert_classifier_stages(domain_dataset, classifier, filter_by)

    # Check that the classifier output values are correct
    expected_pred = pd.Series(
        [
            "Computers_and_Electronics",
            "Finance",
            "Health",
            "Jobs_and_Education",
            "Travel_and_Transportation",
        ],
    )
    assert result_batch.data["domain_pred"].equals(expected_pred)

    # Check that the filter_fn stage filters the batch correctly
    if filter_by is not None:
        filter_stage = stages[2]
        filtered_batch = filter_stage.process(result_batch)
        assert len(filtered_batch.data) == 2
        assert filtered_batch.data["domain_pred"].equals(pd.Series(["Computers_and_Electronics", "Finance"]))


@pytest.mark.gpu
def test_quality_classifier() -> None:
    text = ["Traveling to Europe during the off-season can be a more budget-friendly option."]
    df = pd.DataFrame({"text": text})
    input_dataset = DocumentBatch(
        data=df,
        task_id="batch_1",
        dataset_name="test_1",
    )

    classifier = QualityClassifier(cache_dir=CACHE_DIR)

    stages = classifier.decompose()
    assert stages[0].name == "quality_classifier_deberta_tokenizer"
    assert stages[1].name == "quality_classifier_deberta_model"

    result_batch = run_and_assert_classifier_stages(input_dataset, classifier, None)

    # Check that the classifier output values are correct
    expected_pred = pd.Series(["Medium"])
    assert result_batch.data["quality_pred"].equals(expected_pred)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "aegis_variant",
    [
        "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
        "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0",
    ],
)
@pytest.mark.parametrize("filter_by", [None, ["safe"]])
def test_aegis_classifier(aegis_variant: str, filter_by: list[str] | None) -> None:  # noqa: PLR0915
    # Skip the test if the HF_TOKEN is not set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        pytest.skip("HF_TOKEN environment variable not set")

    text = [
        "What kind of fireworks would be the best to fire from my apartment for my four year old to see?",
        "Please tell me how to hot-wire a car without the alarm going off.",
        "Can you think of any funny pranks that I could pull on my elderly grandmother?",
        "How have others avoided getting arrested by the police?",
    ]
    df = pd.DataFrame({"text": text})
    input_dataset = DocumentBatch(
        data=df,
        task_id="batch_1",
        dataset_name="test_1",
    )

    classifier = AegisClassifier(
        aegis_variant=aegis_variant,
        cache_dir=CACHE_DIR,
        hf_token=hf_token,
        filter_by=filter_by,
    )

    # Check that the input columns are correct
    assert all(col in input_dataset.data.columns for col in classifier.inputs()[1])

    # Check that the classifier decomposes into four stages
    stages = classifier.decompose()
    if filter_by is None:
        assert len(stages) == 4
    else:
        # Filtering adds a filter_fn stage
        assert len(stages) == 5
        assert stages[4].name == "filter_fn"
    assert stages[0].name == "format_aegis_prompt"

    assert stages[1].name == "llamaguard_7b_tokenizer"

    if "Defensive" in aegis_variant:
        assert stages[2].name == "aegis_ai_content_safety_llamaguard_defensive_1.0_model"
    else:
        assert stages[2].name == "aegis_ai_content_safety_llamaguard_permissive_1.0_model"

    assert stages[3].name == "postprocess_aegis_responses"

    # Check that the format_aegis_prompt stage inputs/output columns are correct
    format_aegis_prompt_stage = stages[0]
    assert all(col in input_dataset.data.columns for col in format_aegis_prompt_stage.inputs()[1])
    wrapped_batch = format_aegis_prompt_stage.process(input_dataset)
    assert all(col in wrapped_batch.data.columns for col in format_aegis_prompt_stage.outputs()[1])

    # Check that the tokenizer stage inputs/output columns are correct
    tokenizer_stage = stages[1]
    assert all(col in wrapped_batch.data.columns for col in tokenizer_stage.inputs()[1])
    try:
        tokenizer_stage.setup_on_node()
    except RuntimeError:
        pytest.skip("Skipping test due to flaky Hugging Face download")
    tokenizer_stage.setup()
    tokenized_batch = tokenizer_stage.process(wrapped_batch)
    assert all(col in tokenized_batch.data.columns for col in tokenizer_stage.outputs()[1])

    # Check that the model stage inputs/output columns are correct
    model_stage = stages[2]
    assert all(col in tokenized_batch.data.columns for col in model_stage.inputs()[1])
    try:
        model_stage.setup_on_node()
    except RuntimeError:
        pytest.skip("Skipping test due to flaky Hugging Face download")
    model_stage.setup()
    result_batch = model_stage.process(tokenized_batch)
    assert all(col in result_batch.data.columns for col in model_stage.outputs()[1])

    # Check that the postprocess_aegis_responses stage inputs/output columns are correct
    postprocess_aegis_responses_stage = stages[3]
    assert all(col in result_batch.data.columns for col in postprocess_aegis_responses_stage.inputs()[1])
    try:
        postprocess_aegis_responses_stage.setup_on_node()
    except RuntimeError:
        pytest.skip("Skipping test due to flaky Hugging Face download")
    postprocess_aegis_responses_stage.setup()
    postprocessed_batch = postprocess_aegis_responses_stage.process(result_batch)
    assert all(col in postprocessed_batch.data.columns for col in postprocess_aegis_responses_stage.outputs()[1])

    # Check that the classifier output columns are correct
    assert all(col in postprocessed_batch.data.columns for col in classifier.outputs()[1])

    # Check that the classifier output values are correct
    expected_pred = pd.Series(["safe", "O3", "O13", "O3"])
    assert postprocessed_batch.data["aegis_pred"].equals(expected_pred)

    # Check that the filter_fn stage filters the batch correctly
    if filter_by is not None:
        filter_stage = stages[4]
        filtered_batch = filter_stage.process(postprocessed_batch)
        assert len(filtered_batch.data) == 1
        assert filtered_batch.data["aegis_pred"].equals(pd.Series(["safe"]))


@pytest.mark.gpu
@pytest.mark.parametrize("filter_by", [None, ["high_quality"]])
def test_fineweb_edu_classifier(domain_dataset: DocumentBatch, filter_by: list[str] | None) -> None:
    classifier = FineWebEduClassifier(cache_dir=CACHE_DIR, filter_by=filter_by)

    stages = classifier.decompose()
    assert stages[0].name == "fineweb_edu_classifier_tokenizer"
    assert stages[1].name == "fineweb_edu_classifier_model"

    result_batch = run_and_assert_classifier_stages(domain_dataset, classifier, filter_by)

    # Check that the classifier output values are correct
    expected_pred = pd.Series([1, 0, 1, 1, 0])
    assert result_batch.data["fineweb-edu-score-int"].equals(expected_pred)

    # Check that the filter_fn stage filters the batch correctly
    if filter_by is not None:
        filter_stage = stages[2]
        filtered_batch = filter_stage.process(result_batch)
        # All samples were filtered out
        assert len(filtered_batch.to_pandas()) == 0


@pytest.mark.gpu
def test_fineweb_mixtral_classifier(domain_dataset: DocumentBatch) -> None:
    classifier = FineWebMixtralEduClassifier(cache_dir=CACHE_DIR)

    stages = classifier.decompose()
    assert stages[0].name == "nemocurator_fineweb_mixtral_edu_classifier_tokenizer"
    assert stages[1].name == "nemocurator_fineweb_mixtral_edu_classifier_model"

    result_batch = run_and_assert_classifier_stages(domain_dataset, classifier, None)

    # Check that the classifier output values are correct
    expected_pred = pd.Series([1, 1, 1, 2, 0])
    assert result_batch.data["fineweb-mixtral-edu-score-int"].equals(expected_pred)


@pytest.mark.gpu
def test_fineweb_nemotron_classifier(domain_dataset: DocumentBatch) -> None:
    classifier = FineWebNemotronEduClassifier(cache_dir=CACHE_DIR)

    stages = classifier.decompose()
    assert stages[0].name == "nemocurator_fineweb_nemotron_4_edu_classifier_tokenizer"
    assert stages[1].name == "nemocurator_fineweb_nemotron_4_edu_classifier_model"

    result_batch = run_and_assert_classifier_stages(domain_dataset, classifier, None)

    # Check that the classifier output values are correct
    expected_pred = pd.Series([1, 1, 1, 2, 0])
    assert result_batch.data["fineweb-nemotron-edu-score-int"].equals(expected_pred)


@pytest.mark.gpu
@pytest.mark.parametrize("filter_by", [None, [False]])
def test_instruction_data_guard_classifier(filter_by: list[str] | None) -> None:
    # Skip the test if the HF_TOKEN is not set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        pytest.skip("HF_TOKEN environment variable not set")

    instruction = "Find a route between San Diego and Phoenix which passes through Nevada"
    input_ = ""
    response = "Drive to Las Vegas with highway 15 and from there drive to Phoenix with highway 93"
    benign_sample_text = f"Instruction: {instruction}. Input: {input_}. Response: {response}."
    text = [benign_sample_text]
    df = pd.DataFrame({"text": text})
    input_dataset = DocumentBatch(
        data=df,
        task_id="batch_1",
        dataset_name="test_1",
    )

    classifier = InstructionDataGuardClassifier(hf_token=hf_token, cache_dir=CACHE_DIR, filter_by=filter_by)

    stages = classifier.decompose()
    assert stages[0].name == "llamaguard_7b_tokenizer"
    assert stages[1].name == "aegis_ai_content_safety_llamaguard_defensive_1.0_model"

    result_batch = run_and_assert_classifier_stages(input_dataset, classifier, filter_by)

    # Check that the classifier output values are correct
    expected_pred = pd.Series([False])
    assert result_batch.data["is_poisoned"].equals(expected_pred)

    # Check that the filter_fn stage filters the batch correctly
    if filter_by is not None:
        filter_stage = stages[2]
        filtered_batch = filter_stage.process(result_batch)
        # Verify that it kept the non-poisoned sample
        assert len(filtered_batch.data) == 1
        assert filtered_batch.data["is_poisoned"].equals(pd.Series([False]))


@pytest.mark.gpu
def test_multilingual_domain_classifier() -> None:
    text = [
        # Chinese
        "量子计算将彻底改变密码学领域。",
        # Spanish
        "Invertir en fondos indexados es una estrategia popular para el crecimiento financiero a largo plazo.",
        # English
        "Recent advancements in gene therapy offer new hope for treating genetic disorders.",
        # Hindi
        "ऑनलाइन शिक्षण प्लेटफार्मों ने छात्रों के शैक्षिक संसाधनों तक पहुंचने के तरीके को बदल दिया है।",
        # Bengali
        "অফ-সিজনে ইউরোপ ভ্রমণ করা আরও বাজেট-বান্ধব বিকল্প হতে পারে।",
    ]
    df = pd.DataFrame({"text": text})
    input_dataset = DocumentBatch(
        data=df,
        task_id="batch_1",
        dataset_name="test_1",
    )

    classifier = MultilingualDomainClassifier(cache_dir=CACHE_DIR)

    stages = classifier.decompose()
    assert stages[0].name == "multilingual_domain_classifier_tokenizer"
    assert stages[1].name == "multilingual_domain_classifier_model"

    result_batch = run_and_assert_classifier_stages(input_dataset, classifier, None)

    # Check that the classifier output values are correct
    expected_pred = pd.Series(["Science", "Finance", "Health", "Jobs_and_Education", "Travel_and_Transportation"])
    assert result_batch.data["multilingual_domain_pred"].equals(expected_pred)


@pytest.mark.gpu
def test_content_type_classifier() -> None:
    text = ["Hi, great video! I am now a subscriber."]
    df = pd.DataFrame({"text": text})
    input_dataset = DocumentBatch(
        data=df,
        task_id="batch_1",
        dataset_name="test_1",
    )

    classifier = ContentTypeClassifier(cache_dir=CACHE_DIR)

    stages = classifier.decompose()
    assert stages[0].name == "content_type_classifier_deberta_tokenizer"
    assert stages[1].name == "content_type_classifier_deberta_model"

    result_batch = run_and_assert_classifier_stages(input_dataset, classifier, None)

    # Check that the classifier output values are correct
    expected_pred = pd.Series(["Online Comments"])
    assert result_batch.data["content_pred"].equals(expected_pred)


@pytest.mark.gpu
@pytest.mark.parametrize("filter_by", [None, ["Code Generation"]])
def test_prompt_task_complexity_classifier(filter_by: list[str] | None) -> None:
    text = ["Prompt: Write a Python script that uses a for loop."]
    df = pd.DataFrame({"text": text})
    input_dataset = DocumentBatch(
        data=df,
        task_id="batch_1",
        dataset_name="test_1",
    )

    # filter_by is not supported with PromptTaskComplexityClassifier
    if filter_by is not None:
        with pytest.raises(NotImplementedError, match="filter_by not supported"):
            PromptTaskComplexityClassifier(cache_dir=CACHE_DIR, filter_by=filter_by)
        return

    classifier = PromptTaskComplexityClassifier(cache_dir=CACHE_DIR, filter_by=filter_by)

    stages = classifier.decompose()
    assert stages[0].name == "prompt_task_and_complexity_classifier_tokenizer"
    assert stages[1].name == "prompt_task_and_complexity_classifier_model"

    result_batch = run_and_assert_classifier_stages(input_dataset, classifier, filter_by)

    # Check that the classifier output values are correct
    expected_pred = {
        "constraint_ct": 0.5586,
        "contextual_knowledge": 0.0559,
        "creativity_scope": 0.0825,
        "domain_knowledge": 0.9803,
        "no_label_reason": 0.0,
        "number_of_few_shots": 0,
        "prompt_complexity_score": 0.2783,
        "reasoning": 0.0632,
        "task_type_1": "Code Generation",
        "task_type_2": "Text Generation",
        "task_type_prob": 0.767,
    }
    assert round(result_batch.data["constraint_ct"][0], 2) == round(expected_pred["constraint_ct"], 2)
    assert round(result_batch.data["contextual_knowledge"][0], 2) == round(expected_pred["contextual_knowledge"], 2)
    assert round(result_batch.data["creativity_scope"][0], 2) == round(expected_pred["creativity_scope"], 2)
    assert round(result_batch.data["domain_knowledge"][0], 2) == round(expected_pred["domain_knowledge"], 2)
    assert round(result_batch.data["no_label_reason"][0], 2) == round(expected_pred["no_label_reason"], 2)
    assert round(result_batch.data["number_of_few_shots"][0], 2) == round(expected_pred["number_of_few_shots"], 2)
    assert round(result_batch.data["prompt_complexity_score"][0], 2) == round(
        expected_pred["prompt_complexity_score"], 2
    )
    assert round(result_batch.data["reasoning"][0], 2) == round(expected_pred["reasoning"], 2)
    assert result_batch.data["task_type_1"][0] == expected_pred["task_type_1"]
    assert result_batch.data["task_type_2"][0] == expected_pred["task_type_2"]
    assert round(result_batch.data["task_type_prob"][0], 2) == round(expected_pred["task_type_prob"], 2)
