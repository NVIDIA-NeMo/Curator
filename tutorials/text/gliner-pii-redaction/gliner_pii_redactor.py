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

from dataclasses import dataclass
from typing import Any

from gliner import GLiNER

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

# Model identifier from Hugging Face
GLINER_PII_MODEL_IDENTIFIER = "nvidia/gliner-pii"
# Full list of entities and quasi identifiers
PII_LABELS = [
    "first_name",
    "last_name",
    "name",
    "street_address",
    "city",
    "state",
    "postcode",
    "country",
    "address",
    "latitude",
    "longitude",
    "coordinate",
    "age",
    "phone_number",
    "fax_number",
    "email",
    "ssn",
    "unique_identifier",
    "medical_record_number",
    "health_plan_beneficiary_number",
    "account_number",
    "certificate_license_number",
    "vehicle_identifier",
    "license_plate",
    "device_identifier",
    "biometric_identifier",
    "url",
    "ipv4",
    "ipv6",
    "national_id",
    "tax_id",
    "bank_routing_number",
    "swift_bic",
    "credit_debit_card",
    "cvv",
    "pin",
    "employee_id",
    "api_key",
    "customer_id",
    "user_name",
    "password",
    "mac_address",
    "http_cookie",
    "date",
    "date_time",
    "blood_type",
    "gender",
    "sexuality",
    "political_view",
    "race",
    "ethnicity",
    "religious_belief",
    "language",
    "education",
    "job_title",
    "employment_status",
    "company_name",
]


@dataclass
class GlinerPiiRedactor(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for redacting PII entities from text.
    It accepts a list of labels to redact and a threshold for the confidence score.

    Args:
        text_field (str): The field to read the text from. Default is "text".
        labels (list[str]): The labels to redact.
        threshold (float): The threshold for the confidence score. Default is 0.5.
        use_gpu (bool): Whether to use GPU acceleration. Default is True.
        model_inference_batch_size (int): The batch size for model inference. Default is 128.

    """

    text_field: str = "text"
    labels: list[str] | None = None
    threshold: float = 0.5
    use_gpu: bool = True
    model_inference_batch_size: int = 128
    name: str = "gliner_pii_redactor"

    def __post_init__(self):
        if self.labels is None:
            self.labels = PII_LABELS
        if self.use_gpu:
            self.resources = Resources(cpus=1, gpus=1)
            self.device = "cuda"
        else:
            self.resources = Resources(cpus=1)
            self.device = "cpu"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": True}

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        _ = GLiNER.from_pretrained(GLINER_PII_MODEL_IDENTIFIER, local_files_only=False)

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.model = GLiNER.from_pretrained(
            GLINER_PII_MODEL_IDENTIFIER, map_location=self.device, local_files_only=True
        )

    def redact_entities(self, text: str, entities: list[dict[str, Any]]) -> str:
        # Ensure the entities are sorted by start index
        entities = sorted(entities, key=lambda x: x["start"])
        for entity in entities:
            # Replace "text" with "{label}"
            # Only replace the first instance of the entity text
            # This is safe because the "entities" list is sorted in the order that they appear in the text
            text = text.replace(entity["text"], f"{{{entity['label']}}}", 1)
        return text

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Applies the redactor to a dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with the PII entities redacted

        """
        df = batch.to_pandas()

        # Run model inference via the GLiNER library
        # This returns a list of dictionaries per document, each containing the entities and their confidence scores
        entities = self.model.inference(
            df[self.text_field].tolist(),
            self.labels,
            threshold=self.threshold,
            batch_size=self.model_inference_batch_size,
        )

        # Redact the entities from the text via the redact_entities method
        df[self.text_field] = [
            self.redact_entities(text, ent) for text, ent in zip(df[self.text_field], entities, strict=True)
        ]

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
