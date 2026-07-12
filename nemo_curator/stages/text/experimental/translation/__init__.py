"""Experimental translation stages for text data.

APIs in this package are subject to change while the translation workflow is
being validated.
"""

from nemo_curator.stages.text.experimental.translation.evaluation.faith import (
    FaithEvalFilter,
    FaithThresholdFilterStage,
)
from nemo_curator.stages.text.experimental.translation.evaluation.text_quality import (
    TextQualityMetricStage,
    compute_text_quality_metric,
)
from nemo_curator.stages.text.experimental.translation.pipeline import TranslationStage
from nemo_curator.stages.text.experimental.translation.stages.reassembly import ReassemblyStage
from nemo_curator.stages.text.experimental.translation.stages.segmentation import SegmentationStage
from nemo_curator.stages.text.experimental.translation.stages.translate import SegmentTranslationStage
from nemo_curator.stages.text.experimental.translation.utils.field_paths import (
    extract_nested_fields,
    is_wildcard_path,
    normalize_text_field,
    parse_structured_value,
    set_nested_fields,
)
from nemo_curator.stages.text.experimental.translation.utils.metadata import (
    build_translation_metadata,
    merge_faith_scores_into_metadata,
    reconstruct_messages_with_translation,
)

__all__ = [
    "FaithEvalFilter",
    "FaithThresholdFilterStage",
    "ReassemblyStage",
    "SegmentTranslationStage",
    "SegmentationStage",
    "TextQualityMetricStage",
    "TranslationStage",
    "build_translation_metadata",
    "compute_text_quality_metric",
    "extract_nested_fields",
    "is_wildcard_path",
    "merge_faith_scores_into_metadata",
    "normalize_text_field",
    "parse_structured_value",
    "reconstruct_messages_with_translation",
    "set_nested_fields",
]
