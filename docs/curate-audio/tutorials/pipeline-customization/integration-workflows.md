---
description: "Build complex workflows combining audio, text, and multimodal processing for comprehensive data curation"
categories: ["tutorials"]
tags: ["integration-workflows", "multimodal", "workflow-integration", "advanced-pipelines", "cross-modal"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "multimodal"
---

# Integration Workflows

Build complex workflows that combine audio curation with text processing and multimodal capabilities for comprehensive data preparation. This tutorial covers cross-modal integration patterns and advanced pipeline architectures.

## Multimodal Integration Patterns

### Audio-Text Integration Workflow

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import WordCountFilter, FastTextLangId
from nemo_curator.tasks import AudioBatch, DocumentBatch

def create_audio_text_integration_pipeline() -> Pipeline:
    """Integrate audio processing with text curation."""
    
    pipeline = Pipeline(name="audio_text_integration")
    
    # Audio processing branch
    pipeline.add_stage(CreateInitialManifestFleursStage(
        lang="en_us", split="train", raw_data_dir="/data/audio"
    ))
    
    pipeline.add_stage(InferenceAsrNemoStage(
        model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
    ).with_(resources=Resources(gpus=1.0)))
    
    pipeline.add_stage(GetPairwiseWerStage())
    pipeline.add_stage(GetAudioDurationStage(
        audio_filepath_key="audio_filepath", duration_key="duration"
    ))
    
    # Quality filtering
    pipeline.add_stage(
        PreserveByValueStage(
            input_value_key="wer",
            target_value=25.0,
            operator="le",
        )
    )
    
    # Convert to text format
    pipeline.add_stage(AudioToDocumentStage())
    
    # Text processing branch
    pipeline.add_stage(ScoreFilter(
        WordCountFilter(min_words=5, max_words=100),
        text_field="pred_text",  # Process ASR predictions
        score_field="word_count_score"
    ))
    
    pipeline.add_stage(
        ScoreFilter(
            FastTextLangId(model_path="/path/to/lid.176.bin", min_langid_score=0.3),
            text_field="pred_text",
            score_field="language",
        )
    )
    
    # Cross-modal validation
    @processing_stage(name="cross_modal_validation")
    def validate_audio_text_consistency(document_batch: DocumentBatch) -> DocumentBatch:
        validated_data = []
        
        for doc in document_batch.data:
            # Check audio-text consistency
            duration = doc.get("duration", 0)
            pred_text = doc.get("pred_text", "")
            wer = doc.get("wer", 100)
            
            # Consistency checks
            char_rate = len(pred_text) / duration if duration > 0 else 0
            
            # Expected character rate: 8-25 chars/second
            rate_consistent = 8.0 <= char_rate <= 25.0
            quality_acceptable = wer <= 30.0
            
            if rate_consistent and quality_acceptable:
                doc["cross_modal_validated"] = True
                validated_data.append(doc)
        
        return DocumentBatch(data=validated_data)
    
    pipeline.add_stage(validate_audio_text_consistency)
    
    return pipeline
```

### Audio-Video Integration

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.stages.video.clipping.video_frame_extraction import VideoFrameExtractionStage
from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage

def create_audio_video_integration_pipeline() -> Pipeline:
    """Integrate video reading, frame extraction, and caption generation."""

    pipeline = Pipeline(name="audio_video_integration")

    # Load and process videos
    pipeline.add_stage(VideoReader(input_video_path="/data/videos", video_limit=None, verbose=True))
    pipeline.add_stage(VideoFrameExtractionStage())
    pipeline.add_stage(CaptionGenerationStage())

    return pipeline
```

## Advanced Workflow Patterns

### Iterative Refinement Workflow

```python
def create_iterative_refinement_pipeline() -> Pipeline:
    """Create pipeline with iterative quality refinement."""
    
    pipeline = Pipeline(name="iterative_refinement")
    
    # Initial processing
    pipeline.add_stage(initial_audio_loading)
    pipeline.add_stage(initial_asr_inference)
    
    # Iterative refinement loop
    for iteration in range(3):  # 3 refinement iterations
        
        # Quality assessment
        pipeline.add_stage(GetPairwiseWerStage(
            wer_key=f"wer_iter_{iteration}"
        ))
        
        # Conditional reprocessing
        @processing_stage(name=f"conditional_reprocess_{iteration}")
        def conditional_reprocess(audio_batch: AudioBatch) -> AudioBatch:
            # Identify samples needing reprocessing
            reprocess_candidates = []
            final_results = []
            
            for item in audio_batch.data:
                current_wer = item.get(f"wer_iter_{iteration}", 100)
                
                if current_wer > 40.0 and iteration < 2:  # Reprocess if poor quality
                    reprocess_candidates.append(item)
                else:
                    final_results.append(item)
            
            # Reprocess with different model or parameters
            if reprocess_candidates:
                reprocess_batch = AudioBatch(
                    data=reprocess_candidates,
                    filepath_key=audio_batch.filepath_key
                )
                
                # Use different model for reprocessing
                alternative_asr = InferenceAsrNemoStage(
                    model_name="nvidia/stt_multilingual_fastconformer_hybrid_large_pc",
                    pred_text_key=f"pred_text_iter_{iteration + 1}"
                )
                
                reprocessed = alternative_asr.process(reprocess_batch)
                final_results.extend(reprocessed.data)
            
            return AudioBatch(data=final_results, filepath_key=audio_batch.filepath_key)
        
        pipeline.add_stage(conditional_reprocess)
    
    # Final quality selection
    @processing_stage(name="select_best_transcription")
    def select_best_transcription(audio_batch: AudioBatch) -> AudioBatch:
        for item in audio_batch.data:
            # Find best transcription across iterations
            best_wer = float('inf')
            best_transcription = ""
            
            for i in range(3):
                wer_key = f"wer_iter_{i}"
                pred_key = f"pred_text_iter_{i}"
                
                if wer_key in item and pred_key in item:
                    if item[wer_key] < best_wer:
                        best_wer = item[wer_key]
                        best_transcription = item[pred_key]
            
            item["final_pred_text"] = best_transcription
            item["final_wer"] = best_wer
        
        return audio_batch
    
    pipeline.add_stage(select_best_transcription)
    
    return pipeline
```

### Quality-Driven Workflow Routing

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import PreserveByValueStage
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.tasks import AudioBatch

def create_quality_driven_routing_pipeline() -> Pipeline:
    """Annotate samples with a quality tier and filter sequentially based on thresholds."""

    pipeline = Pipeline(name="quality_driven_routing")

    # Initial assessment (compute WER or another quality metric)
    pipeline.add_stage(GetPairwiseWerStage(text_key="text", pred_text_key="pred_text", wer_key="quick_wer"))

    # Tag each record with a quality tier label
    @processing_stage(name="quality_tier_labelling")
    def label_quality(audio_batch: AudioBatch) -> AudioBatch:
        for item in audio_batch.data:
            wer = item.get("quick_wer", 100)
            if wer <= 15:
                item["quality_tier"] = "high_quality"
            elif wer <= 40:
                item["quality_tier"] = "medium_quality"
            elif wer <= 70:
                item["quality_tier"] = "low_quality"
            else:
                item["quality_tier"] = "very_poor"
        return audio_batch

    pipeline.add_stage(label_quality)

    # Example: keep only samples up to 40% WER (high + medium)
    pipeline.add_stage(
        PreserveByValueStage(
            input_value_key="quick_wer",
            target_value=40.0,
            operator="le",
        )
    )

    return pipeline
```

## Best Practices

### Workflow Design

1. **Modular Architecture**: Design reusable workflow components
2. **Quality Gates**: Implement quality checkpoints throughout workflows
3. **Fallback Strategies**: Provide alternative paths for edge cases
4. **Performance Monitoring**: Track workflow efficiency and bottlenecks

### Integration Strategy

1. **Data Consistency**: Maintain consistent data formats across modalities
2. **Metadata Preservation**: Keep important metadata through all transformations
3. **Error Propagation**: Handle errors gracefully across workflow stages
4. **Resource Management**: Optimize resource allocation for complex workflows

## Related Topics

- **[Custom ASR Models](custom-asr-models.md)** - Specialized model integration
- **[Custom Quality Filters](custom-filters.md)** - Domain-specific filtering
- **[Text Integration](../../process-data/text-integration/index.md)** - Audio-text workflow patterns
- **[Multimodal Concepts](../../../about/concepts/index.md)** - Cross-modal processing concepts

