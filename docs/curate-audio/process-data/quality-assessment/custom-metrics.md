---
description: "Implement custom quality assessment metrics for specialized audio curation use cases and domain-specific requirements"
categories: ["audio-processing"]
tags: ["custom-metrics", "quality-assessment", "domain-specific", "extensibility", "advanced-filtering"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "audio-only"
---

# Custom Quality Metrics

Create custom quality assessment metrics for specialized audio curation use cases, domain-specific requirements, and advanced filtering strategies beyond standard WER and duration filtering.

```{important}
The example thresholds and weights in this article are illustrative and should be calibrated for your specific dataset and goals. Performance may vary significantly based on audio quality, domain, and language.
```

## Creating Custom Quality Stages

### Basic Custom Metric Stage

```python
from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch
from dataclasses import dataclass

@dataclass
class CustomAudioQualityStage(LegacySpeechStage):
    """Template for custom audio quality assessment."""
    
    # Configuration parameters
    quality_threshold: float = 0.5
    metric_name: str = "custom_quality"
    
    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        # Extract relevant data
        audio_filepath = data_entry.get("audio_filepath", "")
        text = data_entry.get("text", "")
        pred_text = data_entry.get("pred_text", "")
        duration = data_entry.get("duration", 0)
        
        # Calculate custom quality metric
        quality_score = self.calculate_quality_metric(
            data_entry, audio_filepath, text, pred_text, duration
        )
        
        # Add metric to data entry
        data_entry[self.metric_name] = quality_score
        
        # Optional: Apply threshold filtering
        if quality_score >= self.quality_threshold:
            data_entry[f"{self.metric_name}_passed"] = True
            return [AudioBatch(data=data_entry)]
        else:
            return []  # Filter out low-quality samples
    
    def calculate_quality_metric(self, data_entry: dict, audio_filepath: str, text: str, 
                               pred_text: str, duration: float) -> float:
        """Override this method to implement custom quality calculation."""
        # Placeholder implementation
        return 1.0  # Replace with actual metric calculation
```

## Domain-Specific Quality Metrics

### Conversational Speech Quality

```python
@dataclass
class ConversationalQualityStage(CustomAudioQualityStage):
    """Quality assessment specialized for conversational speech."""
    metric_name: str = "conversational_quality"
    
    def calculate_quality_metric(self, data_entry: dict, audio_filepath: str, text: str,
                               pred_text: str, duration: float) -> float:
        """Calculate conversational speech quality score."""
        
        quality_factors = []
        
        # Factor 1: Natural speech patterns (disfluencies are normal)
        disfluency_markers = ["uh", "um", "er", "ah", "like", "you know"]
        disfluency_count = sum(
            text.lower().count(marker) + pred_text.lower().count(marker)
            for marker in disfluency_markers
        )
        
        # Moderate disfluencies are natural in conversation
        if 0 <= disfluency_count <= 3:
            disfluency_score = 1.0
        elif disfluency_count <= 6:
            disfluency_score = 0.8
        else:
            disfluency_score = 0.5  # Too many disfluencies
        
        quality_factors.append(("disfluency_naturalness", disfluency_score, 0.2))
        
        # Factor 2: Speech continuity (avoid choppy speech)
        word_count = len(text.split())
        if word_count > 0 and duration > 0:
            speech_rate = word_count / duration
            
            # Optimal conversational rate: 2-4 words per second
            if 2.0 <= speech_rate <= 4.0:
                continuity_score = 1.0
            elif 1.5 <= speech_rate < 2.0 or 4.0 < speech_rate <= 5.0:
                continuity_score = 0.8
            else:
                continuity_score = 0.4
        else:
            continuity_score = 0.0
        
        quality_factors.append(("speech_continuity", continuity_score, 0.3))
        
        # Factor 3: Transcription consistency (lower WER weight for conversation)
        wer = data_entry.get("wer", 100)
        if wer <= 30:
            transcription_score = 1.0
        elif wer <= 50:
            transcription_score = 0.7
        else:
            transcription_score = 0.3
        
        quality_factors.append(("transcription_quality", transcription_score, 0.3))
        
        # Factor 4: Content completeness
        if text and pred_text:
            # Check for sentence completeness markers
            sentence_endings = [".", "!", "?"]
            has_ending = any(text.strip().endswith(end) for end in sentence_endings)
            completeness_score = 1.0 if has_ending else 0.7
        else:
            completeness_score = 0.0
        
        quality_factors.append(("content_completeness", completeness_score, 0.2))
        
        # Calculate weighted quality score
        total_score = sum(score * weight for _, score, weight in quality_factors)
        
        # Store detailed breakdown
        data_entry["conversational_quality_breakdown"] = {
            factor_name: {"score": score, "weight": weight}
            for factor_name, score, weight in quality_factors
        }
        
        return total_score
```

### Broadcast Speech Quality

```python
@dataclass
class BroadcastQualityStage(CustomAudioQualityStage):
    """Quality assessment for broadcast and professional speech."""
    metric_name: str = "broadcast_quality"
    
    def calculate_quality_metric(self, data_entry: dict, audio_filepath: str, text: str,
                               pred_text: str, duration: float) -> float:
        """Calculate broadcast speech quality score."""
        
        quality_factors = []
        
        # Factor 1: Pronunciation clarity (strict WER requirements)
        wer = data_entry.get("wer", 100)
        if wer <= 5:
            clarity_score = 1.0
        elif wer <= 15:
            clarity_score = 0.8
        elif wer <= 25:
            clarity_score = 0.5
        else:
            clarity_score = 0.2
        
        quality_factors.append(("pronunciation_clarity", clarity_score, 0.4))
        
        # Factor 2: Professional speech characteristics
        # Check for professional vocabulary and structure
        professional_indicators = [
            "according to", "furthermore", "however", "therefore",
            "in conclusion", "as mentioned", "specifically"
        ]
        
        professional_count = sum(
            1 for indicator in professional_indicators
            if indicator in text.lower()
        )
        
        professional_score = min(1.0, professional_count / 2.0)  # Normalize
        quality_factors.append(("professional_language", professional_score, 0.2))
        
        # Factor 3: Audio technical quality
        try:
            import soundfile as sf
            info = sf.info(audio_filepath)
            
            # Prefer higher sample rates for broadcast
            if info.samplerate >= 44100:
                technical_score = 1.0
            elif info.samplerate >= 22050:
                technical_score = 0.8
            elif info.samplerate >= 16000:
                technical_score = 0.6
            else:
                technical_score = 0.3
                
        except ImportError:
            logger.warning("soundfile not available for technical quality analysis")
            technical_score = 0.5  # Default when dependencies missing
        except Exception:
            technical_score = 0.5  # Default if cannot analyze
        
        quality_factors.append(("technical_quality", technical_score, 0.2))
        
        # Factor 4: Optimal duration for broadcast segments
        if 5.0 <= duration <= 30.0:
            duration_score = 1.0
        elif 2.0 <= duration < 5.0 or 30.0 < duration <= 60.0:
            duration_score = 0.7
        else:
            duration_score = 0.4
        
        quality_factors.append(("broadcast_duration", duration_score, 0.2))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in quality_factors)
        
        return total_score
```

### Telephony Speech Quality

```python
@dataclass
class TelephonyQualityStage(CustomAudioQualityStage):
    """Quality assessment for telephony and call center audio."""
    metric_name: str = "telephony_quality"
    
    def calculate_quality_metric(self, data_entry: dict, audio_filepath: str, text: str,
                               pred_text: str, duration: float) -> float:
        """Calculate telephony-specific quality score."""
        
        quality_factors = []
        
        # Factor 1: Bandwidth-adjusted WER expectations
        wer = data_entry.get("wer", 100)
        # More lenient WER for telephony due to bandwidth limitations
        if wer <= 25:
            telephony_wer_score = 1.0
        elif wer <= 40:
            telephony_wer_score = 0.8
        elif wer <= 60:
            telephony_wer_score = 0.5
        else:
            telephony_wer_score = 0.2
        
        quality_factors.append(("telephony_wer", telephony_wer_score, 0.3))
        
        # Factor 2: Call characteristics
        # Check for typical call patterns
        call_patterns = [
            "hello", "hi", "thank you", "goodbye", "yes", "no",
            "can you", "could you", "please", "sorry"
        ]
        
        pattern_matches = sum(
            1 for pattern in call_patterns
            if pattern in text.lower() or pattern in pred_text.lower()
        )
        
        call_naturalness = min(1.0, pattern_matches / 3.0)
        quality_factors.append(("call_naturalness", call_naturalness, 0.2))
        
        # Factor 3: Duration appropriateness for calls
        # Telephony segments are typically shorter
        if 2.0 <= duration <= 15.0:
            duration_score = 1.0
        elif 1.0 <= duration < 2.0 or 15.0 < duration <= 25.0:
            duration_score = 0.8
        else:
            duration_score = 0.4
        
        quality_factors.append(("call_duration", duration_score, 0.2))
        
        # Factor 4: Compression tolerance
        # Telephony audio often has compression artifacts
        try:
            import soundfile as sf
            info = sf.info(audio_filepath)
            
            # 8kHz is standard for telephony
            if info.samplerate == 8000:
                compression_score = 1.0
            elif info.samplerate == 16000:
                compression_score = 0.9
            else:
                compression_score = 0.7
                
        except ImportError:
            logger.warning("soundfile not available for compression analysis")
            compression_score = 0.5
        except Exception:
            compression_score = 0.5
        
        quality_factors.append(("compression_tolerance", compression_score, 0.3))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in quality_factors)
        
        return total_score
```

## Acoustic Quality Metrics

### Audio Signal Quality

```python
from loguru import logger

@dataclass
class AcousticQualityStage(CustomAudioQualityStage):
    """Assess acoustic signal quality characteristics."""
    metric_name: str = "acoustic_quality"
    
    def calculate_quality_metric(self, data_entry: dict, audio_filepath: str, text: str,
                               pred_text: str, duration: float) -> float:
        """Calculate acoustic signal quality."""
        
        try:
            import soundfile as sf
            import numpy as np
            
            # Read audio data
            audio_data, sample_rate = sf.read(audio_filepath)
            
            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            quality_factors = []
            
            # Factor 1: Signal-to-Noise Ratio estimation
            # Simple energy-based SNR estimation
            signal_energy = np.mean(audio_data ** 2)
            noise_floor = np.percentile(audio_data ** 2, 10)  # Bottom 10% as noise estimate
            
            if noise_floor > 0:
                snr_estimate = 10 * np.log10(signal_energy / noise_floor)
                
                if snr_estimate >= 20:
                    snr_score = 1.0
                elif snr_estimate >= 15:
                    snr_score = 0.8
                elif snr_estimate >= 10:
                    snr_score = 0.6
                else:
                    snr_score = 0.3
            else:
                snr_score = 0.5
            
            quality_factors.append(("snr_estimate", snr_score, 0.3))
            
            # Factor 2: Dynamic range
            dynamic_range = np.max(np.abs(audio_data)) - np.min(np.abs(audio_data))
            
            if dynamic_range >= 0.5:
                range_score = 1.0
            elif dynamic_range >= 0.3:
                range_score = 0.8
            elif dynamic_range >= 0.1:
                range_score = 0.6
            else:
                range_score = 0.2  # Very low dynamic range
            
            quality_factors.append(("dynamic_range", range_score, 0.2))
            
            # Factor 3: Clipping detection
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio_data) >= clipping_threshold)
            clipping_ratio = clipped_samples / len(audio_data)
            
            if clipping_ratio <= 0.001:  # < 0.1% clipped
                clipping_score = 1.0
            elif clipping_ratio <= 0.01:   # < 1% clipped
                clipping_score = 0.7
            else:
                clipping_score = 0.3       # Significant clipping
            
            quality_factors.append(("clipping_detection", clipping_score, 0.2))
            
            # Factor 4: Silence detection
            silence_threshold = 0.01
            silence_samples = np.sum(np.abs(audio_data) < silence_threshold)
            silence_ratio = silence_samples / len(audio_data)
            
            # Some silence is normal, too much indicates poor recording
            if silence_ratio <= 0.1:
                silence_score = 1.0
            elif silence_ratio <= 0.3:
                silence_score = 0.8
            else:
                silence_score = 0.4  # Too much silence
            
            quality_factors.append(("silence_analysis", silence_score, 0.3))
            
            # Calculate weighted acoustic quality
            acoustic_quality = sum(score * weight for _, score, weight in quality_factors)
            
            # Store detailed breakdown
            data_entry["acoustic_quality_breakdown"] = {
                factor_name: {"score": score, "weight": weight}
                for factor_name, score, weight in quality_factors
            }
            
            return acoustic_quality
            
        except ImportError:
            logger.warning(f"soundfile or numpy not available for acoustic analysis")
            return 0.5  # Default score when dependencies missing
        except Exception as e:
            logger.warning(f"Acoustic analysis failed for {audio_filepath}: {e}")
            return 0.0  # Failed analysis
```

### Language-Specific Metrics

```python
from dataclasses import dataclass, field

@dataclass
class LanguageSpecificQualityStage(CustomAudioQualityStage):
    """Language-specific quality assessment."""
    
    language_configs: dict = field(default_factory=lambda: {
        "en": {
            "expected_phonemes": 44,
            "syllable_rate": (2.0, 6.0),  # syllables per second
            "common_words": ["the", "and", "to", "of", "a", "in", "is", "it"]
        },
        "es": {
            "expected_phonemes": 24,
            "syllable_rate": (2.5, 7.0),
            "common_words": ["el", "la", "de", "que", "y", "en", "un", "es"]
        },
        "zh": {
            "expected_phonemes": 21,
            "syllable_rate": (1.5, 4.0),
            "common_characters": ["的", "是", "在", "有", "个", "人", "这", "中"]
        }
    })
    
    def calculate_quality_metric(self, data_entry: dict, audio_filepath: str, text: str,
                               pred_text: str, duration: float) -> float:
        """Calculate language-specific quality metrics."""
        
        language = data_entry.get("language", "en")
        lang_config = self.language_configs.get(language, self.language_configs["en"])
        
        quality_factors = []
        
        # Factor 1: Common word presence (indicates natural language)
        if language == "zh":
            # Character-based for Chinese
            common_elements = lang_config["common_characters"]
            text_elements = list(text)
        else:
            # Word-based for other languages
            common_elements = lang_config["common_words"]
            text_elements = text.lower().split()
        
        common_count = sum(
            1 for element in common_elements
            if element in text_elements
        )
        
        naturalness_score = min(1.0, common_count / 3.0)  # Normalize to 0-1
        quality_factors.append(("language_naturalness", naturalness_score, 0.4))
        
        # Factor 2: Syllable rate analysis (approximate)
        if duration > 0:
            # Rough syllable estimation
            vowel_count = sum(1 for char in text.lower() if char in "aeiouáéíóúàèìòù")
            syllable_rate = vowel_count / duration
            
            min_rate, max_rate = lang_config["syllable_rate"]
            
            if min_rate <= syllable_rate <= max_rate:
                syllable_score = 1.0
            elif min_rate * 0.7 <= syllable_rate <= max_rate * 1.3:
                syllable_score = 0.7
            else:
                syllable_score = 0.3
        else:
            syllable_score = 0.0
        
        quality_factors.append(("syllable_rate", syllable_score, 0.3))
        
        # Factor 3: Character/word distribution
        if language == "zh":
            # Chinese character frequency analysis
            unique_chars = len(set(text))
            total_chars = len(text)
            diversity_ratio = unique_chars / total_chars if total_chars > 0 else 0
            
            # Good diversity indicates natural text
            if 0.6 <= diversity_ratio <= 0.9:
                diversity_score = 1.0
            elif 0.4 <= diversity_ratio < 0.6 or 0.9 < diversity_ratio <= 1.0:
                diversity_score = 0.8
            else:
                diversity_score = 0.4
        else:
            # Word-based diversity for other languages
            words = text.lower().split()
            unique_words = len(set(words))
            total_words = len(words)
            diversity_ratio = unique_words / total_words if total_words > 0 else 0
            
            if 0.7 <= diversity_ratio <= 1.0:
                diversity_score = 1.0
            elif 0.5 <= diversity_ratio < 0.7:
                diversity_score = 0.8
            else:
                diversity_score = 0.5
        
        quality_factors.append(("lexical_diversity", diversity_score, 0.3))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in quality_factors)
        
        return total_score
```

## Multi-Modal Quality Metrics

### Audio-Text Consistency

```python
@dataclass
class AudioTextConsistencyStage(CustomAudioQualityStage):
    """Assess consistency between audio characteristics and text content."""
    metric_name: str = "audio_text_consistency"
    
    def calculate_quality_metric(self, data_entry: dict, audio_filepath: str, text: str,
                               pred_text: str, duration: float) -> float:
        """Calculate audio-text consistency score."""
        
        quality_factors = []
        
        # Factor 1: Duration-text length consistency
        char_count = len(text)
        if duration > 0 and char_count > 0:
            chars_per_second = char_count / duration
            
            # Expected range: 8-25 characters per second
            if 8.0 <= chars_per_second <= 25.0:
                length_consistency = 1.0
            elif 5.0 <= chars_per_second < 8.0 or 25.0 < chars_per_second <= 35.0:
                length_consistency = 0.7
            else:
                length_consistency = 0.3
        else:
            length_consistency = 0.0
        
        quality_factors.append(("length_consistency", length_consistency, 0.3))
        
        # Factor 2: Transcription agreement
        if text and pred_text:
            # Simple token overlap
            text_tokens = set(text.lower().split())
            pred_tokens = set(pred_text.lower().split())
            
            if text_tokens and pred_tokens:
                overlap = len(text_tokens.intersection(pred_tokens))
                union = len(text_tokens.union(pred_tokens))
                jaccard_similarity = overlap / union
                
                agreement_score = jaccard_similarity
            else:
                agreement_score = 0.0
        else:
            agreement_score = 0.0
        
        quality_factors.append(("transcription_agreement", agreement_score, 0.4))
        
        # Factor 3: Content coherence
        # Check for repeated words (indicates potential transcription errors)
        words = text.lower().split()
        if len(words) > 1:
            repeated_words = len(words) - len(set(words))
            repetition_ratio = repeated_words / len(words)
            
            if repetition_ratio <= 0.1:
                coherence_score = 1.0
            elif repetition_ratio <= 0.2:
                coherence_score = 0.8
            else:
                coherence_score = 0.4  # Too much repetition
        else:
            coherence_score = 0.5  # Single word
        
        quality_factors.append(("content_coherence", coherence_score, 0.3))
        
        # Calculate weighted consistency score
        total_score = sum(score * weight for _, score, weight in quality_factors)
        
        return total_score
```

## Ensemble Quality Metrics

### Combined Quality Assessment

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.tasks import AudioBatch
from nemo_curator.stages.audio.common import PreserveByValueStage

def create_ensemble_quality_pipeline() -> Pipeline:
    """Create pipeline with multiple quality assessment methods."""
    
    pipeline = Pipeline(name="ensemble_quality")
    
    # Calculate multiple quality metrics
    pipeline.add_stage(ConversationalQualityStage())
    pipeline.add_stage(AcousticQualityStage())
    pipeline.add_stage(AudioTextConsistencyStage())
    
    # Combine quality scores
    @processing_stage(name="ensemble_quality_combiner")
    def combine_quality_scores(audio_batch: AudioBatch) -> AudioBatch:
        for item in audio_batch.data:
            # Extract individual quality scores
            conversational_q = item.get("conversational_quality", 0.5)
            acoustic_q = item.get("acoustic_quality", 0.5)
            consistency_q = item.get("audio_text_consistency", 0.5)
            
            # Weighted ensemble
            ensemble_quality = (
                0.4 * conversational_q +
                0.3 * acoustic_q +
                0.3 * consistency_q
            )
            
            item["ensemble_quality"] = ensemble_quality
            
            # Quality tier assignment
            if ensemble_quality >= 0.8:
                item["quality_tier"] = "premium"
            elif ensemble_quality >= 0.6:
                item["quality_tier"] = "high"
            elif ensemble_quality >= 0.4:
                item["quality_tier"] = "medium"
            else:
                item["quality_tier"] = "low"
        
        return audio_batch
    
    pipeline.add_stage(combine_quality_scores)
    
    # Filter based on ensemble score
    pipeline.add_stage(
        PreserveByValueStage(
            input_value_key="ensemble_quality",
            target_value=0.6,
            operator="ge"
        )
    )
    
    return pipeline
```

## Metric Validation and Calibration

### Cross-Validation with Human Assessment

The calibration functions require additional dependencies. Install them with:

```bash
pip install scipy scikit-learn
```

```python
def calibrate_custom_metric(custom_scores: list[float], 
                          human_ratings: list[float]) -> dict:
    """Calibrate custom quality metric against human assessments."""
    
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(custom_scores, human_ratings)
    spearman_corr, spearman_p = spearmanr(custom_scores, human_ratings)
    
    # Calculate prediction accuracy
    mae = mean_absolute_error(human_ratings, custom_scores)
    
    calibration_results = {
        "correlation_analysis": {
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p
        },
        
        "prediction_accuracy": {
            "mean_absolute_error": mae,
            "rmse": np.sqrt(np.mean((np.array(custom_scores) - np.array(human_ratings)) ** 2))
        },
        
        "calibration_quality": {
            "strong_correlation": pearson_corr >= 0.7,
            "significant": pearson_p < 0.05,
            "acceptable_error": mae <= 0.2
        }
    }
    
    return calibration_results
```

### A/B Testing Framework

```{note}
The A/B testing framework requires numpy, which is already included as a core dependency.
```

```python
def ab_test_quality_metrics(audio_data: list[dict], 
                          metric_a_func: callable,
                          metric_b_func: callable,
                          validation_set: list[dict]) -> dict:
    """A/B test different quality metrics."""
    
    import numpy as np
    # Calculate metrics for both approaches
    scores_a = [metric_a_func(item) for item in audio_data]
    scores_b = [metric_b_func(item) for item in audio_data]
    
    # Test on validation set
    val_scores_a = [metric_a_func(item) for item in validation_set]
    val_scores_b = [metric_b_func(item) for item in validation_set]
    
    # Compare distributions
    comparison = {
        "metric_a": {
            "mean": np.mean(scores_a),
            "std": np.std(scores_a),
            "validation_mean": np.mean(val_scores_a)
        },
        "metric_b": {
            "mean": np.mean(scores_b), 
            "std": np.std(scores_b),
            "validation_mean": np.mean(val_scores_b)
        },
        "comparison": {
            "correlation": np.corrcoef(scores_a, scores_b)[0, 1],
            "mean_difference": np.mean(scores_a) - np.mean(scores_b),
            "consistency": np.std(scores_a) - np.std(scores_b)
        }
    }
    
    return comparison
```
