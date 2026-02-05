# Image Filtering Thresholds

Guide for tuning aesthetic and NSFW filter thresholds.

## Aesthetic Score

The aesthetic filter uses a CLIP-based model to predict visual quality.

### Score Distribution

| Score Range | Quality | Typical Content |
|-------------|---------|-----------------|
| 0.0 - 0.3 | Low | Blurry, poorly composed, artifacts |
| 0.3 - 0.5 | Moderate | Average photos, screenshots |
| 0.5 - 0.7 | Good | Well-composed photos |
| 0.7 - 0.9 | High | Professional photography |
| 0.9 - 1.0 | Exceptional | Award-winning images |

### Recommended Thresholds

| Use Case | Threshold | Retention Rate |
|----------|-----------|----------------|
| Web scrape cleanup | 0.3 | ~70-80% |
| General training | 0.5 | ~40-50% |
| High-quality dataset | 0.7 | ~15-25% |
| Premium subset | 0.85 | ~5-10% |

### Configuration

```yaml
- _target_: nemo_curator.stages.image.filters.aesthetic_filter.ImageAestheticFilterStage
  model_dir: ${model_dir}
  score_threshold: 0.5  # Adjust based on use case
  num_gpus_per_worker: 0.25
```

## NSFW Score

The NSFW filter detects potentially unsafe content.

### Score Interpretation

| Score | Meaning |
|-------|---------|
| 0.0 - 0.2 | Safe content |
| 0.2 - 0.5 | Possibly suggestive |
| 0.5 - 0.8 | Likely inappropriate |
| 0.8 - 1.0 | Explicit content |

### Recommended Thresholds

| Use Case | Threshold | Notes |
|----------|-----------|-------|
| Strict (enterprise) | 0.2 | May have false positives |
| Standard | 0.5 | Balanced |
| Permissive | 0.8 | Only obvious NSFW |

### Configuration

```yaml
- _target_: nemo_curator.stages.image.filters.nsfw_filter.ImageNSFWFilterStage
  model_dir: ${model_dir}
  score_threshold: 0.5  # Images ABOVE this are filtered out
  num_gpus_per_worker: 0.25
```

## Combined Filtering

When using both filters together:

```yaml
# Keep high-quality, safe images
aesthetic_threshold: 0.5  # Keep score >= 0.5
nsfw_threshold: 0.5       # Remove score >= 0.5
```

## Tuning Tips

1. **Start conservative**: Begin with moderate thresholds (0.5)
2. **Sample first**: Process a small subset to check results
3. **Adjust iteratively**: Fine-tune based on inspection
4. **Consider domain**: Art datasets may need different thresholds than photos
