"""Audio feature computation: fbank, resampling, CMN, and learned frontends."""

import logging
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

logger = logging.getLogger(__name__)


def load_audio(
    wav_path: str,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Load a wav file and return a 1-D float32 tensor at *target_sr*.

    Returns shape ``(num_samples,)``.
    """
    import soundfile as sf

    pcm, sr = sf.read(wav_path, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    pcm = torch.from_numpy(pcm).unsqueeze(0)  # (1, T)
    if sr != target_sr:
        pcm = torchaudio.transforms.Resample(sr, target_sr)(pcm)
    return pcm.squeeze(0)  # (T,)


def compute_fbank(
    pcm: torch.Tensor,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    apply_cmn: bool = True,
) -> torch.Tensor:
    """Compute log-mel filterbank features from a 1-D PCM tensor.

    Returns shape ``(num_frames, num_mel_bins)``.
    """
    waveform = pcm.unsqueeze(0) if pcm.dim() == 1 else pcm  # (1, T)
    feat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=25,
        frame_shift=10,
        sample_frequency=sample_rate,
    )
    if apply_cmn:
        feat = feat - feat.mean(dim=0)
    return feat  # (T', D)


def compute_frontend_features(
    pcm: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Run a learned frontend (s3prl / w2vbert / whisper_encoder) on raw PCM.

    The frontend is expected to be attached to ``model.frontend``.
    Returns shape ``(1, T', D)`` ready for the speaker backbone.
    """
    waveform = pcm.unsqueeze(0).to(device)  # (1, T)
    wav_len = torch.LongTensor([waveform.shape[1]]).to(device)
    with torch.no_grad():
        features, _ = model.frontend(waveform, wav_len)
    return features  # (1, T', D)


def compute_features(
    pcm: torch.Tensor,
    model: torch.nn.Module,
    frontend_type: str,
    device: torch.device,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    apply_cmn: bool = True,
) -> torch.Tensor:
    """Unified entry point: compute features for any frontend type.

    Returns shape ``(T', D)`` (unbatched).
    """
    if frontend_type == "fbank":
        return compute_fbank(pcm, sample_rate, num_mel_bins, apply_cmn)
    else:
        feats = compute_frontend_features(pcm, model, device)
        return feats.squeeze(0)  # (T', D)
