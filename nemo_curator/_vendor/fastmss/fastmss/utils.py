import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import lhotse
import numpy as np
from lhotse import MonoCut, Recording
from lhotse.parallel import parallel_map
from lhotse.supervision import AlignmentItem, SupervisionSegment
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def concatenate_audio_with_crossfade(audio_segments, crossfade_samples=1024):
    """
    Concatenate multiple audio segments with crossfading at boundaries.

    Parameters:
    -----------
    audio_segments : list of np.ndarray
        List of 1D numpy arrays representing audio signals to concatenate
    crossfade_samples : int, optional
        Number of samples for crossfade transition (default: 1024)

    Returns:
    --------
    np.ndarray
        Concatenated audio with crossfaded boundaries
    """
    if len(audio_segments) == 0:
        return np.array([])

    if len(audio_segments) == 1:
        return audio_segments[0]

    # Allocate more space than needed to handle variable crossfade lengths
    # We'll trim to actual size at the end
    max_length = sum(len(seg) for seg in audio_segments)
    output = np.zeros(max_length, dtype=audio_segments[0].dtype)
    current_pos = 0

    # Process first segment (no fade in)
    first_seg = audio_segments[0]
    output[: len(first_seg)] = first_seg
    current_pos = len(first_seg)

    # Create crossfade windows (cosine fade)
    fade_in = 0.5 * (
        1 - np.cos(np.pi * np.arange(crossfade_samples) / crossfade_samples)
    )
    fade_out = 0.5 * (
        1 + np.cos(np.pi * np.arange(crossfade_samples) / crossfade_samples)
    )

    # Process remaining segments with crossfading
    for seg in audio_segments[1:]:
        # Ensure crossfade doesn't exceed segment length
        actual_crossfade = min(crossfade_samples, len(seg) // 4, current_pos)

        if actual_crossfade > 0:
            # Calculate crossfade region
            crossfade_start = current_pos - actual_crossfade

            # Fade out the previous segment
            output[crossfade_start:current_pos] *= fade_out[:actual_crossfade]

            # Fade in the new segment and add
            output[crossfade_start:current_pos] += (
                seg[:actual_crossfade] * fade_in[:actual_crossfade]
            )

            # Add the rest of the new segment
            remaining_seg = seg[actual_crossfade:]
            output[current_pos : current_pos + len(remaining_seg)] = remaining_seg
            current_pos += len(remaining_seg)
        else:
            # No crossfade possible
            output[current_pos : current_pos + len(seg)] = seg
            current_pos += len(seg)

    # Trim to actual length used
    return output[:current_pos]


def repeat_audio_with_crossfade(audio, target_length, crossfade_samples=1024):
    """
    Repeat audio to reach target length with crossfading at boundaries.

    Parameters:
    -----------
    audio : np.ndarray
        1D numpy array representing the audio signal
    target_length : int
        Desired length of the output audio in samples
    crossfade_samples : int, optional
        Number of samples for crossfade transition (default: 1024)

    Returns:
    --------
    np.ndarray
        Audio repeated to target length with crossfaded boundaries
    """
    if len(audio) >= target_length:
        return audio[:target_length]

    # Ensure crossfade doesn't exceed audio length
    crossfade_samples = min(crossfade_samples, len(audio) // 4)

    # Create crossfade window (cosine fade)
    fade_in = 0.5 * (
        1 - np.cos(np.pi * np.arange(crossfade_samples) / crossfade_samples)
    )
    fade_out = 0.5 * (
        1 + np.cos(np.pi * np.arange(crossfade_samples) / crossfade_samples)
    )

    # Initialize output array
    output = np.zeros(target_length, dtype=audio.dtype)
    current_pos = 0

    # First copy (no fade in)
    copy_length = min(len(audio), target_length)
    output[:copy_length] = audio[:copy_length]
    current_pos = copy_length

    # Repeat with crossfading
    while current_pos < target_length:
        remaining = target_length - current_pos
        copy_length = min(len(audio), remaining)

        # Calculate crossfade region
        crossfade_start = current_pos - crossfade_samples
        crossfade_end = current_pos

        if crossfade_start >= 0 and copy_length > crossfade_samples:
            # Apply crossfade
            # Fade out the previous segment
            output[crossfade_start:crossfade_end] *= fade_out

            # Fade in the new segment and add
            new_segment = audio[:copy_length].copy()
            new_segment[:crossfade_samples] *= fade_in
            output[
                crossfade_start : crossfade_start + crossfade_samples
            ] += new_segment[:crossfade_samples]

            # Add the rest of the new segment
            if copy_length > crossfade_samples:
                output[current_pos : current_pos + copy_length - crossfade_samples] = (
                    new_segment[crossfade_samples:]
                )
        else:
            # No crossfade needed (edge cases)
            output[current_pos : current_pos + copy_length] = audio[:copy_length]

        current_pos += copy_length

    return output


def split_recording_by_channels(recording: Recording) -> List[MonoCut]:
    """
    Split a multi-channel Recording into separate single-channel Cuts.
    Args:
        recording: Recording with multi-channel AudioSource
    Returns:
        List of Cuts, one per channel
    """
    # Get the multi-channel source
    source = recording.sources[0]
    channels = source.channels
    single_channel_cuts = []

    for channel in channels:
        # Create Cut for this specific channel
        channel_cut = MonoCut(
            id=f"{recording.id}_ch{channel}",
            start=0,
            duration=recording.duration,
            channel=channel,  # This tells Lhotse which channel to extract
            recording=recording,  # Reference to the original multi-channel recording
        )
        single_channel_cuts.append(channel_cut)

    return single_channel_cuts


def _alignments_to_recording_frame(
    items: List[AlignmentItem],
    monocut: MonoCut,
    supervision: SupervisionSegment,
    tol: float = 0.05,
) -> List[AlignmentItem]:
    """
    Make sure word alignments are expressed w.r.t. the start of the recording,
    i.e. that they fall inside ``[monocut.start, monocut.end]``.

    If they do not, try to detect whether they were stored relative to the cut
    or relative to the supervision (both common when a forced aligner was run
    on pre-cut audio) and shift them accordingly. Raise a descriptive error if
    no frame of reference makes them fit the cut, instead of letting the
    splitting logic produce segments with negative durations.
    """
    cut_start, cut_end = monocut.start, monocut.start + monocut.duration

    def _fits(alis: List[AlignmentItem]) -> bool:
        return all(
            a.start >= cut_start - tol and a.start + a.duration <= cut_end + tol
            for a in alis
        )

    if _fits(items):
        return items

    # Candidate frames, tried in order: relative to the cut, relative to the
    # supervision (whose start is itself relative to the cut).
    candidates = [
        ("cut", cut_start),
        ("supervision", cut_start + supervision.start),
    ]
    for name, offset in candidates:
        shifted = [a.with_offset(offset) for a in items]
        if _fits(shifted):
            logging.getLogger(__name__).warning(
                f"Cut {monocut.id}: word alignments appear to be relative to the "
                f"{name} (not to the recording); shifting them by {offset:.3f}s. "
                "Consider fixing the manifests so alignments are w.r.t. the recording."
            )
            return shifted

    raise ValueError(
        f"Cut {monocut.id} (start={cut_start:.3f}, end={cut_end:.3f}) has word "
        f"alignments spanning [{min(a.start for a in items):.3f}, "
        f"{max(a.start + a.duration for a in items):.3f}] that do not fit inside "
        "the cut under any frame of reference (recording / cut / supervision). "
        "Fix the alignments in the manifests or disable splitting with "
        "split_fa_factor=null."
    )


def split_monocut_at_pauses(
    monocut: MonoCut, pause_threshold: float = 0.2, drop_without_alignment=True
) -> List[MonoCut]:
    """
    Split a MonoCut at pauses longer than the specified threshold while preserving alignments.

    When splitting at gaps:
    - The previous segment ends at gap_start + gap_duration/2
    - The next segment starts at gap_start + gap_duration/2
    - Leading silence longer than threshold creates a separate silent segment
    - Trailing silence longer than threshold creates a separate silent segment
    - For gaps shorter than threshold, segments include the full gap

    Args:
        monocut: The MonoCut to split
        pause_threshold: Minimum pause duration in seconds to split at (default: 0.2s = 200ms)

    Returns:
        List of MonoCut objects split at long pauses
    """

    if not monocut.supervisions or not monocut.supervisions[0].alignment:
        # No alignment info, return original cut
        if drop_without_alignment:
            return []
        else:
            return [monocut]

    elif pause_threshold is None:
        return [monocut]

    supervision = monocut.supervisions[0]
    word_alignments = supervision.alignment["word"]

    # Filter out empty symbols upfront
    valid_alignments = [item for item in word_alignments if item.symbol.strip() != ""]

    if len(valid_alignments) == 0:
        return [monocut]  # No valid words, return original

    # The splitting logic below compares alignment times against the cut's
    # absolute (recording-level) start/end, so alignments must be expressed
    # w.r.t. the start of the recording (Lhotse's documented convention).
    # Manifests produced by running a forced aligner on already-cut audio
    # often store them relative to the cut/supervision instead; when the cut
    # does not start at 0 this silently yields negative gaps and durations.
    valid_alignments = _alignments_to_recording_frame(
        valid_alignments, monocut, supervision
    )

    # Check for leading silence
    first_word = valid_alignments[0]
    leading_silence = first_word.start - monocut.start

    # Check for trailing silence
    last_word = valid_alignments[-1]
    last_word_end = last_word.start + last_word.duration
    monocut_end = monocut.start + monocut.duration
    trailing_silence = monocut_end - last_word_end

    # Find all potential split points (including leading/trailing silence)
    split_points = []
    segment_boundaries = (
        []
    )  # (start_word_idx, end_word_idx, segment_start_time, segment_end_time)

    # Handle leading silence
    current_segment_start_time = monocut.start
    current_word_start_idx = 0

    if leading_silence >= pause_threshold:
        # Leading silence gets split in the middle
        split_time = monocut.start + leading_silence / 2
        split_points.append(split_time)
        current_segment_start_time = split_time

    # Find gaps between words that exceed threshold
    for i in range(len(valid_alignments) - 1):
        current_item = valid_alignments[i]
        next_item = valid_alignments[i + 1]

        current_end = current_item.start + current_item.duration
        gap_start = current_end
        gap_end = next_item.start
        gap_duration = gap_end - gap_start

        if gap_duration >= pause_threshold:
            # End current segment at middle of gap
            split_time = gap_start + gap_duration / 2

            # Add current segment
            segment_boundaries.append(
                (
                    current_word_start_idx,
                    i,  # end at current word (inclusive)
                    current_segment_start_time,
                    split_time,
                )
            )

            # Start new segment at same split point
            current_segment_start_time = split_time
            current_word_start_idx = i + 1
            split_points.append(split_time)

    # Handle the final segment
    final_segment_end_time = monocut_end
    if trailing_silence >= pause_threshold:
        # Split trailing silence in the middle
        split_time = last_word_end + trailing_silence / 2
        # End the words segment at the split
        segment_boundaries.append(
            (
                current_word_start_idx,
                len(valid_alignments) - 1,
                current_segment_start_time,
                split_time,
            )
        )
        split_points.append(split_time)
    else:
        # Include trailing silence in the final segment
        segment_boundaries.append(
            (
                current_word_start_idx,
                len(valid_alignments) - 1,
                current_segment_start_time,
                final_segment_end_time,
            )
        )

    # If no splits were made, return original
    if not split_points:
        adjusted_alignments = []
        supervision = monocut.supervisions[0]
        for item in supervision.alignment["word"]:
            adjusted_item = AlignmentItem(
                symbol=item.symbol,
                start=item.start - monocut.start,
                duration=item.duration,
                score=item.score,
            )
            adjusted_alignments.append(adjusted_item)
        new_supervision = SupervisionSegment(
            id=f"{supervision.id}",
            recording_id=supervision.recording_id,
            start=0,  # Relative to the new cut
            duration=supervision.duration,
            channel=supervision.channel,
            text=supervision.text,
            language=supervision.language,
            speaker=supervision.speaker,
            gender=supervision.gender,
            custom=supervision.custom,
            alignment={"word": adjusted_alignments},
        )
        monocut.supervisions = [new_supervision]
        return [monocut]

    # Create new MonoCuts for each segment
    result_cuts = []
    for seg_idx, (
        start_word_idx,
        end_word_idx,
        segment_start,
        segment_end,
    ) in enumerate(segment_boundaries):
        segment_words = valid_alignments[start_word_idx : end_word_idx + 1]

        if not segment_words:
            continue

        segment_duration = segment_end - segment_start
        if segment_duration <= 0:
            raise ValueError(
                f"Cut {monocut.id}: computed a segment with non-positive duration "
                f"({segment_duration:.3f}s, words={[w.symbol for w in segment_words]}). "
                "This indicates inconsistent alignment / cut boundaries in the manifests."
            )

        # Adjust alignment times to be relative to segment start
        adjusted_alignments = []
        for item in segment_words:
            adjusted_item = AlignmentItem(
                symbol=item.symbol,
                start=item.start - segment_start,
                duration=item.duration,
                score=item.score,
            )
            adjusted_alignments.append(adjusted_item)

        # Create segment text
        segment_text = " ".join(item.symbol for item in segment_words)

        # Create new supervision with adjusted alignments
        new_supervision = SupervisionSegment(
            id=f"{supervision.id}-{seg_idx:03d}",
            recording_id=supervision.recording_id,
            start=0,  # Relative to the new cut
            duration=segment_duration,
            channel=supervision.channel,
            text=segment_text,
            language=supervision.language,
            speaker=supervision.speaker,
            gender=supervision.gender,
            custom=supervision.custom,
            alignment={"word": adjusted_alignments},
        )

        # Create new MonoCut
        new_cut = MonoCut(
            id=f"{monocut.id}-{seg_idx:03d}",
            start=segment_start,
            duration=segment_duration,
            channel=monocut.channel,
            supervisions=[new_supervision],
            recording=monocut.recording,
            custom=monocut.custom,
        )

        result_cuts.append(new_cut)

    return result_cuts


def split_monocuts_batch(
    monocuts: List[MonoCut], pause_threshold: float = 0.2, num_jobs: int = 1
) -> List[MonoCut]:
    """
    Split a list of MonoCuts at pauses longer than the specified threshold.

    Args:
        monocuts: List of MonoCuts to split
        pause_threshold: Minimum pause duration in seconds to split at (default: 0.2s = 200ms)

    Returns:
        List of all resulting MonoCut objects
    """
    result = []
    helper_func = partial(split_monocut_at_pauses, pause_threshold=pause_threshold)
    n_monocuts = len(monocuts)
    monocuts = iter(monocuts)
    for elem in tqdm(
        parallel_map(helper_func, monocuts, num_jobs=num_jobs),
        total=n_monocuts,
        desc="Splitting cuts using forced alignment.",
    ):
        result.extend(elem)
    return lhotse.CutSet(result)
