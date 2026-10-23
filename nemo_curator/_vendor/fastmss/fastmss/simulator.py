import logging
import os
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import lhotse
import numpy as np
import soundfile as sf
from lhotse import AudioSource, Recording
from lhotse.supervision import AlignmentItem, SupervisionSegment
from lhotse.utils import uuid4
from scipy.signal import convolve, firwin

from fastmss.hmm_turn_taking import TransitionParams, TransitionType

from .utils import concatenate_audio_with_crossfade, repeat_audio_with_crossfade

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe audio cache at module level
_audio_cache = {}
_audio_cache_lock = threading.Lock()


def cached_load_audio(cut):
    """Load audio for a cut, using a thread-safe in-memory cache.
    Returns a copy so callers can mutate freely."""
    cid = cut.id
    if cid in _audio_cache:
        return _audio_cache[cid].copy()
    audio = cut.load_audio()
    with _audio_cache_lock:
        if len(_audio_cache) < 4096:
            _audio_cache[cid] = audio.copy()
    return audio


class ConversationalMeetingSimulator:
    def __init__(
        self,
        cfg,
        output_dir,
        spk2cuts,
        rirs=None,
        noise_files=None,
    ):
        super().__init__()

        self.output_dir = output_dir
        self.cfg = cfg
        self.epsilon = 0.03  # For truncated exponential distribution
        self.rirs = rirs
        self.noise_files = noise_files  # optional background noise.
        self.start_offset = getattr(self.cfg, "start_offset", 0)
        # if not available, gaussian is used.
        self.hmm_params = (
            TransitionParams(**cfg.hmm_params)
            if hasattr(cfg, "hmm_params")
            else TransitionParams()
        )
        if cfg.hmm_fit_transitions_to:
            self.hmm_params.fit(lhotse.load_manifest(cfg.hmm_fit_transitions_to))

        if cfg.boost_overlap_factor is not None:
            self.hmm_params.boost_overlap_factor(cfg.boost_overlap_factor)

        self.spk2cuts = spk2cuts
        self.speakers = list(spk2cuts.keys())

        # Validate Markov matrix
        if self.cfg.use_markov:
            row_sums = np.sum(self.hmm_params.p_markov, axis=1)

            if not np.allclose(row_sums, 1.0):
                logger.warning("Markov matrix rows don't sum to 1, normalizing...")
                self.hmm_params.p_markov = self.hmm_params.p_markov / (
                    row_sums[:, None] + 1e-8
                )

    def sample_exponential_duration(self, beta: float) -> float:
        """Sample duration from exponential distribution"""
        return np.random.exponential(beta)

    def sample_overlap_ratio(self, beta: float) -> float:
        """Sample overlap ratio from truncated exponential distribution"""
        # Sample from exponential and truncate to [epsilon, 1-epsilon]
        ratio = np.random.exponential(beta)
        return np.clip(ratio, self.epsilon, 1.0 - self.epsilon)

    def get_offset(self, prev_cut, prev_offset, prev_offset_spk, transition_type):

        prev_duration = prev_cut.duration

        if transition_type == TransitionType.TURN_HOLD:
            pause_duration = self.sample_exponential_duration(self.hmm_params.beta_th)
            return max(
                prev_offset + prev_cut.duration + pause_duration, prev_offset_spk
            )

        elif transition_type == TransitionType.TURN_SWITCH:
            pause_duration = self.sample_exponential_duration(self.hmm_params.beta_ts)
            return max(
                prev_offset + prev_cut.duration + pause_duration, prev_offset_spk
            )

        elif transition_type == TransitionType.INTERRUPTION:
            overlap_ratio = self.sample_overlap_ratio(self.hmm_params.beta_ir)
            overlap_duration = overlap_ratio * prev_duration

            return max(
                prev_offset + prev_cut.duration - overlap_duration, prev_offset_spk
            )

        elif transition_type == TransitionType.BACKCHANNEL:
            # Use overlap_ratio sampling instead of random uniform
            overlap_ratio = self.sample_overlap_ratio(self.hmm_params.beta_bc)
            start_offset = overlap_ratio * prev_duration

            return max(prev_offset + prev_cut.duration - start_offset, prev_offset_spk)

    def select_transition_type(
        self, prev_transition: Optional[TransitionType] = None
    ) -> TransitionType:
        """Select next transition type based on random or Markov selection"""
        if self.cfg.use_markov and prev_transition is not None:
            # Use Markov chain
            prev_idx = list(TransitionType).index(prev_transition)
            probs = self.hmm_params.p_markov[prev_idx]
        else:
            # Use independent random selection
            probs = self.hmm_params.p_ind

        # Sample transition type
        transition_idx = np.random.choice(len(TransitionType), p=probs)
        return list(TransitionType)[transition_idx]

    def find_valid_offset(
        self,
        proposed_offset,
        new_duration,
        new_speaker,
        utterances,
        offsets,
        durations,
        max_concurrent,
    ):
        """
        Find the earliest valid offset that doesn't exceed max concurrent speakers.
        Vectorized implementation for speed.

        Args:
            proposed_offset: Initially proposed offset from get_offset()
            new_duration: Duration of the new utterance
            new_speaker: Speaker ID of the new utterance
            utterances: List of existing MonoCut objects
            offsets: List of start times for existing utterances
            durations: List of durations for existing utterances
            max_concurrent: Maximum allowed concurrent speakers

        Returns:
            Adjusted offset that respects the max_concurrent constraint
        """
        if max_concurrent is None or max_concurrent <= 0:
            return proposed_offset

        # Build arrays once
        off_arr = np.array(offsets)
        dur_arr = np.array(durations)
        end_arr = off_arr + dur_arr
        speakers = [utt.supervisions[0].speaker for utt in utterances]

        def max_concurrent_in_range(start, end):
            """Check max concurrent speakers over a range using event boundaries."""
            # Only consider utterances that overlap [start, end)
            mask = (off_arr < end) & (end_arr > start)
            if not np.any(mask):
                return 1  # just the new speaker

            # Collect all boundary events within [start, end)
            rel_starts = np.maximum(off_arr[mask], start)
            rel_ends = np.minimum(end_arr[mask], end)
            idx = np.where(mask)[0]

            # Check at each boundary point
            boundary_times = np.unique(np.concatenate([rel_starts, rel_ends]))
            # Use midpoints to check inside intervals
            if len(boundary_times) < 2:
                check_times = boundary_times
            else:
                check_times = (boundary_times[:-1] + boundary_times[1:]) / 2

            max_c = 1
            for t in check_times:
                active = (off_arr[mask] <= t) & (end_arr[mask] > t)
                active_spk = set(speakers[i] for i in idx[active])
                c = len(active_spk) + (1 if new_speaker not in active_spk else 0)
                if c > max_c:
                    max_c = c
            return max_c

        proposed_end = proposed_offset + new_duration
        if max_concurrent_in_range(proposed_offset, proposed_end) <= max_concurrent:
            return proposed_offset

        # Find the next valid time from utterance end times
        end_times = end_arr[end_arr > proposed_offset]

        if len(end_times) == 0:
            return proposed_offset

        end_times = np.sort(end_times)

        for candidate_offset in end_times:
            candidate_end = candidate_offset + new_duration
            if (
                max_concurrent_in_range(candidate_offset, candidate_end)
                <= max_concurrent
            ):
                return candidate_offset

        return max(
            off_arr[-1] + dur_arr[-1] if len(off_arr) > 0 else 0, proposed_offset
        )

    def create_fir_highpass(self, cutoff_freq, num_taps=101, window="hamming"):
        """
        Create a FIR highpass filter using the window method.

        Parameters:
        cutoff_freq (float): Cutoff frequency in Hz
        sample_rate (float): Sample rate in Hz
        num_taps (int): Number of filter taps (filter length). Should be odd.
        window (str): Window function ('hamming', 'hann', 'blackman', etc.)

        Returns:
        numpy.ndarray: FIR filter coefficients
        """
        sample_rate = self.cfg.samplerate
        # Normalize the cutoff frequency (0 to 1, where 1 is Nyquist frequency)
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        # Create lowpass filter first
        lowpass_coeffs = firwin(num_taps, normalized_cutoff, window=window)

        # Convert to highpass by spectral inversion
        highpass_coeffs = -lowpass_coeffs
        highpass_coeffs[num_taps // 2] += 1  # Add impulse at center

        return highpass_coeffs

    def add_gaussian_noise(self, audio, min_speech_level_db, range_db_offset=(-15, 3)):
        """
        Add Gaussian noise to audio signal. Noise level is capped at 5 dB below
        the minimum speech level.

        Args:
            audio: Input audio signal (numpy array)
            min_speech_level_db: Minimum speech level in dB - noise won't exceed this by more than 5 dB

        Returns:
            Audio signal with added noise
        """
        if len(audio) == 0:
            return audio

        # Calculate maximum allowed noise level (5 dB below min speech level)
        max_noise_level_db = min_speech_level_db + np.random.uniform(*range_db_offset)

        # Generate random noise level between 0 and max allowed
        # Using a range that gives reasonable noise levels
        # noise_level_db = np.random.uniform(max_noise_level_db - 20, max_noise_level_db)
        noise_rms = 10 ** (max_noise_level_db / 20.0)

        # Generate Gaussian noise
        noise = np.random.normal(0, noise_rms, audio.shape)

        # Add noise to signal
        noisy_audio = audio + noise

        return noisy_audio

    def add_real_noise(self, audio, min_speech_level_db, range_db_offset=(-15, 3)):
        # Check if multi-noise sampling is enabled
        multi_noise_enabled = getattr(self.cfg, "multi_noise_sampling", True)

        if len(audio) == 0:
            return audio

        tgt_len = audio.shape[-1]

        if multi_noise_enabled:
            # Sample multiple noise files to cover the entire meeting duration
            # Account for crossfading by sampling a bit more than needed
            crossfade_samples = int(0.5 * self.cfg.samplerate)  # 0.5 second crossfade
            noise_segments = []
            total_sampled = 0

            # Keep sampling until we have enough (accounting for crossfading loss)
            while total_sampled < tgt_len:
                # Sample a noise file
                c_noise_file = str(np.random.choice(self.noise_files))
                info = sf.SoundFile(c_noise_file)
                assert info.samplerate == self.cfg.samplerate

                # Determine how much to read from this file
                available_length = len(info)

                # On first iteration, read more; on subsequent, read chunks
                if len(noise_segments) == 0:
                    # First segment - read a good chunk
                    read_length = min(available_length, tgt_len)
                else:
                    # Subsequent segments - read chunks, accounting for crossfade loss
                    remaining_needed = (
                        tgt_len
                        - total_sampled
                        + (len(noise_segments) * crossfade_samples)
                    )
                    read_length = min(
                        available_length, max(remaining_needed, crossfade_samples * 2)
                    )

                if available_length > read_length:
                    # Read a random segment from this file
                    cursor = np.random.randint(0, available_length - read_length)
                    c_noise, _ = sf.read(
                        c_noise_file, start=cursor, stop=cursor + read_length
                    )
                else:
                    # Read the entire file
                    c_noise, _ = sf.read(c_noise_file)

                # Handle multi-channel noise by selecting a random channel
                if c_noise.ndim > 1:
                    if c_noise.shape[-1] > 1:
                        ch = np.random.randint(0, c_noise.shape[-1])
                        c_noise = c_noise[:, ch]

                # Normalize each segment before concatenation to unit RMS
                rms = np.sqrt(np.mean(c_noise**2))
                c_noise = c_noise / (rms + 1e-8)

                noise_segments.append(c_noise)
                total_sampled += len(c_noise)

            # Concatenate all noise segments with crossfading
            c_noise = concatenate_audio_with_crossfade(
                noise_segments, crossfade_samples
            )

            # Ensure we have exactly the target length
            if len(c_noise) < tgt_len:
                # Should rarely happen now, but pad if needed
                c_noise = np.pad(c_noise, (0, tgt_len - len(c_noise)), mode="constant")
            elif len(c_noise) > tgt_len:
                # Trim to exact length
                c_noise = c_noise[:tgt_len]

            # Re-normalize after concatenation (crossfading changes the RMS)
            # Use RMS normalization instead of std for consistency with speech normalization
            rms = np.sqrt(np.mean(c_noise**2))
            c_noise = c_noise / (rms + 1e-8)
        else:
            # Original single-noise behavior
            c_noise_file = str(np.random.choice(self.noise_files))
            info = sf.SoundFile(c_noise_file)
            assert info.samplerate == self.cfg.samplerate

            if len(info) > tgt_len:
                cursor = np.random.randint(0, len(info) - tgt_len)
            else:
                cursor = 0

            c_noise, _ = sf.read(c_noise_file, start=cursor, stop=cursor + tgt_len)

            if c_noise.ndim > 1:
                if c_noise.shape[-1] > 1:
                    ch = np.random.randint(0, c_noise.shape[-1])
                    c_noise = c_noise[:, ch]

            if c_noise.shape[0] < tgt_len:
                c_noise = repeat_audio_with_crossfade(c_noise, tgt_len, 16000)

            # Normalize before applying gain
            # Use RMS normalization instead of std for consistency with speech normalization
            rms = np.sqrt(np.mean(c_noise**2))
            c_noise = c_noise / (rms + 1e-8)

        # Calculate maximum allowed noise level (5 dB below min speech level)
        max_noise_level_db = min_speech_level_db + np.random.uniform(*range_db_offset)

        # Apply target noise level (gain)
        noise_rms = 10 ** (max_noise_level_db / 20.0)
        c_noise = c_noise * noise_rms

        # Add noise to signal
        noisy_audio = audio + c_noise

        return noisy_audio

    def normalize_to(self, audio, target_level_db):
        """
        Normalize audio to a target RMS level in dB.

        Args:
            audio: Input audio signal (numpy array)
            target_level_db: Target RMS level in dB (can be single value or array)

        Returns:
            Normalized audio signal
        """
        if len(audio) == 0:
            return audio

        # Calculate current peak RMS level
        rms = np.sqrt(np.mean(audio**2))

        # Avoid division by zero
        if rms == 0:
            return audio, 1e-8

        # Convert target level from dB to linear scale
        # Assuming 0 dB corresponds to RMS = 1.0
        if isinstance(target_level_db, (list, np.ndarray)):
            target_level_db = np.random.choice(target_level_db)

        target_rms_linear = 10 ** (target_level_db / 20.0)

        # Calculate gain needed
        gain = target_rms_linear / rms

        # Apply gain
        normalized_audio = audio * gain

        return normalized_audio, gain

    def gen_audio(self, indx):
        # Resolve OmegaConf to SimpleNamespace once to avoid __getattr__ overhead
        if not isinstance(self.cfg, SimpleNamespace):
            from omegaconf import OmegaConf

            self.cfg = SimpleNamespace(**OmegaConf.to_container(self.cfg, resolve=True))

        min_spk, max_spk = self.cfg.min_max_spk
        n_speakers = np.random.randint(min_spk, max_spk + 1)
        target_dur = self.cfg.duration

        sampled_spk = np.random.choice(self.speakers, n_speakers, replace=False)

        current_time = 0
        utt_indx = 0
        prev_transition = None
        prev_cut = None
        prev_speaker = None
        seen_speakers = set()
        prev_offset_spk = {}

        utterances = []
        offsets = []
        transition_types = []

        # Conditional FIR: only create fir_highpass if use_fir is True
        if self.cfg.use_fir:
            fir_highpass = self.create_fir_highpass(40, 63)

        if self.rirs is not None:
            c_room_rirs = self.rirs[np.random.randint(0, len(self.rirs))]

        while current_time < target_dur or len(seen_speakers) < len(sampled_spk):

            transition_type = self.select_transition_type(prev_transition)
            if transition_type == TransitionType.TURN_HOLD:
                # Same speaker
                if prev_speaker is not None:
                    current_speaker = prev_speaker
                else:
                    # random choice of next speaker first utterance
                    current_speaker = np.random.choice(sampled_spk)
            else:
                # Different speaker
                if prev_speaker is not None:
                    available_speakers = [s for s in sampled_spk if s != prev_speaker]
                    current_speaker = np.random.choice(available_speakers)
                else:
                    current_speaker = np.random.choice(sampled_spk)

            cut = np.random.choice(self.spk2cuts[current_speaker])

            # Load audio # optionally perturb speed here ? I think it is faster with torchaudio.
            if self.cfg.speed_perturb:
                # keep these fixed
                factor = np.random.uniform(*self.cfg.speed_perturb_range)
                cut = cut.perturb_speed(factor)

            c_spk = cut.supervisions[0].speaker

            if utt_indx == 0:
                initial_offset = np.random.uniform(self.start_offset)
                offsets.append(initial_offset)
                utterances.append(cut)
                transition_types.append(None)
            else:
                if not c_spk in prev_offset_spk.keys():
                    c_offset = self.get_offset(
                        prev_cut, prev_offset, 0.0, transition_type
                    )
                else:
                    c_offset = self.get_offset(
                        prev_cut, prev_offset, prev_offset_spk[c_spk], transition_type
                    )

                # Apply max concurrent speakers constraint if configured
                max_concurrent = getattr(self.cfg, "max_concurrent_speakers", None)
                if max_concurrent is not None and max_concurrent > 0:
                    # Build durations list for existing utterances
                    durations = [u.duration for u in utterances]
                    # Find valid offset respecting the constraint
                    c_offset = self.find_valid_offset(
                        c_offset,
                        cut.duration,
                        c_spk,
                        utterances,
                        offsets,
                        durations,
                        max_concurrent,
                    )

                offsets.append(c_offset)
                utterances.append(cut)
                transition_types.append(transition_type)

            utt_indx += 1
            prev_cut = cut
            prev_speaker = c_spk
            seen_speakers.add(prev_speaker)
            prev_transition = transition_type
            prev_offset = offsets[-1]
            if prev_speaker not in prev_offset_spk.keys():
                prev_offset_spk[prev_speaker] = prev_offset + cut.duration
            else:
                prev_offset_spk[prev_speaker] = max(
                    prev_offset + cut.duration + 1e-2, prev_offset_spk[prev_speaker]
                )

            current_time = prev_offset + cut.duration

        base_gain = np.random.uniform(*self.cfg.base_gain)
        speech_lvls = []
        for _ in utterances:
            c_gain = np.random.uniform(*self.cfg.rel_gain)
            speech_lvls.append(base_gain + c_gain)

        # fetch the maximum length
        output_audio = np.zeros((1, int(current_time * self.cfg.samplerate)))

        # we need to check if all cuts have only one supervions TODO
        all_spk = list(set([x.supervisions[0].speaker for x in utterances]))

        # Conditional spk2audio initialization: only create the dicts if save_spk is True
        if self.cfg.save_spk:
            spk2audio = {x: deepcopy(output_audio) for x in all_spk}
            spk2audio_anechoic = {x: deepcopy(output_audio) for x in all_spk}
        else:
            spk2audio = {}
            spk2audio_anechoic = {}

        do_reverb = self.cfg.reverberate and np.random.random() < getattr(
            self.cfg, "reverb_prob", 1.0
        )
        for cut, offset, c_speech_lvl in zip(utterances, offsets, speech_lvls):
            # load audio here using cached loader
            c_audio = cached_load_audio(cut)
            # initial_duration = c_audio.shape[-1]
            # remove dc offset via highpass filtering here.
            # remove anything under 65 Hz to avoid recognizing speaker from artifacts in recording
            assert c_audio.shape[0] == 1  # TODO support multi-channel
            # this reduces overfitting on low freq noise e.g. distinguishing speakers by noise
            c_audio = c_audio - np.mean(c_audio, -1, keepdims=True)
            if self.cfg.use_fir:
                c_audio = convolve(c_audio, fir_highpass[None, :], mode="full")

            if do_reverb:
                # Simple random RIR selection (not position-aware)
                c_spk = cut.supervisions[0].speaker
                c_rir_file = str(np.random.choice(c_room_rirs))

                c_rir, fs = sf.read(c_rir_file)
                assert fs == self.cfg.samplerate
                # load anechoic rir too
                c_rir_anechoic = (
                    str(Path(c_rir_file).parent / Path(c_rir_file).stem)
                    + "-anechoic.flac"
                )
                c_rir_anechoic, fs = sf.read(c_rir_anechoic)
                assert fs == self.cfg.samplerate

                # Find direct sound (peak) and trim RIRs to start there
                # This removes pre-delay and maintains time alignment
                direct_idx = np.argmax(np.abs(c_rir))
                c_rir = c_rir[direct_idx:]

                direct_idx_anechoic = np.argmax(np.abs(c_rir_anechoic))
                c_rir_anechoic = c_rir_anechoic[direct_idx_anechoic:]

                # Now convolve with mode="full" to preserve reverb tail
                c_audio_anechoic = convolve(
                    c_audio, c_rir_anechoic[None, :], mode="full"
                )

                if c_rir.ndim == 1:
                    c_rir = c_rir[:, np.newaxis]
                c_audio = convolve(c_audio, c_rir.T, mode="full")

                # Ensure anechoic matches reverberant length
                if c_audio_anechoic.shape[-1] < c_audio.shape[-1]:
                    c_audio_anechoic = np.pad(
                        c_audio_anechoic,
                        ((0, 0), (0, c_audio.shape[-1] - c_audio_anechoic.shape[-1])),
                        mode="constant",
                    )
                elif c_audio_anechoic.shape[-1] > c_audio.shape[-1]:
                    c_audio_anechoic = c_audio_anechoic[:, : c_audio.shape[-1]]

            else:
                assert self.cfg.save_anechoic == False
            # gain adjust
            c_audio, c_gain = self.normalize_to(c_audio, c_speech_lvl)

            if do_reverb:
                c_audio_anechoic = c_gain * c_audio_anechoic

            offset = int(offset * self.cfg.samplerate)
            maxlen = output_audio.shape[-1]
            if (offset + c_audio.shape[-1]) > maxlen:
                residual = (offset + c_audio.shape[-1]) - maxlen
                output_audio = np.pad(
                    output_audio, ((0, 0), (0, residual)), mode="constant"
                )

                if self.cfg.save_spk:
                    for c_spk in spk2audio.keys():
                        spk2audio[c_spk] = np.pad(
                            spk2audio[c_spk], ((0, 0), (0, residual)), mode="constant"
                        )
                    if self.cfg.save_anechoic:
                        for c_spk in spk2audio_anechoic.keys():
                            spk2audio_anechoic[c_spk] = np.pad(
                                spk2audio_anechoic[c_spk],
                                ((0, 0), (0, residual)),
                                mode="constant",
                            )

            output_audio[:, offset : offset + c_audio.shape[-1]] += c_audio

            if self.cfg.save_spk:
                c_spk = cut.supervisions[0].speaker
                # save only first channel
                spk2audio[c_spk][:, offset : offset + c_audio.shape[-1]] += c_audio[0][
                    None, :
                ]
                if self.cfg.save_anechoic:
                    spk2audio_anechoic[c_spk][
                        :, offset : offset + c_audio_anechoic.shape[-1]
                    ] += c_audio_anechoic

        # add some gaussian noise here.
        # if we have noise we can add that too. e.g. wham and sins, qut etc

        noise_prob = getattr(self.cfg, "noise_probability_global", 1.0)
        if self.cfg.add_noise and np.random.random() < noise_prob:
            min_lvl = min(speech_lvls)
            if self.noise_files is None:
                output_audio = self.add_gaussian_noise(output_audio, min_lvl, (-30, 3))
            else:
                output_audio = self.add_real_noise(
                    output_audio, min_lvl, self.cfg.noise_rel_gain
                )

        maxval = np.amax(np.abs(output_audio))
        gain_f = 0.99 / maxval

        if maxval > 0.99:
            output_audio = output_audio * gain_f
            if self.cfg.save_spk:
                for k in spk2audio.keys():
                    spk2audio[k] = spk2audio[k] * gain_f

                if self.cfg.save_anechoic:
                    for k in spk2audio_anechoic.keys():
                        spk2audio_anechoic[k] = spk2audio_anechoic[k] * gain_f

        # Generate unique filename
        recording_id = str(uuid4())
        audio_filename = f"{recording_id}.wav"
        Path(self.output_dir).mkdir(exist_ok=True)
        audio_path = os.path.join(self.output_dir, audio_filename)

        # Save audio to disk
        sf.write(audio_path, output_audio.T, self.cfg.samplerate)

        if self.cfg.save_spk:
            for k in spk2audio:
                sf.write(
                    os.path.join(self.output_dir, f"{recording_id}-spk-{k}.wav"),
                    spk2audio[k].T,
                    self.cfg.samplerate,
                )
            if self.cfg.save_anechoic:
                for k in spk2audio_anechoic:
                    sf.write(
                        os.path.join(
                            self.output_dir, f"{recording_id}-spk-{k}-anechoic.wav"
                        ),
                        spk2audio_anechoic[k].T,
                        self.cfg.samplerate,
                    )

        # Create Lhotse Recording
        recording = Recording(
            id=recording_id,
            sources=[AudioSource(type="file", channels=[0], source=audio_path)],
            sampling_rate=self.cfg.samplerate,
            num_samples=output_audio.shape[-1],
            duration=output_audio.shape[-1] / self.cfg.samplerate,
        )

        # Create Lhotse Supervisions for each utterance
        supervisions = []
        for i, (cut, offset) in enumerate(zip(utterances, offsets)):
            orig_supervision = cut.supervisions[0]

            if (
                hasattr(orig_supervision, "alignment")
                and "word" in orig_supervision.alignment.keys()
            ):
                # add the offset to the alignments here
                adjusted_alignment = []

                for align_item in orig_supervision.alignment["word"]:
                    # Adjust start and end times by adding the offset
                    adjusted_item = AlignmentItem(
                        symbol=align_item.symbol,
                        start=align_item.start + offset,
                        duration=align_item.duration,
                        score=getattr(align_item, "score", None),
                    )
                    adjusted_alignment.append(adjusted_item)
                alignment = adjusted_alignment
            else:
                alignment = None

            supervision = SupervisionSegment(
                id=f"{recording_id}_{i:04d}",
                recording_id=recording_id,
                start=offset,
                duration=cut.duration,
                channel=0,
                speaker=orig_supervision.speaker,
                text=getattr(orig_supervision, "text", ""),  # Use text if available
                language=getattr(orig_supervision, "language", None),
                alignment=(
                    {"word": alignment} if alignment is not None else None
                ),  # Include adjusted alignments
                custom={
                    "transition_type": (
                        transition_types[i].name
                        if transition_types[i] is not None
                        else "FIRST"
                    ),
                    "speech_level_db": speech_lvls[i],
                    "original_cut_id": cut.id,
                },
            )
            supervisions.append(supervision)

        return recording, supervisions
