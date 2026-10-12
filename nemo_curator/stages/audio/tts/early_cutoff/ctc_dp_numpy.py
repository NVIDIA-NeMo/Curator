# ruff: noqa
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class NumpyTerminalPhoneLikelihood:
    min_complete_frames: int
    age_bin_log_probs: np.ndarray
    partial_log_probability: float
    complete_log_probability: float
    full_log_probability: float
    partial_posterior: float
    log_likelihood_ratio: float
    consistency_error: float


@dataclass
class NumpyPrefixForwardResult:
    prefix_log_probs: np.ndarray
    final_state_log_probs: np.ndarray
    extended_targets: np.ndarray
    trellis: np.ndarray | None = None
    terminal_phone: NumpyTerminalPhoneLikelihood | None = None


@dataclass
class NumpyPhoneAlignment:
    phone_index: int
    token_id: int
    start_frame: int | None
    end_frame: int | None
    duration_frames: int
    posterior_mean_frame: float | None
    posterior_std_frames: float | None
    acoustic_margin: float | None
    occupancy: float


@dataclass
class NumpyAlignmentResult:
    log_probability: float
    state_path: np.ndarray
    state_posteriors: np.ndarray
    phones: list[NumpyPhoneAlignment]


def minimum_ctc_frames_numpy(targets: np.ndarray) -> int:
    targets = np.asarray(targets, dtype=np.int64)
    return int(targets.size + np.count_nonzero(targets[1:] == targets[:-1]))


def _states(targets: np.ndarray, blank_id: int) -> np.ndarray:
    states = np.full(2 * targets.size + 1, blank_id, dtype=np.int64)
    states[1::2] = targets
    return states


def _validate(log_probs: np.ndarray, targets: np.ndarray, blank_id: int) -> None:
    if log_probs.ndim != 2 or log_probs.shape[0] == 0:
        raise ValueError("expected non-empty log_probs [T,V]")
    if targets.ndim != 1:
        raise ValueError("expected targets [N]")
    if not 0 <= blank_id < log_probs.shape[1]:
        raise ValueError("blank_id is outside vocabulary")
    if np.isnan(log_probs).any() or np.isposinf(log_probs).any():
        raise ValueError("log_probs contain NaN/+inf")
    if targets.size and (targets.min() < 0 or targets.max() >= log_probs.shape[1]):
        raise ValueError("target token is outside vocabulary")
    if np.any(targets == blank_id):
        raise ValueError("target contains CTC blank")


def _skip_mask(states: np.ndarray, blank_id: int) -> np.ndarray:
    mask = np.zeros(states.size, dtype=bool)
    mask[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    return mask


def ctc_prefix_forward_numpy(
    log_probs: np.ndarray,
    targets: np.ndarray,
    *,
    blank_id: int = 0,
    return_trellis: bool = False,
    terminal_phone_min_complete_frames: int | None = None,
) -> NumpyPrefixForwardResult:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    _validate(log_probs, targets, blank_id)
    if terminal_phone_min_complete_frames is not None and (
        isinstance(terminal_phone_min_complete_frames, bool)
        or not isinstance(terminal_phone_min_complete_frames, int)
        or terminal_phone_min_complete_frames < 2
    ):
        raise ValueError("terminal_phone_min_complete_frames must be an integer >= 2")
    states = _states(targets, blank_id)
    emissions = log_probs[:, states]
    skip_mask = _skip_mask(states, blank_id)
    previous = np.full(states.size, -np.inf, dtype=np.float64)
    previous[0] = 0.0
    trellis = np.empty((log_probs.shape[0], states.size), dtype=np.float64) if return_trellis else None
    track_terminal = terminal_phone_min_complete_frames is not None and targets.size > 0
    terminal_age = (
        np.full(
            int(terminal_phone_min_complete_frames),
            -np.inf,
            dtype=np.float64,
        )
        if track_terminal
        else None
    )
    final_phone_state = states.size - 2
    for time_idx, frame in enumerate(emissions):
        old = previous
        step = np.empty_like(previous)
        step[0] = -np.inf
        step[1:] = old[:-1]
        skip = np.full_like(previous, -np.inf)
        skip[2:] = old[:-2]
        skip[~skip_mask] = -np.inf
        current = np.logaddexp(np.logaddexp(old, step), skip) + frame
        if terminal_age is not None:
            predecessor = old[final_phone_state - 1]
            if skip_mask[final_phone_state]:
                predecessor = np.logaddexp(
                    predecessor,
                    old[final_phone_state - 2],
                )
            new_age = np.full_like(terminal_age, -np.inf)
            new_age[0] = predecessor + frame[final_phone_state]
            if terminal_age.size > 2:
                new_age[1:-1] = terminal_age[:-2] + frame[final_phone_state]
            new_age[-1] = np.logaddexp(terminal_age[-2], terminal_age[-1]) + frame[final_phone_state]
            terminal_age = new_age
        if trellis is not None:
            trellis[time_idx] = current
        previous = current

    prefix = np.empty(targets.size + 1, dtype=np.float64)
    prefix[0] = previous[0]
    if targets.size:
        prefix[1:] = np.logaddexp(previous[1::2], previous[2::2])
    terminal_phone = None
    if terminal_age is not None:
        partial_log_probability = float(np.logaddexp.reduce(terminal_age[:-1]))
        complete_log_probability = float(np.logaddexp(previous[-1], terminal_age[-1]))
        full_log_probability = float(np.logaddexp(partial_log_probability, complete_log_probability))
        reconstructed_phone = float(np.logaddexp.reduce(terminal_age))
        phone_state_log_probability = float(previous[final_phone_state])
        consistency_error = (
            reconstructed_phone - phone_state_log_probability
            if math.isfinite(reconstructed_phone) and math.isfinite(phone_state_log_probability)
            else 0.0
        )
        partial_posterior = (
            float(np.exp(partial_log_probability - full_log_probability))
            if math.isfinite(full_log_probability) and math.isfinite(partial_log_probability)
            else 0.0
        )
        log_likelihood_ratio = (
            partial_log_probability - complete_log_probability
            if math.isfinite(partial_log_probability) and math.isfinite(complete_log_probability)
            else 30.0
            if math.isfinite(partial_log_probability)
            else -30.0
            if math.isfinite(complete_log_probability)
            else 0.0
        )
        terminal_phone = NumpyTerminalPhoneLikelihood(
            min_complete_frames=int(terminal_phone_min_complete_frames),
            age_bin_log_probs=terminal_age,
            partial_log_probability=partial_log_probability,
            complete_log_probability=complete_log_probability,
            full_log_probability=full_log_probability,
            partial_posterior=partial_posterior,
            log_likelihood_ratio=log_likelihood_ratio,
            consistency_error=consistency_error,
        )
    return NumpyPrefixForwardResult(
        prefix,
        previous,
        states,
        trellis,
        terminal_phone,
    )


def ctc_viterbi_path_numpy(
    log_probs: np.ndarray,
    targets: np.ndarray,
    *,
    blank_id: int = 0,
) -> tuple[float, np.ndarray]:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    _validate(log_probs, targets, blank_id)
    states = _states(targets, blank_id)
    emissions = log_probs[:, states]
    skip_mask = _skip_mask(states, blank_id)
    previous = np.full(states.size, -np.inf, dtype=np.float64)
    previous[0] = 0.0
    backpointers = np.zeros((log_probs.shape[0], states.size), dtype=np.int8)
    for time_idx, frame in enumerate(emissions):
        step = np.empty_like(previous)
        step[0] = -np.inf
        step[1:] = previous[:-1]
        skip = np.full_like(previous, -np.inf)
        skip[2:] = previous[:-2]
        skip[~skip_mask] = -np.inf
        candidates = np.stack((previous, step, skip))
        offsets = np.argmax(candidates, axis=0)
        previous = np.take_along_axis(candidates, offsets[None, :], axis=0)[0] + frame
        backpointers[time_idx] = offsets

    final_states = np.array([0] if not targets.size else [states.size - 2, states.size - 1])
    final_values = previous[final_states]
    best = int(np.argmax(final_values))
    state = int(final_states[best])
    score = float(final_values[best])
    if not math.isfinite(score):
        raise ValueError("CTC target has no feasible Viterbi path")
    path = np.empty(log_probs.shape[0], dtype=np.int64)
    for time_idx in range(log_probs.shape[0] - 1, -1, -1):
        path[time_idx] = state
        state -= int(backpointers[time_idx, state])
    return score, path


def ctc_forward_backward_numpy(
    log_probs: np.ndarray,
    targets: np.ndarray,
    *,
    blank_id: int = 0,
) -> tuple[float, np.ndarray, np.ndarray]:
    forward = ctc_prefix_forward_numpy(
        log_probs,
        targets,
        blank_id=blank_id,
        return_trellis=True,
    )
    assert forward.trellis is not None
    alpha = forward.trellis
    states = forward.extended_targets
    log_z = float(forward.prefix_log_probs[-1])
    beta = np.full_like(alpha, -np.inf)
    beta[-1, 0] = 0.0 if targets.size == 0 else -np.inf
    if targets.size:
        beta[-1, -2:] = 0.0
    skip_destination = _skip_mask(states, blank_id)
    for time_idx in range(log_probs.shape[0] - 2, -1, -1):
        destination = log_probs[time_idx + 1, states] + beta[time_idx + 1]
        step = np.empty_like(destination)
        step[:-1] = destination[1:]
        step[-1] = -np.inf
        skip_values = np.where(skip_destination, destination, -np.inf)
        skip = np.full_like(destination, -np.inf)
        skip[:-2] = skip_values[2:]
        beta[time_idx] = np.logaddexp(np.logaddexp(destination, step), skip)
    gamma = np.exp(alpha + beta - log_z) if math.isfinite(log_z) else np.zeros_like(alpha)
    return log_z, gamma, states


def align_phones_numpy(
    log_probs: np.ndarray,
    targets: np.ndarray,
    *,
    blank_id: int = 0,
) -> NumpyAlignmentResult:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    log_z, gamma, _ = ctc_forward_backward_numpy(
        log_probs,
        targets,
        blank_id=blank_id,
    )
    _, path = ctc_viterbi_path_numpy(log_probs, targets, blank_id=blank_id)
    if targets.size == 0:
        return NumpyAlignmentResult(log_z, path, gamma, [])

    top_two_indices = np.argpartition(log_probs, -2, axis=1)[:, -2:]
    top_two_values = np.take_along_axis(log_probs, top_two_indices, axis=1)
    order = np.argsort(top_two_values, axis=1)
    top_ids = np.take_along_axis(top_two_indices, order, axis=1)
    top_values = np.take_along_axis(top_two_values, order, axis=1)
    frame_numbers = np.arange(log_probs.shape[0], dtype=np.float64)
    phone_gamma = gamma[:, 1::2]
    occupancies = phone_gamma.sum(axis=0)
    means = np.divide(
        (phone_gamma * frame_numbers[:, None]).sum(axis=0),
        occupancies,
        out=np.zeros_like(occupancies),
        where=occupancies > 1e-8,
    )
    variances = np.divide(
        (phone_gamma * np.square(frame_numbers[:, None] - means[None, :])).sum(axis=0),
        occupancies,
        out=np.zeros_like(occupancies),
        where=occupancies > 1e-8,
    )
    token_scores = log_probs[:, targets]
    competitors = np.where(
        top_ids[:, 1, None] == targets[None, :],
        top_values[:, 0, None],
        top_values[:, 1, None],
    )
    margins = np.divide(
        (phone_gamma * (token_scores - competitors)).sum(axis=0),
        occupancies,
        out=np.zeros_like(occupancies),
        where=occupancies > 1e-8,
    )
    phones = []
    for phone_idx, token in enumerate(targets.tolist()):
        state = 2 * phone_idx + 1
        occupancy = float(occupancies[phone_idx])
        path_frames = np.flatnonzero(path == state)
        start = int(path_frames[0]) if path_frames.size else None
        end = int(path_frames[-1]) if path_frames.size else None
        mean = std = margin = None
        if occupancy > 1e-8:
            mean = float(means[phone_idx])
            std = math.sqrt(max(float(variances[phone_idx]), 0.0))
            margin = float(margins[phone_idx])
        phones.append(
            NumpyPhoneAlignment(
                phone_index=phone_idx,
                token_id=token,
                start_frame=start,
                end_frame=end,
                duration_frames=int(path_frames.size),
                posterior_mean_frame=mean,
                posterior_std_frames=std,
                acoustic_margin=margin,
                occupancy=occupancy,
            )
        )
    return NumpyAlignmentResult(log_z, path, gamma, phones)
