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

import io
from dataclasses import dataclass
from typing import cast

import av
import cv2
import numpy.typing as npt
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from torch import Tensor

# We error on any video with a width or height less than this.
# The motion detection algorithm can't handle any resolutions less than this.
_MIN_SIDE_RESOLUTION = 256


class VideoResolutionTooSmallError(Exception):
    """Exception raised when video resolution is below the minimum required size.

    This error occurs when either the width or height of the video is less than
    the minimum resolution threshold required for motion detection.
    """


@dataclass
class MotionInfo:
    """Container for motion detection results.

    This class stores the results of motion detection analysis, including:
    - Whether the video has small motion
    - The minimum motion value in a 256x256 patch
    - The global average motion value across the entire videoq
    """

    is_small_motion: bool
    per_patch_min_256: float
    global_mean: float


@dataclass
class DecodedData:
    """Container for decoded video frames containing motion vector data.

    This class stores a list of decoded frames, each containing motion vector data,
    and the dimensions of the RGB decoded frame used to construct the flow vector.
    """

    # List of decoded frames containing motion vector data, not RGB
    frames: list[npt.NDArray]  # type: ignore[type-arg]
    # pass the dimensions of the RGB decoded frame to construct flow vector
    frame_size: torch.Size

    def get_major_size(self) -> int:
        """Calculate total size in bytes of all frames in the decoded data.

        Returns:
            Total size in bytes.

        """
        total_size = 0
        for frame in self.frames:
            total_size += frame.nbytes
        return total_size


def motion_vectors_to_flowfield(mvs: Tensor, size: list[int], flow: Tensor | None = None) -> Tensor:
    """Compute a canonical flow from motion vectors."""
    # get relevant info for later
    batch_size = mvs.shape[0]
    n_vectors = mvs.shape[1]
    block_sizes = mvs[..., 0:2]
    dst = mvs[..., 4:6]
    motion = mvs[..., 7:9]
    motion_scale = mvs[..., 9].unsqueeze(-1)
    device = mvs.device

    # Make indices for the batch number, this will be used as part of an index later
    batch_indices = torch.arange(batch_size, device=device)
    batch_pos = batch_indices.view(-1, 1).expand(-1, n_vectors).view(batch_size, n_vectors, 1)

    # compute sub-pixel src locations
    # `src = dst + motion / motion_scale`
    delta = -motion / motion_scale

    # add batch position to the source x, y positions
    dst_complete = torch.cat([batch_pos, dst], dim=-1)

    # create a "flow" where each source pixel is annotated with its destination pixel
    if flow is None or flow.shape != (batch_size, *size, 2):
        flow = torch.zeros(batch_size, *size, 2, device=device)
    else:
        flow.zero_()

    # choices for block size
    block_options = [
        torch.as_tensor([8, 8], device=device),
        torch.as_tensor([16, 16], device=device),
        torch.as_tensor([16, 8], device=device),
        torch.as_tensor([8, 16], device=device),
    ]

    # separate motion vectors by block size
    selected_blocks = [(block_sizes == b).all(-1).unsqueeze(-1) for b in block_options]

    # make offsets from center point for each of the block sizes.
    # this is ugly because we need to prepend 0 as a "batch offset" since the points are (batch, x, y)
    offsets = [
        torch.stack(torch.meshgrid(*[torch.arange(-b // 2, b // 2, device=device) for b in bs]))
        for bs in block_options
    ]

    # tile each of the mvs so that they contain the center point repeated for the size of the entire block
    dst_tiles = [
        dst_complete.masked_select(blocks).view(-1, 1, 1, 1).tile(1, 1, *block_size).view(-1, 3, *block_size)
        for blocks, block_size in zip(selected_blocks, block_options, strict=False)
    ]

    # # split each tensor into three indices, batch, h, w
    dst_b, dst_x, dst_y = zip(*[d.split(1, dim=1) for d in dst_tiles], strict=False)  # type: ignore[no-untyped-call]

    # Add in the offsets
    dst_x1 = [(mv_b + offset_b[0]) for mv_b, offset_b in zip(dst_x, offsets, strict=False)]
    dst_y1 = [(mv_b + offset_b[1]) for mv_b, offset_b in zip(dst_y, offsets, strict=False)]

    # check bounds (needed for indexing)
    dst_x2 = [mv_b.where(mv_b > 0, torch.as_tensor(0, device=device)) for mv_b in dst_x1]
    dst_y2 = [mv_b.where(mv_b > 0, torch.as_tensor(0, device=device)) for mv_b in dst_y1]

    dst_x3 = [mv_b.where(mv_b < size[-1], size[-1] - 1) for mv_b in dst_x2]
    dst_y3 = [mv_b.where(mv_b < size[-2], size[-2] - 1) for mv_b in dst_y2]

    # flatten the indices and concat them
    dst_x4 = torch.cat([mv.flatten() for mv in dst_x3], dim=0)
    dst_y4 = torch.cat([mv.flatten() for mv in dst_y3], dim=0)
    dst_b4 = torch.cat([mv.flatten() for mv in dst_b], dim=0)

    # tile and offset the source positions
    delta_tiles = [
        delta.masked_select(blocks).view(-1, 1, 1).tile(1, *block_size).view(-1, 2, *block_size)
        for blocks, block_size in zip(selected_blocks, block_options, strict=False)
    ]
    delta_flat = torch.cat([mv.movedim(1, -1).flatten(0, 2) for mv in delta_tiles], dim=0)

    # index the flow image and set it to the destination
    flow.index_put_((dst_b4.long(), dst_y4.long(), dst_x4.long()), delta_flat, accumulate=False)

    return flow


def decode_for_motion(  # noqa: C901
    video: io.BytesIO,
    thread_count: int = 4,
    target_fps: float = 2.0,
    target_duration_ratio: float = 0.5,
) -> DecodedData:
    """Decode video for motion detection.

    This function decodes a video stream to extract motion vectors.

    Args:
        video: Input video stream as a bytes object.
        thread_count: Number of threads to use for decoding.
        target_fps: Target frames per second for motion detection.
        target_duration_ratio: Ratio of target duration to source duration.

    Returns:
        DecodedData object containing motion vectors and frame dimensions.

    """
    with cast("av.container.InputContainer", av.open(video, metadata_errors="ignore")) as input_container:
        stream = input_container.streams.video[0]
        ctx = stream.codec_context
        # Set this flag to return motion vectors
        ctx.flags2 |= av.codec.context.Flags2.EXPORT_MVS
        ctx.thread_type = av.codec.context.ThreadType.AUTO
        ctx.thread_count = thread_count
        mv_data = []
        shape = torch.Size([1, 1])

        # Get video framerate and duration
        if stream.average_rate:
            source_fps = float(stream.average_rate)
        else:
            # Use nominal base rate if average not available
            source_fps = float(stream.base_rate if stream.base_rate else 30)

        # Get duration in seconds
        duration_seconds = 0
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = stream.duration * int(stream.time_base)
        if duration_seconds <= 0 and hasattr(input_container, "duration") and input_container.duration is not None:
            duration_seconds = int(input_container.duration / int(av.time_base))
        if duration_seconds <= 0:
            # Default assumption if we can't determine duration
            duration_seconds = 30

        # Calculate how many frames to sample based on ratio
        target_seconds = duration_seconds * target_duration_ratio
        max_frames = max(10, round(target_fps * target_seconds))  # At least 10 frames regardless of ratio

        # Calculate frame step to achieve target FPS
        # If source is 30fps and target is 2fps, we sample every 15 frames
        sample_step = max(1, round(source_fps / target_fps))

        for i, frame in enumerate(input_container.decode(video=0)):
            # Only process frames at our target step interval
            if i % sample_step != 0:
                continue

            # Process shape info from first frame
            if i == 0:
                shape = torch.Size([frame.height, frame.width, 3])
                if shape[0] < _MIN_SIDE_RESOLUTION or shape[1] < _MIN_SIDE_RESOLUTION:
                    error_msg = f"Expected min resolution of {_MIN_SIDE_RESOLUTION}, but got a resolution of {shape}"
                    raise VideoResolutionTooSmallError(error_msg)

            # Process motion vectors
            for sd in frame.side_data:
                if sd.type == av.sidedata.sidedata.Type.MOTION_VECTORS:  # type: ignore[attr-defined]
                    mv = structured_to_unstructured(
                        sd.to_ndarray(),  # type: ignore[attr-defined]
                    )
                    mv_data.append(mv)
                    break  # Skip checking other side data once we find motion vectors

            # Early stopping optimization: limit based on calculated max frames
            if len(mv_data) >= max_frames:
                break

    return DecodedData(frames=mv_data, frame_size=shape)


def check_if_small_motion(  # noqa: PLR0913
    mv_list: list[npt.NDArray],  # type: ignore[type-arg]
    frame_shape: torch.Size,
    global_mean_threshold: float = 0.00098,
    per_patch_min_256_threshold: float = 0.000001,
    *,
    use_gpu: bool = False,
    batch_size: int = 256,
) -> MotionInfo:
    """Check if a video has small motion.

    This function checks if a video has small motion by calculating the global mean
    and per-pixel average motion values.

    Args:
        mv_list: List of motion vectors.
        frame_shape: Shape of the frame.
        global_mean_threshold: Threshold for global mean motion.
        per_patch_min_256_threshold: Threshold for per-patch minimum motion.
        use_gpu: Whether to use GPU for computation.
        batch_size: Size of the batch for processing.

    Returns:
        MotionInfo object containing the results of the motion detection.

    """
    device = torch.device("cuda" if use_gpu else "cpu")

    global_sum_tensor = torch.tensor(0.0, device=device)
    per_pixel_sum_tensor = torch.zeros((frame_shape[0], frame_shape[1]), device=device)
    num_frames = 0
    preallocated_flow = torch.zeros(batch_size, frame_shape[0], frame_shape[1], 2, device=device)

    for batch_offset in range(0, len(mv_list), batch_size):
        current_batch_size = min(batch_size, len(mv_list) - batch_offset)

        # Pad all tensors to the same dimensions (the number of blocks varies per frame).
        # Padding with zeros is valid because the blocks of size (0, 0) will not be selected.
        max_n_vectors = max(mv_item.shape[0] for mv_item in mv_list[batch_offset : batch_offset + current_batch_size])
        mv_data_padded = torch.zeros(current_batch_size, max_n_vectors, 10, dtype=torch.float32, device=device)

        for batch_id in range(batch_offset, batch_offset + current_batch_size):
            data = torch.tensor(mv_list[batch_id][:, 1:], dtype=torch.float32, device=device)
            mv_data_padded[batch_id - batch_offset, : data.shape[0], :] = data

        flow_field = motion_vectors_to_flowfield(mv_data_padded, [frame_shape[0], frame_shape[1]], preallocated_flow)
        magnitudes = torch.linalg.vector_norm(flow_field, dim=3) / (frame_shape[1] + frame_shape[0])

        global_sum_tensor += magnitudes.sum()
        per_pixel_sum_tensor += magnitudes.sum(dim=0)
        num_frames += current_batch_size

    total_elements = num_frames * frame_shape[0] * frame_shape[1]
    global_mean = (global_sum_tensor / total_elements).item()
    per_pixel_avg = (per_pixel_sum_tensor / num_frames).cpu().numpy()
    per_patch_min_256 = float(cv2.resize(per_pixel_avg, None, fx=1 / 256, fy=1 / 256).min())

    is_small_motion = global_mean < global_mean_threshold or per_patch_min_256 < per_patch_min_256_threshold

    return MotionInfo(is_small_motion, per_patch_min_256, global_mean)
