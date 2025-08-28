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

import enum
import queue
from pathlib import Path
from typing import Any

import torch
from loguru import logger

try:
    import cvcuda  # type: ignore[import-untyped]
    import nvcv  # type: ignore[import-untyped]
    import pycuda.driver as cuda  # type: ignore[import-untyped]
    import PyNvVideoCodec as Nvc  # type: ignore[import-untyped]

    pixel_format_to_cvcuda_code = {
        Nvc.Pixel_Format.YUV444: cvcuda.ColorConversion.YUV2RGB,  # type: ignore[import-untyped]
        Nvc.Pixel_Format.NV12: cvcuda.ColorConversion.YUV2RGB_NV12,  # type: ignore[import-untyped]
    }
except (ImportError, RuntimeError):
    logger.warning("PyNvVideoCodec is not installed, some features will be disabled.")
    Nvc = None
    cvcuda = None
    nvcv = None
    cuda = None
    pixel_format_to_cvcuda_code = {}


class FrameExtractionPolicy(enum.Enum):
    """Policy for extracting frames from video, supporting full extraction or FPS-based sampling.

    This enum defines the available strategies for frame extraction from video content.
    """

    full = 0  # Decode video and return all frames
    fps = 1  # Decode video and return frames at a certain sampling fps


class VideoBatchDecoder:
    """GPU-accelerated video decoder that processes video frames in batches.

    This class handles video decoding using NVIDIA hardware acceleration, supporting
    batch processing of frames with color space conversion and resizing capabilities.
    """

    def __init__(  # noqa: PLR0913
        self,
        batch_size: int,
        target_width: int,
        target_height: int,
        device_id: int,
        cuda_ctx: str,  # pyright: ignore[reportAttributeAccessIssue]
        cvcuda_stream: str,  # pyright: ignore[reportAttributeAccessIssue]
    ) -> None:
        """Initialize video batch decoder with GPU acceleration parameters.

        Args:
            batch_size: Number of frames to process in each batch.
            target_width: Target width for decoded frames.
            target_height: Target height for decoded frames.
            device_id: GPU device ID to use.
            cuda_ctx: CUDA context for GPU operations.
            cvcuda_stream: CUDA stream for parallel processing.

        """
        if Nvc is None:
            msg = "PyNvVideoCodec is not available, please install it."
            raise RuntimeError(msg)

        self.batch_size = batch_size
        if self.batch_size <= 0:
            msg = "Batch size should be a valid number."
            raise ValueError(msg)

        self.target_height = target_height
        self.target_width = target_width
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.cvcuda_stream = cvcuda_stream
        self.decoder: NvVideoDecoder | None = None
        self.torch_YUVtensor = None
        self.cvcuda_RGBtensor = None
        self.nvDemux = None  # pyright: ignore[reportAttributeAccessIssue]
        self.fps = None
        self.prev_batch_size = self.batch_size
        self.input_path: str | None = None

    def get_fps(self) -> int | None:
        """Get the frame rate of the video.

        Returns:
            Frame rate of the video.

        """
        return self.fps

    def __call__(
        self,
        input_path: str,
    ) -> torch.Tensor | None:
        """Process video frames in batches using GPU acceleration.

        Args:
            input_path: Path to the video file to process.

        Returns:
            Processed video frames as a tensor.

        """
        # Check if we need to allocate the decoder for its first use.
        if self.input_path != input_path:
            self.input_path = input_path
            self.decoder = NvVideoDecoder(
                self.input_path,
                self.device_id,
                self.batch_size,
                self.cuda_ctx,
                self.cvcuda_stream,
            )
            self.cvcuda_RGBtensor = None
            self.fps = self.decoder.nvDemux.FrameRate()

        if self.decoder is None:
            msg = "Decoder is not initialized"
            raise RuntimeError(msg)

        # Calculate the target width and height if any one of them is -1
        if self.target_width == -1 or self.target_height == -1:
            width, height = self.decoder.w, self.decoder.h
            minimum_width = 256
            downscale_factor = 1 if width < minimum_width else width // minimum_width

            self.target_width = round(width / downscale_factor)
            self.target_height = round(height / downscale_factor)

        # Get the NHWC YUV tensor from the decoder
        self.torch_YUVtensor = self.decoder.get_next_frames()
        # Check if we are done decoding
        if self.torch_YUVtensor is None:
            return None

        cvcuda_yuv_tensor = cvcuda.as_tensor(self.torch_YUVtensor.cuda(self.device_id), "NHWC")

        # Check the code for the color conversion based in the pixel format
        cvcuda_code = pixel_format_to_cvcuda_code.get(self.decoder.pixelFormat)
        if cvcuda_code is None:
            error_msg = f"Unsupported pixel format: {self.decoder.pixelFormat}"
            raise ValueError(error_msg)

        # Check layout to make sure it is what we expected
        if cvcuda_yuv_tensor.layout != "NHWC":
            error_msg = "Unexpected tensor layout, NHWC expected."
            raise ValueError(error_msg)

        # this may be different than batch size since last frames may not be a multiple of batch size
        actual_batch_size = cvcuda_yuv_tensor.shape[0]

        # Create a CVCUDA tensor for color conversion YUV->RGB
        # Allocate only for the first time or for the last batch.

        if (
            not self.cvcuda_RGBtensor
            or actual_batch_size != self.batch_size
            or actual_batch_size != self.prev_batch_size
        ):
            self.torch_RGBtensor = torch.empty(
                (actual_batch_size, self.decoder.h, self.decoder.w, 3),
                dtype=torch.uint8,
                device=f"cuda:{self.device_id}",
            )
            self.cvcuda_RGBtensor = cvcuda.as_tensor(
                self.torch_RGBtensor.cuda(self.device_id),
                "NHWC",
            )

        # Convert from YUV to RGB. Conversion code is based on the pixel format.
        cvcuda.cvtcolor_into(self.cvcuda_RGBtensor, cvcuda_yuv_tensor, cvcuda_code, stream=self.cvcuda_stream)

        torch_rgb_tensor_resized = torch.empty(
            (self.cvcuda_RGBtensor.shape[0], self.target_height, self.target_width, self.cvcuda_RGBtensor.shape[3]),
            dtype=torch.uint8,
            device=f"cuda:{self.device_id}",
        )
        cvcuda_rgb_tensor_resized = cvcuda.as_tensor(
            torch_rgb_tensor_resized.cuda(self.device_id),
            "NHWC",
        )
        cvcuda.resize_into(
            cvcuda_rgb_tensor_resized,
            self.cvcuda_RGBtensor,
            cvcuda.Interp.LINEAR,
            stream=self.cvcuda_stream,
        )
        self.prev_batch_size = actual_batch_size
        return torch_rgb_tensor_resized


class NvVideoDecoder:
    """Low-level NVIDIA hardware-accelerated video decoder.

    This class provides direct access to NVIDIA's hardware video decoding capabilities,
    handling frame decoding and memory management for video processing pipelines.
    """

    def __init__(
        self,
        enc_file: str,
        device_id: int,
        batch_size: int,
        cuda_ctx: Any,  # noqa: ANN401
        cvcuda_stream: Any,  # noqa: ANN401
    ) -> None:
        """Create instance of HW-accelerated video decoder.

        :param enc_file: Full path to the MP4 file that needs to be decoded.
        :param device_id: id of video card which will be used for decoding & processing.
        :param cuda_ctx: A cuda context object.
        """
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.input_path = enc_file
        self.cvcuda_stream = cvcuda_stream
        # Demuxer is instantiated only to collect required information about
        # certain video file properties.
        self.nvDemux = Nvc.PyNvDemuxer(self.input_path)  # pyright: ignore[reportAttributeAccessIssue]
        self.nvDec = Nvc.CreateDecoder(  # pyright: ignore[reportAttributeAccessIssue]
            gpuid=self.device_id,
            codec=self.nvDemux.GetNvCodecId(),
            cudacontext=self.cuda_ctx.handle,  # pyright: ignore[reportAttributeAccessIssue]
            cudastream=self.cvcuda_stream.handle,  # pyright: ignore[reportAttributeAccessIssue]
            enableasyncallocations=False,
            usedevicememory=1,
        )
        logger.info("after initializing nvDec")
        self.w, self.h = self.nvDemux.Width(), self.nvDemux.Height()
        self.pixelFormat = self.nvDec.GetPixelFormat()

        self.batch_size = batch_size
        self.decoded_frame_size = None
        self.input_frame_list = queue.Queue()  # type: ignore[X]
        self.nv12_frame_size = None
        self.seq_triggered = False
        self.decoded_frame_cnt = 0
        self.local_frame_index = 0
        self.sent_frame_cnt = 0

    # frame iterator
    def generate_decoded_frames(self) -> list[torch.Tensor]:
        """Generate decoded frames from the video.

        Returns:
            List of decoded frames as tensors.

        """
        for packet in self.nvDemux:
            list_frames = self.nvDec.Decode(packet)
            for decoded_frame in list_frames:
                nvcv_tensor = nvcv.as_tensor(nvcv.as_image(decoded_frame.nvcv_image(), nvcv.Format.U8))
                if nvcv_tensor.layout == "NCHW":
                    nchw_shape = nvcv_tensor.shape
                    nhwc_shape = (nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1])
                    torch_nhwc = torch.empty(
                        nhwc_shape,
                        dtype=torch.uint8,
                        device=f"cuda:{self.device_id}",
                    )
                    cvcuda_nhwc = cvcuda.as_tensor(torch_nhwc.cuda(self.device_id), "NHWC")
                    cvcuda.reformat_into(cvcuda_nhwc, nvcv_tensor, stream=self.cvcuda_stream)
                    # Push the decoded frame with the reformatted frame to keep it alive.
                    self.input_frame_list.put(torch_nhwc)
                else:
                    error_msg = "Unexpected tensor layout, NCHW expected."
                    raise ValueError(error_msg)

                self.local_frame_index = self.local_frame_index + 1
                self.decoded_frame_cnt = self.decoded_frame_cnt + 1

            if self.local_frame_index > 0 and self.local_frame_index >= self.batch_size:
                self.sent_frame_cnt = self.sent_frame_cnt + self.local_frame_index
                # `print(f"Frames sent: {self.sent_frame_cnt}, {self.local_frame_index}")`
                sent_frame_list = []
                for _i in range(self.batch_size):
                    sent_frame_list.append(self.input_frame_list.get())
                    self.local_frame_index = self.local_frame_index - 1
                return sent_frame_list

        if self.local_frame_index > 0:
            self.sent_frame_cnt = self.sent_frame_cnt + self.local_frame_index
            # `print(f"Frames sent: {self.sent_frame_cnt}, {self.local_frame_index}")`
            sent_frame_list = []
            for _i in range(self.local_frame_index):
                sent_frame_list.append(self.input_frame_list.get())
                self.local_frame_index = self.local_frame_index - 1
            return sent_frame_list

        return []

    def get_next_frames(self) -> torch.Tensor | None:
        """Get the next frames from the video.

        Returns:
            Next frames from the video as a tensor.

        """
        decoded_frames = self.generate_decoded_frames()
        if len(decoded_frames) == 0:
            return None
        if len(decoded_frames) == 1:  # this case we dont need stack the tensor
            return decoded_frames[0]
        # convert from list of tensors to a single tensor (NHWC)
        # tensorNHWC_torch
        # Sync stream before the decoded frames go out of scope
        return torch.cat(decoded_frames)


def gpu_decode_for_stitching(  # noqa: PLR0913
    device_id: int,
    ctx: str,
    stream: int,
    input_path: Path,
    frame_list: list[int],
    batch_size: int = 1,
) -> list[torch.Tensor]:
    """Decode video frames for stitching using GPU acceleration.

    Args:
        device_id: GPU device ID.
        ctx: CUDA context.
        stream: CUDA stream.
        input_path: Path to the video file to process.
        frame_list: List of frame indices to decode.
        batch_size: Number of frames to process in each batch.

    Returns:
        List of decoded frames as tensors.

    """
    cuda_ctx = ctx
    cvcuda_stream = cvcuda.cuda.as_stream(stream)
    torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
    frames = []

    decoder = VideoBatchDecoder(
        batch_size,
        224,
        224,
        device_id,
        cuda_ctx,
        cvcuda_stream,
    )

    # Loop through all input frames
    frame_idx = 0
    while True:
        # Make sure that cvcuda and torch are using the same stream
        with torch.cuda.stream(torch_stream):  # pyright: ignore[reportArgumentType]
            batch = decoder(input_path.as_posix())
            if batch is None:
                break  # No more frames to decode

            actual_batch_size = batch.shape[0]
            batch = torch.as_tensor(batch.cuda(), device=f"cuda:{device_id}")

            for i in range(frame_idx, frame_idx + actual_batch_size):
                if i in frame_list:
                    # There can be cases where the clip is so small that boundary frames fall outside
                    # or end up being the same. In such case, simply checking if "i in frame_list" won't work.
                    # I am not sure if we want to discard such clips or keep them or what to do about the boundary.
                    # So, this is a pure hack to make the pipeline work.

                    for _j in range(frame_list.count(i)):
                        # use list.extend
                        frames.append(batch[i - frame_idx, :, :, :])

            frame_idx += actual_batch_size

    # No sync required here because it's in the same thread
    return frames


class PyNvcFrameExtractor:
    """High-level frame extraction interface using PyNvVideoCodec.

    This class provides a simplified interface for extracting frames from videos using
    hardware acceleration, supporting both full extraction and FPS-based sampling.
    """

    def __init__(
        self,
        width: int,
        height: int,
        batch_size: int,
    ) -> None:
        """Initialize the PyNvcFrameExtractor.

        Args:
            width: Width of the video frames.
            height: Height of the video frames.
            batch_size: Number of frames to process in each batch.

        """
        device_id = 0
        cuda_device = cuda.Device(device_id)  # type: ignore[X]
        cuda_ctx = cuda_device.retain_primary_context()
        cuda_ctx.push()
        cvcuda_stream = cvcuda.Stream()
        # `cvcuda_stream = cvcuda.cuda.as_stream(cvcuda_stream)`
        self.torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
        cuda_ctx.pop()

        self.decoder = VideoBatchDecoder(
            batch_size,
            width,
            height,
            device_id,
            cuda_ctx,
            cvcuda_stream,
        )

    def __call__(
        self,
        input_path: Path,
        extraction_policy: FrameExtractionPolicy = FrameExtractionPolicy.full,
        sampling_fps: int = 1,
    ) -> torch.Tensor:
        """Extract frames from the video.

        Args:
            input_path: Path to the video file to process.
            extraction_policy: Policy for extracting frames.
            sampling_fps: Sampling rate for FPS-based extraction.

        Returns:
            List of decoded frames as tensors.

        """
        tensor_list = []
        video_fps: int | None = None
        # Loop through all input frames
        while True:
            # Make sure that cvcuda and torch are using the same stream
            with torch.cuda.stream(self.torch_stream):  # pyright: ignore[reportArgumentType]
                batch = self.decoder(input_path.as_posix())
                if batch is None:
                    break  # No more frames to decode
                if extraction_policy == FrameExtractionPolicy.fps:
                    if video_fps is None:
                        video_fps = self.decoder.get_fps()
                        if video_fps is None:
                            msg = "Unable to get video FPS"
                            raise RuntimeError(msg)
                    batch = batch[:: int(video_fps / sampling_fps)]
                tensor_list.append(batch)
        return torch.cat(tensor_list, dim=0)
