import io
import math
from dataclasses import dataclass
from functools import reduce

from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import Video, VideoTask
from ray_curator.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature, extract_frames


@dataclass
class ClipFrameExtractionStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage for extracting frames from video clips.

    This class processes video clips through a series of steps including frame extraction,
    target frame rate selection, and frame extraction signature creation.
    """
    extraction_policies: tuple[FrameExtractionPolicy, ...] = (FrameExtractionPolicy.sequence, )
    target_fps: list[float | int] | None = None
    target_res: tuple[int, int] | None = None
    verbose: bool = False
    num_cpus: int = 3

    @property
    def name(self) -> str:
        return "clip_frame_extraction"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips.extracted_frames"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self.target_fps is None or len(self.target_fps) == 0:
            self.target_fps = [2]
        if self.target_res is None:
            self.target_res = (-1, -1)
        logger.info(f"ClipFrameExtractionStage will extract frames at {self.target_fps} FPS")

    @property
    def resources(self) -> Resources:
        return Resources(cpus=self.num_cpus)

    def lcm_multiple(self, fps: list[float | int]) -> float | int:
        """Compute LCM of a list of fps targets."""

        def lcm(a: float, b: float) -> float | int:
            return abs(a * b) // math.gcd(int(a), int(b))

        return reduce(lcm, fps)

    def process(self, task: VideoTask) -> VideoTask:
        video: Video = task.data
        for clip in video.clips:
            if clip.buffer is None:
                logger.error(f"Clip {clip.uuid} has no buffer")
                clip.errors["buffer"] = "empty"
                continue

            try:
                for policy in self.extraction_policies:
                    """
                    To save on decode costs, calculate the least-common-multiple(LCM) of fps
                    targets and apply decord.get_batch on this LCM fps
                    """
                    use_lcm_fps = len(self.target_fps) > 1 and all(
                        (fps.is_integer() if isinstance(fps, float) else isinstance(fps, int))
                        for fps in self.target_fps
                    )
                    if use_lcm_fps:
                        lcm = self.lcm_multiple(self.target_fps)
                        with io.BytesIO(clip.buffer) as fp:
                            frames = extract_frames(
                                fp,
                                extraction_policy=policy,
                                sample_rate_fps=lcm,
                                target_res=self.target_res,
                                num_threads=self.num_cpus,
                            )
                            for fps in self.target_fps:
                                signature = FrameExtractionSignature(
                                    extraction_policy=policy,
                                    target_fps=fps,
                                ).to_str()
                                clip.extracted_frames[signature] = frames[:: int(lcm / fps)]
                    else:
                        for fps in self.target_fps:
                            with io.BytesIO(clip.buffer) as fp:
                                frames = extract_frames(
                                    fp,
                                    extraction_policy=policy,
                                    sample_rate_fps=fps,
                                    target_res=self.target_res,
                                    num_threads=self.num_cpus,
                                )
                                signature = FrameExtractionSignature(
                                    extraction_policy=policy,
                                    target_fps=fps,
                                ).to_str()
                                clip.extracted_frames[signature] = frames
                                if self.verbose:
                                    logger.info(f"Extracted {len(frames)} frames from clip {clip.uuid} at {fps} fps")
            except (ValueError, OSError, RuntimeError) as e:
                logger.exception(f"Error extracting frames for clip {clip.uuid}: {e}")
                clip.errors["frame_extraction"] = "video_decode_failed"
                # reset the buffer to disable further operations on this clip
                clip.buffer = None
                continue

        if self.verbose:
            logger.info(f"ClipFrameExtractionStage extracted frames for {len(video.clips)} clips")
        return task




