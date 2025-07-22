# ruff: noqa: ANN401, C901

import io
import pathlib
import pickle
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import Clip, ClipStats, Video, VideoTask
from ray_curator.utils.storage_utils import get_full_path
from ray_curator.utils.writer_utils import write_bytes, write_json, write_parquet


@dataclass
class WriteSpec:
    """Specification for writing data - replaces all hardcoded methods."""
    path: str                    # Path template: "clips/{uuid}.mp4"
    field: str                   # Source field: "buffer"
    format: str                  # Format: "mp4", "json", "pickle", "webp"
    condition: Callable = None   # Optional: lambda obj: obj.buffer is not None
    transform: str = None        # Optional: "to_dict", "to_list"


def extract_clip_metadata(clip: Clip, video_metadata: Any, caption_models: list[str], enhanced_caption_models: list[str]) -> dict[str, Any]:
    """Extract metadata from clip - consolidated from original."""
    data = {
        "span_uuid": str(clip.uuid),
        "source_video": str(clip.source_video),
        "duration_span": list(clip.span),
        "width_source": video_metadata.width,
        "height_source": video_metadata.height,
        "framerate_source": video_metadata.framerate,
    }

    # Add clip metadata
    if clip_metadata := clip.extract_metadata():
        data.update(clip_metadata)

    # Add scores
    if clip.motion_score_global_mean is not None:
        data["motion_score"] = {
            "global_mean": clip.motion_score_global_mean,
            "per_patch_min_256": clip.motion_score_per_patch_min_256,
        }
    if clip.aesthetic_score is not None:
        data["aesthetic_score"] = clip.aesthetic_score
    if len(clip.errors) > 0:
        data["errors"] = list(clip.errors)

    # Add windows
    data["windows"] = []
    for window in clip.windows:
        curr_window = {"start_frame": window.start_frame, "end_frame": window.end_frame}

        # Captions
        for model in caption_models:
            if model in window.caption:
                curr_window[f"{model}_caption"] = window.caption[model]
        for model in enhanced_caption_models:
            if model in window.enhanced_caption:
                curr_window[f"{model}_enhanced_caption"] = window.enhanced_caption[model]

        data["windows"].append(curr_window)

    data["valid"] = bool(clip.buffer and len(clip.windows) > 0)
    return data


@dataclass
class GenericClipWriterStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that writes clips and metadata for clip transcoding.

    This class processes video clips through a series of steps including embedding generation,
    metadata extraction, and writing to storage.
    """

    output_path: str
    input_path: str
    upload_clips: bool
    dry_run: bool
    generate_embeddings: bool
    generate_previews: bool
    generate_captions: bool
    embedding_algorithm: str = "cosmos-embed1"
    caption_models: list[str] = None
    enhanced_caption_models: list[str] = None
    verbose: bool = False
    max_workers: int = 6
    _name: str = "clip_writer"

    def __post_init__(self):
        if self.caption_models is None:
            self.caption_models = []
        if self.enhanced_caption_models is None:
            self.enhanced_caption_models = []

        self.write_specs = [
            # Clips
            WriteSpec("clips/{uuid}.mp4", "buffer", "mp4", lambda c: c.buffer is not None),
            WriteSpec("metas/v0/{uuid}.json", "self", "json", transform="metadata"),
            WriteSpec("iv2_embd/{uuid}.pickle", "intern_video_2_embedding", "pickle", lambda c: c.intern_video_2_embedding is not None),
            WriteSpec("ce1_embd/{uuid}.pickle", "cosmos_embed1_embedding", "pickle", lambda c: c.cosmos_embed1_embedding is not None),
            # Filtered clips
            WriteSpec("filtered_clips/{uuid}.mp4", "buffer", "mp4", lambda c: hasattr(c, "_filtered") and c.buffer is not None),
            WriteSpec("metas/v0/{uuid}.json", "self", "json", lambda c: hasattr(c, "_filtered"), transform="metadata"),
            # Windows
            WriteSpec("previews/{clip_uuid}/{window}.webp", "webp_bytes", "webp", lambda w: w.webp_bytes is not None),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @property
    def resources(self) -> Resources:
        return Resources(cpus=0.25)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self.storage_client = None
        self._iv2_buffer: list[dict] = []
        self._ce1_buffer: list[dict] = []

    def process(self, task: VideoTask) -> VideoTask:
        """Process video using schema-driven approach."""
        video = task.data

        # Mark filtered clips
        for clip in video.filtered_clips:
            clip._filtered = True

        # Collect all objects to write
        all_objects = video.clips + video.filtered_clips

        # Add windows with context
        for clip in video.clips:
            for window in clip.windows:
                window.clip_uuid = clip.uuid
                window.window = f"{window.start_frame}_{window.end_frame}"
                all_objects.append(window)

        # Write everything in parallel using specs
        video.clip_stats = self._write_objects_parallel(all_objects, video)

        # Write aggregated embeddings
        self._write_embeddings_parquet(video)

        # Write video metadata (unchanged)
        self._write_video_metadata(video)

        # Cleanup
        self._cleanup_video_data(video)

        if self.verbose:
            logger.info(f"Video {video.input_path}: {len(video.clips)} clips → {self.output_path}")

        return task

    def _write_objects_parallel(self, objects: list[Any], video: Video) -> ClipStats:
        """Write all objects in parallel using write specs."""
        stats = ClipStats()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for obj in objects:
                for spec in self.write_specs:
                    if self._should_write(obj, spec):
                        future = executor.submit(self._write_single, obj, spec, video)
                        futures.append((future, spec, obj))

            # Collect results
            for future, spec, obj in futures:
                try:
                    if future.result():
                        self._update_stats(stats, spec, obj)
                except (OSError, ValueError, RuntimeError) as e:  # noqa: PERF203
                    logger.error(f"Write error {spec.path}: {e}")

        return stats

    def _should_write(self, obj: Any, spec: WriteSpec) -> bool:
        """Check if object should be written per spec."""
        # Check field exists
        if spec.field != "self" and not hasattr(obj, spec.field):
            return False
        if spec.field != "self" and getattr(obj, spec.field) is None:
            return False

        # Check condition
        return not spec.condition or spec.condition(obj)

    def _write_single(self, obj: Any, spec: WriteSpec, video: Video) -> bool: # noqa: PLR0912
        """Write single object according to spec."""
        try:
            # Get data
            data = obj if spec.field == "self" else getattr(obj, spec.field)

            # Transform if needed
            if spec.transform == "metadata":
                data = extract_clip_metadata(obj, video.metadata, self.caption_models, self.enhanced_caption_models)
            elif spec.transform == "to_list":
                data = data.reshape(-1).tolist()

            # Generate path
            path_vars = {}
            if hasattr(obj, "uuid"):
                path_vars["uuid"] = str(obj.uuid)
            if hasattr(obj, "clip_uuid"):
                path_vars["clip_uuid"] = str(obj.clip_uuid)
            if hasattr(obj, "window"):
                path_vars["window"] = obj.window

            output_path = get_full_path(self.output_path, spec.path.format(**path_vars))

            if self.dry_run:
                return True

            # Write by format
            if spec.format in ["mp4", "webp"]:
                write_bytes(data, output_path, f"{spec.format} data", str(output_path), verbose=self.verbose, client=self.storage_client)
            elif spec.format == "json":
                write_json(data, output_path, "json data", str(output_path), verbose=self.verbose, client=self.storage_client)
            elif spec.format == "pickle":
                buffer = io.BytesIO()
                pickle.dump(data, buffer)
                write_bytes(buffer.getvalue(), output_path, "pickle data", str(output_path), verbose=self.verbose, client=self.storage_client)

            # Add to embedding buffers for parquet
            if spec.field == "intern_video_2_embedding" and data is not None:
                self._iv2_buffer.append({"id": str(obj.uuid), "embedding": data.reshape(-1).tolist()})
            elif spec.field == "cosmos_embed1_embedding" and data is not None:
                self._ce1_buffer.append({"id": str(obj.uuid), "embedding": data.reshape(-1).tolist()})

        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Write error: {e}")
            return False
        else:
            return True

    def _update_stats(self, stats: ClipStats, spec: WriteSpec, obj: Any) -> None:
        """Update stats based on what was written."""
        if spec.format == "mp4":
            stats.num_transcoded += 1
            if not hasattr(obj, "_filtered"):
                stats.num_passed += 1
        elif spec.field in ["intern_video_2_embedding", "cosmos_embed1_embedding"]:
            stats.num_with_embeddings += 1
        elif spec.format == "webp":
            stats.num_with_webp += 1
        elif spec.transform == "metadata" and hasattr(obj, "windows"):
            # Check captions
            has_caption = any(any(w.caption.values()) for w in obj.windows if hasattr(w, "caption"))
            if has_caption:
                stats.num_with_caption += 1
            # Duration
            if hasattr(obj, "span"):
                duration = obj.span[1] - obj.span[0]
                stats.total_clip_duration += duration
                stats.max_clip_duration = max(stats.max_clip_duration, duration)

    def _write_embeddings_parquet(self, video: Video) -> None:
        """Write aggregated embeddings to parquet."""
        video_id = f"{video.input_path}_{video.clip_chunk_index}"

        if self._iv2_buffer and not self.dry_run:
            path = get_full_path(self.output_path, f"iv2_embd_parquet/{video_id}.parquet")
            write_parquet(self._iv2_buffer, path, "iv2_embeddings", str(video.input_path), verbose=self.verbose, client=self.storage_client)
            self._iv2_buffer.clear()

        if self._ce1_buffer and not self.dry_run:
            path = get_full_path(self.output_path, f"ce1_embd_parquet/{video_id}.parquet")
            write_parquet(self._ce1_buffer, path, "ce1_embeddings", str(video.input_path), verbose=self.verbose, client=self.storage_client)
            self._ce1_buffer.clear()

    def _write_video_metadata(self, video: Video) -> None:
        """Write video-level metadata and clip chunk statistics."""
        if isinstance(video.input_video, pathlib.Path):
            input_video_path = video.input_video.as_posix()
        else:
            input_video_path = str(video.input_video)

        # Write video-level metadata from the first clip chunk only
        if video.clip_chunk_index == 0:
            video_data = {
                "video": input_video_path,
                "height": video.metadata.height,
                "width": video.metadata.width,
                "framerate": video.metadata.framerate,
                "num_frames": video.metadata.num_frames,
                "duration": video.metadata.duration,
                "video_codec": video.metadata.video_codec,
                "pixel_format": video.metadata.pixel_format,
                "audio_format": video.metadata.audio_codec,
                "num_total_clips": video.num_total_clips,
                "num_clip_chunks": video.num_clip_chunks,
            }

            # Generate video metadata path
            if not input_video_path.startswith(self.input_path):
                msg = f"Input video path {input_video_path} does not start with {self.input_path}"
                raise ValueError(msg)
            video_metadata_path = input_video_path[len(self.input_path):].lstrip("/") + ".json"
            video_dest = get_full_path(self.output_path, f"processed_videos/{video_metadata_path}")

            if not self.dry_run:
                write_json(video_data, video_dest, "video metadata", input_video_path,
                          verbose=self.verbose, client=self.storage_client)

        # Each clip chunk writes its own statistics and window captions
        chunk_data = {
            "video": input_video_path,
            "clip_chunk_index": video.clip_chunk_index,
            "num_clips_filtered_by_motion": video.clip_stats.num_filtered_by_motion,
            "num_clips_filtered_by_aesthetic": video.clip_stats.num_filtered_by_aesthetic,
            "num_clips_passed": video.clip_stats.num_passed,
            "num_clips_transcoded": video.clip_stats.num_transcoded,
            "num_clips_with_embeddings": video.clip_stats.num_with_embeddings,
            "num_clips_with_caption": video.clip_stats.num_with_caption,
            "num_clips_with_webp": video.clip_stats.num_with_webp,
            "total_clip_duration": video.clip_stats.total_clip_duration,
            "max_clip_duration": video.clip_stats.max_clip_duration,
            "clips": [str(clip.uuid) for clip in video.clips],
            "filtered_clips": [str(clip.uuid) for clip in video.filtered_clips],
            "all_windows": {},
            "all_windows_enhanced_caption": {},
        }

        # Collect all window captions organized by clip
        for clip in video.clips:
            clip_uuid = str(clip.uuid)
            chunk_data["all_windows"][clip_uuid] = {}
            chunk_data["all_windows_enhanced_caption"][clip_uuid] = {}

            for window in clip.windows:
                window_key = f"{window.start_frame}_{window.end_frame}"

                # Try each caption model in order, using the first one available
                for model in self.caption_models:
                    if model in window.caption:
                        chunk_data["all_windows"][clip_uuid][window_key] = window.caption[model]
                        break

                # Try each enhanced caption model in order, using the first one found
                for model in self.enhanced_caption_models:
                    if model in window.enhanced_caption:
                        chunk_data["all_windows_enhanced_caption"][clip_uuid][window_key] = window.enhanced_caption[model]
                        break

        # Generate clip chunk path
        clip_chunk_path = input_video_path[len(self.input_path):].lstrip("/") + f"_{video.clip_chunk_index}.json"
        chunk_dest = get_full_path(self.output_path, f"processed_clip_chunks/{clip_chunk_path}")

        if not self.dry_run:
            write_json(chunk_data, chunk_dest, "clip chunk metadata", input_video_path,
                      verbose=self.verbose, client=self.storage_client)

    def _cleanup_video_data(self, video: Video) -> None:
        """Clean up intermediate data (same as original)."""
        for clip in video.clips:
            clip.buffer = None
            clip.intern_video_2_embedding = None
            clip.cosmos_embed1_embedding = None
            for window in clip.windows:
                window.mp4_bytes = None
                window.qwen_llm_input = None
                window.caption.clear()
                window.enhanced_caption.clear()
                window.webp_bytes = None
