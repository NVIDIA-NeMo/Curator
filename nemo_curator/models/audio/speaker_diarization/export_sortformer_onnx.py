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

"""Export streaming Sortformer checkpoints with the recovered Riva contract.

The graph starts at acoustic features and exposes the six-input/four-output
streaming interface consumed by :class:`TensorRTSortformerAdapter`. Both legacy
and high-resolution v2.1 checkpoints are supported; the latter keep their
learned upsampler and return predictions at Riva's 80 ms cadence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Sequence


INPUT_NAMES = [
    "chunk",
    "chunk_lengths",
    "spkcache",
    "spkcache_lengths",
    "fifo",
    "fifo_lengths",
]
OUTPUT_NAMES = [
    "predictions",
    "pred_lengths",
    "chunk_embs",
    "chunk_emb_lengths",
]
DYNAMIC_AXES = {
    "chunk": {0: "batch_size", 1: "chunk_frames"},
    "chunk_lengths": {0: "batch_size"},
    "spkcache": {0: "batch_size", 1: "spkcache_len"},
    "spkcache_lengths": {0: "batch_size"},
    "fifo": {0: "batch_size", 1: "fifo_len"},
    "fifo_lengths": {0: "batch_size"},
    "predictions": {0: "batch_size", 1: "output_frames"},
    "pred_lengths": {0: "batch_size"},
    "chunk_embs": {0: "batch_size", 1: "emb_frames"},
    "chunk_emb_lengths": {0: "batch_size"},
}
_NATIVE_VALIDATION_ATOL = 1.0e-5


def _is_high_resolution(model: Any) -> bool:  # noqa: ANN401 - supports multiple NeMo model releases
    """Return whether a checkpoint uses NeMo's optional high-resolution head."""
    return bool(getattr(model, "high_resolution", False))


def _sortformer_model_class() -> type:
    try:
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
    except ImportError as exc:
        msg = "Sortformer ONNX export requires NeMo from the audio_common extra"
        raise ImportError(msg) from exc
    return SortformerEncLabelModel


class RivaStreamingExportMixin:
    """TensorRT-friendly export behavior shared by Sortformer checkpoints."""

    @property
    def input_names(self) -> list[str]:
        return INPUT_NAMES

    @property
    def output_names(self) -> list[str]:
        return OUTPUT_NAMES

    def _call_pre_encode(
        self,
        chunk: torch.Tensor,
        chunk_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Call the encoder API shared by released NeMo Sortformer checkpoints."""
        return self.encoder.pre_encode(x=chunk, lengths=chunk_lengths)

    @staticmethod
    def concat_and_pad(
        embs: Sequence[torch.Tensor],
        lengths: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate ragged cache, FIFO, and chunk allocations without loops."""
        spkcache, fifo, chunk_embs = embs
        spkcache_lengths, fifo_lengths, chunk_emb_lengths = lengths
        total_lengths = spkcache_lengths + fifo_lengths + chunk_emb_lengths
        allocated_frames = spkcache.shape[1] + fifo.shape[1] + chunk_embs.shape[1]
        positions = torch.arange(allocated_frames, device=spkcache.device).unsqueeze(0)
        positions = positions.expand(spkcache.shape[0], -1)

        fifo_start = spkcache_lengths.unsqueeze(1)
        chunk_start = (spkcache_lengths + fifo_lengths).unsqueeze(1)
        valid_end = total_lengths.unsqueeze(1)
        spkcache_mask = positions < fifo_start
        fifo_mask = (positions >= fifo_start) & (positions < chunk_start)
        chunk_mask = (positions >= chunk_start) & (positions < valid_end)
        embedding_dim = spkcache.shape[2]

        def gather_local(source: torch.Tensor, local_positions: torch.Tensor) -> torch.Tensor:
            local_positions = local_positions.clamp(min=0, max=source.shape[1] - 1)
            indices = local_positions.unsqueeze(2).expand(-1, -1, embedding_dim)
            return torch.gather(source, dim=1, index=indices)

        output = gather_local(spkcache, positions) * spkcache_mask.unsqueeze(2)
        output = output + gather_local(fifo, positions - fifo_start) * fifo_mask.unsqueeze(2)
        output = output + gather_local(chunk_embs, positions - chunk_start) * chunk_mask.unsqueeze(2)
        return output, total_lengths

    @staticmethod
    def export_rope_attention(
        attention: Any,  # noqa: ANN401 - NeMo attention variants are runtime-selected
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Lower the newer FlexAttention path to standard ONNX operators."""
        batch_size, num_frames, _ = hidden_states.shape
        num_heads, head_dim = attention.n_heads, attention.head_dim
        query_key_value = (
            attention.w_qkv(hidden_states).view(batch_size, num_frames, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        )
        query, key, value = query_key_value.unbind(0)
        if attention.qk_norm:
            query = attention.q_norm(query).to(value.dtype)
            key = attention.k_norm(key).to(value.dtype)
        query, key = attention.rope(query, key)
        valid_keys = torch.arange(num_frames, device=hidden_states.device).view(1, 1, 1, -1)
        valid_keys = valid_keys < lengths.view(-1, 1, 1, 1)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_keys,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_frames, attention.d_model)
        return attention.out_proj(attended)

    def export_flex_frontend_encoder(
        self,
        combined_embs: torch.Tensor,
        combined_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Export the v2.1 RoPE transformer without FlexAttention."""
        encoder = self.encoder
        if getattr(encoder, "self_attention_model", None) != "rope":
            msg = "Sortformer FlexAttention export requires self_attention_model='rope'"
            raise ValueError(msg)
        hidden_states = encoder.embed_norm(encoder.dropout_pre_encoder(combined_embs))
        for layer in encoder.layers:
            attended = self.export_rope_attention(
                layer.attn,
                layer.norm1(hidden_states),
                combined_lengths,
            )
            hidden_states = hidden_states + layer.drop(attended)
            hidden_states = hidden_states + layer.drop(layer.ffn(layer.norm2(hidden_states)))
        hidden_states = encoder.final_norm(hidden_states)
        if encoder.out_proj is not None:
            hidden_states = encoder.out_proj(hidden_states)
        return hidden_states, combined_lengths

    def uses_flex_frontend_encoder(self) -> bool:
        return self.encoder.__class__.__module__ == "nemo.collections.asr.modules.transformer_encoder"

    def forward_for_export(  # noqa: PLR0913 - fixed exported engine contract
        self,
        chunk: torch.Tensor,
        chunk_lengths: torch.Tensor,
        spkcache: torch.Tensor,
        spkcache_lengths: torch.Tensor,
        fifo: torch.Tensor,
        fifo_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute the recovered Riva streaming graph contract."""
        chunk_embs, chunk_emb_lengths = self._call_pre_encode(chunk, chunk_lengths)
        chunk_emb_lengths = chunk_emb_lengths.to(torch.int64)
        combined_embs, combined_lengths = self.concat_and_pad(
            [spkcache, fifo, chunk_embs],
            [spkcache_lengths, fifo_lengths, chunk_emb_lengths],
        )

        if self.uses_flex_frontend_encoder():
            encoded, pred_lengths = self.export_flex_frontend_encoder(combined_embs, combined_lengths)
            if self.sortformer_modules.encoder_proj is not None:
                encoded = self.sortformer_modules.encoder_proj(encoded)
        else:
            encoded, pred_lengths = self.frontend_encoder(
                processed_signal=combined_embs,
                processed_signal_length=combined_lengths,
                bypass_pre_encode=True,
            )
        encoder_mask = self.sortformer_modules.length_to_mask(pred_lengths, encoded.shape[1])
        transformed = self.transformer_encoder(encoder_states=encoded, encoder_mask=encoder_mask)
        if _is_high_resolution(self):
            transformed = transformed * encoder_mask.unsqueeze(-1)
            transformed = self.sortformer_modules.upsample_hidden(transformed)
            predictions = self.sortformer_modules.forward_speaker_sigmoids(transformed)
            predictions = self.sortformer_modules.downsample_preds(predictions, self.upsample_factor)
        else:
            predictions = self.sortformer_modules.forward_speaker_sigmoids(transformed)
        return predictions, pred_lengths, chunk_embs, chunk_emb_lengths


def _export_model_class(base: type) -> type:
    return type("RivaStreamingExportModel", (RivaStreamingExportMixin, base), {})


def round_model_tensors_through_bf16(model: torch.nn.Module) -> None:
    """Match the recovered BF16 artifact while retaining FP32 storage."""
    with torch.no_grad():
        for tensor in [*model.parameters(), *model.buffers()]:
            if tensor.is_floating_point():
                tensor.copy_(tensor.to(torch.bfloat16).to(torch.float32))


def save_learnable_silence(model: Any, output_path: Path) -> bool:  # noqa: ANN401
    learned_silence = getattr(model.sortformer_modules, "learnable_sil_emb", None)
    if learned_silence is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        output_path,
        learned_silence.detach().cpu().float().numpy(),
        allow_pickle=False,
    )
    return True


def promote_constant_nodes_to_initializers(onnx_path: Path) -> None:
    """Match the initializer layout consumed by the legacy conversion path."""
    try:
        import onnx
    except ImportError as exc:
        msg = "Sortformer ONNX postprocessing requires onnx"
        raise ImportError(msg) from exc
    model = onnx.load(str(onnx_path), load_external_data=True)
    retained_nodes = []
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            retained_nodes.append(node)
            continue
        value_attributes = [attribute for attribute in node.attribute if attribute.name == "value"]
        if len(value_attributes) != 1 or not value_attributes[0].HasField("t"):
            retained_nodes.append(node)
            continue
        tensor = value_attributes[0].t
        tensor.name = node.output[0]
        model.graph.initializer.append(tensor)
    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def make_input_example(model: Any) -> tuple[torch.Tensor, ...]:  # noqa: ANN401
    """Create small allocated tensors; the engine profile supplies real maxima."""
    batch_size = 1
    chunk_frames = 16
    feature_dim = int(model.cfg.preprocessor.features)
    embedding_dim = int(model.cfg.model_defaults.fc_d_model)
    cache_frames = min(int(model.cfg.sortformer_modules.spkcache_len), 8)
    fifo_frames = 8
    chunk = torch.rand(batch_size, chunk_frames, feature_dim, device=model.device)
    chunk_lengths = torch.tensor([chunk_frames], dtype=torch.int64, device=model.device)
    spkcache = torch.randn(batch_size, cache_frames, embedding_dim, device=model.device)
    spkcache_lengths = torch.tensor([max(1, cache_frames // 2)], dtype=torch.int64, device=model.device)
    fifo = torch.randn(batch_size, fifo_frames, embedding_dim, device=model.device)
    fifo_lengths = torch.tensor([3], dtype=torch.int64, device=model.device)
    return chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths


def export_checkpoint(  # noqa: PLR0913 - explicit build controls form the CLI contract
    nemo_model: Path,
    output_onnx: Path,
    *,
    device: str = "cpu",
    bf16_roundtrip: bool = True,
    promote_constants: bool = True,
    validate_native: bool = True,
    learnable_silence_output: Path | None = None,
) -> float | None:
    """Export one checkpoint and return the v2.1 lowering error when measured."""
    if not nemo_model.is_file():
        raise FileNotFoundError(nemo_model)
    model_class = _sortformer_model_class()
    model = model_class.restore_from(restore_path=str(nemo_model), map_location=device)
    model.eval().float()
    if learnable_silence_output is not None:
        save_learnable_silence(model, learnable_silence_output)
    if bf16_roundtrip:
        round_model_tensors_through_bf16(model)

    input_example = make_input_example(model)
    native_predictions = None
    if _is_high_resolution(model) and validate_native:
        with torch.no_grad():
            native_predictions = model.forward_for_export(*input_example)[0]

    model.__class__ = _export_model_class(model.__class__)
    try:
        from omegaconf import open_dict
    except ImportError as exc:
        msg = "Sortformer ONNX export requires omegaconf"
        raise ImportError(msg) from exc
    with open_dict(model.cfg):
        model.cfg.precision = "bf16_mixed"

    max_abs_error = None
    if native_predictions is not None:
        with torch.no_grad():
            exported_predictions = model.forward_for_export(*input_example)[0]
        exported_predictions = exported_predictions[:, : native_predictions.shape[1]]
        max_abs_error = float((native_predictions - exported_predictions).abs().max().item())
        if max_abs_error > _NATIVE_VALIDATION_ATOL:
            msg = (
                "Export-only Sortformer attention lowering failed numerical validation: "
                f"max absolute error {max_abs_error} exceeds 1e-5"
            )
            raise RuntimeError(msg)

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    model.export(
        output=str(output_onnx),
        input_example=input_example,
        onnx_opset_version=16,
        do_constant_folding=True,
        dynamic_axes=DYNAMIC_AXES,
        check_trace=False,
        use_dynamo=False,
    )
    if promote_constants:
        promote_constant_nodes_to_initializers(output_onnx)
    return max_abs_error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nemo_model", type=Path)
    parser.add_argument("output_onnx", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-bf16-roundtrip", action="store_true")
    parser.add_argument("--keep-constant-nodes", action="store_true")
    parser.add_argument("--skip-native-validation", action="store_true")
    parser.add_argument("--learnable-silence-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    max_abs_error = export_checkpoint(
        args.nemo_model,
        args.output_onnx,
        device=args.device,
        bf16_roundtrip=not args.no_bf16_roundtrip,
        promote_constants=not args.keep_constant_nodes,
        validate_native=not args.skip_native_validation,
        learnable_silence_output=args.learnable_silence_output,
    )
    if max_abs_error is not None:
        print(f"native_vs_export_max_abs={max_abs_error:.9g}")


if __name__ == "__main__":
    main()
