"""Embedding adapters for locally downloaded time-series foundation models."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

class ChronosChannelMeanAdapter:
    """Extract one Chronos-family vector per multivariate context window.

    Chronos-T5 and Chronos-Bolt accept univariate sequences. A multivariate
    batch shaped ``(B, C, W)`` is therefore flattened to ``(B*C, W)``.
    Chronos-2 accepts the native three-dimensional batch and returns one
    tensor per sample. In both cases encoder tokens are mean-pooled within
    each channel and channel vectors are mean-pooled to obtain ``(B, D)``.
    """

    def __init__(self, pipeline) -> None:
        if not callable(getattr(pipeline, "embed", None)):
            raise TypeError("The supplied Chronos pipeline does not expose embed()")
        self.pipeline = pipeline

    def embed(self, batch: np.ndarray) -> np.ndarray:
        values = np.asarray(batch, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                f"Expected (batch, channels, time), got shape {values.shape}"
            )

        n_windows, n_channels, window_size = values.shape

        if type(self.pipeline).__name__ == "Chronos2Pipeline":
            # Chronos-2 returns a list of B tensors shaped (C, tokens, D).
            with torch.inference_mode():
                sample_embeddings, _loc_scale = self.pipeline.embed(values)
            window_embeddings = torch.stack(
                [sample.float().mean(dim=(0, 1)) for sample in sample_embeddings]
            )
        else:
            contexts = torch.from_numpy(
                values.reshape(n_windows * n_channels, window_size)
            )
            with torch.inference_mode():
                token_embeddings, _tokenizer_state = self.pipeline.embed(contexts)

            # (B*C, tokens, D) -> (B*C, D) -> (B, C, D) -> (B, D)
            channel_embeddings = token_embeddings.float().mean(dim=1)
            window_embeddings = channel_embeddings.reshape(
                n_windows, n_channels, channel_embeddings.shape[-1]
            ).mean(dim=1)

        return window_embeddings.cpu().numpy().astype(np.float32, copy=False)


class MomentChannelMeanAdapter:
    """Use MOMENT's embedding task (mean over channels and patches)."""

    def __init__(self, model, device: str) -> None:
        self.model = model
        self.device = device

    def embed(self, batch: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            embeddings = self.model.embed(x_enc=values, reduction="mean").embeddings
        return embeddings.float().cpu().numpy().astype(np.float32, copy=False)


class MoiraiHiddenStateAdapter:
    """Mean-pool Moirai transformer states before its distribution head."""

    def __init__(self, model, device: str, patch_size: int = 16) -> None:
        if patch_size not in model.patch_sizes:
            raise ValueError(
                f"patch_size={patch_size} is not supported; "
                f"choose from {model.patch_sizes}"
            )
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.is_moe = type(model).__name__ == "MoiraiMoEModule"

    def embed(self, batch: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        if values.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time), got {values.shape}")
        n_windows, n_channels, window_size = values.shape
        if window_size % self.patch_size:
            raise ValueError(
                f"Window size {window_size} must be divisible by "
                f"patch size {self.patch_size}"
            )

        from uni2ts.common.torch_util import packed_attention_mask, packed_causal_attention_mask

        n_patches = window_size // self.patch_size
        patches = values.reshape(
            n_windows, n_channels, n_patches, self.patch_size
        ).reshape(n_windows, n_channels * n_patches, self.patch_size)
        max_patch = max(self.model.patch_sizes)
        target = torch.zeros(
            n_windows,
            n_channels * n_patches,
            max_patch,
            dtype=torch.float32,
            device=self.device,
        )
        target[..., : self.patch_size] = patches
        observed_mask = torch.zeros_like(target, dtype=torch.bool)
        observed_mask[..., : self.patch_size] = True
        sample_id = torch.ones(
            target.shape[:-1], dtype=torch.long, device=self.device
        )
        time_id = torch.arange(n_patches, device=self.device).repeat(
            n_windows, n_channels
        )
        variate_id = (
            torch.arange(n_channels, device=self.device)
            .repeat_interleave(n_patches)
            .repeat(n_windows, 1)
        )
        patch_size = torch.full_like(sample_id, self.patch_size)

        with torch.inference_mode():
            loc, scale = self.model.scaler(
                target, observed_mask, sample_id, variate_id
            )
            scaled = (target - loc) / scale
            if self.is_moe:
                hidden = F.silu(self.model.in_proj(scaled, patch_size))
                hidden = (
                    self.model.feat_proj(hidden, patch_size)
                    + self.model.res_proj(scaled, patch_size)
                )
                attention_mask = packed_causal_attention_mask(sample_id, time_id)
            else:
                hidden = self.model.in_proj(scaled, patch_size)
                attention_mask = packed_attention_mask(sample_id)
            hidden = self.model.encoder(
                hidden,
                attention_mask,
                time_id=time_id,
                var_id=variate_id,
            )
            embeddings = hidden.float().mean(dim=1)
        return embeddings.cpu().numpy().astype(np.float32, copy=False)


class TimesFm1HiddenStateAdapter:
    """Mean-pool TimesFM 1.0 decoder states before its forecast head."""

    def __init__(self, model, device: str) -> None:
        self.model = model
        self.device = device

    def embed(self, batch: np.ndarray) -> np.ndarray:
        values = np.asarray(batch, dtype=np.float32)
        n_windows, n_channels, window_size = values.shape
        flat = torch.as_tensor(
            values.reshape(n_windows * n_channels, window_size),
            dtype=torch.float32,
            device=self.device,
        )
        padding = torch.zeros_like(flat)
        frequency = torch.zeros(
            (flat.shape[0], 1), dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            model_input, patched_padding, _stats, _ = self.model._preprocess_input(
                input_ts=flat, input_padding=padding
            )
            model_input = model_input + self.model.freq_emb(frequency)
            hidden = self.model.stacked_transformer(model_input, patched_padding)
            channels = hidden.float().mean(dim=1)
            embeddings = channels.reshape(n_windows, n_channels, -1).mean(dim=1)
        return embeddings.cpu().numpy().astype(np.float32, copy=False)


class TimesFm25HiddenStateAdapter:
    """Mean-pool TimesFM 2.5 decoder states before its forecast heads."""

    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper
        self.model = wrapper.model
        self.device = self.model.device

    def embed(self, batch: np.ndarray) -> np.ndarray:
        from timesfm.torch import util

        values = np.asarray(batch, dtype=np.float32)
        n_windows, n_channels, window_size = values.shape
        patch_len = self.model.p
        pad_len = (-window_size) % patch_len
        flat = torch.as_tensor(
            values.reshape(n_windows * n_channels, window_size),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.zeros_like(flat, dtype=torch.bool)
        if pad_len:
            flat = torch.cat(
                [torch.zeros(flat.shape[0], pad_len, device=self.device), flat], dim=1
            )
            masks = torch.cat(
                [
                    torch.ones(
                        masks.shape[0],
                        pad_len,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    masks,
                ],
                dim=1,
            )
        patches = flat.reshape(flat.shape[0], -1, patch_len)
        patch_masks = masks.reshape(masks.shape[0], -1, patch_len)

        with torch.inference_mode():
            count = torch.zeros(flat.shape[0], device=self.device)
            mean = torch.zeros(flat.shape[0], device=self.device)
            std = torch.zeros(flat.shape[0], device=self.device)
            means, stds = [], []
            for patch_index in range(patches.shape[1]):
                (count, mean, std), _ = util.update_running_stats(
                    count,
                    mean,
                    std,
                    patches[:, patch_index],
                    patch_masks[:, patch_index],
                )
                means.append(mean)
                stds.append(std)
            means = torch.stack(means, dim=1)
            stds = torch.stack(stds, dim=1)
            normalized = util.revin(patches, means, stds, reverse=False)
            normalized = torch.where(patch_masks, 0.0, normalized)
            (input_embeddings, hidden, _point, _quantiles), _ = self.model(
                normalized, patch_masks
            )
            del input_embeddings
            channels = hidden.float().mean(dim=1)
            embeddings = channels.reshape(n_windows, n_channels, -1).mean(dim=1)
        return embeddings.cpu().numpy().astype(np.float32, copy=False)


def _load_timesfm1(model_path: Path, device: str, source_root: Path):
    module_path = source_root / "timesfm" / "pytorch_patched_decoder.py"
    spec = importlib.util.spec_from_file_location(
        "timesfm1_patched_decoder", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TimesFM 1.0 implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PatchedTimeSeriesDecoder(module.TimesFMConfig())
    state = torch.load(
        model_path / "torch_model.ckpt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return TimesFm1HiddenStateAdapter(model, device)


def load_foundation_adapter(
    backend: str,
    model_path: Path,
    device: str,
    *,
    timesfm1_source: Path | None = None,
):
    """Load a local model and return the matching embedding adapter."""
    if backend == "chronos":
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained(model_path, device_map=device)
        return ChronosChannelMeanAdapter(pipeline)

    if backend == "moment":
        from momentfm import MOMENTPipeline
        model = MOMENTPipeline.from_pretrained(
            model_path,
            model_kwargs={"task_name": "embedding"},
            local_files_only=True,
        )
        model.init()
        model.to(device).eval()
        return MomentChannelMeanAdapter(model, device)

    if backend in {"moirai", "moirai_moe"}:
        if backend == "moirai":
            from uni2ts.model.moirai import MoiraiModule as module_class
        else:
            from uni2ts.model.moirai_moe import MoiraiMoEModule as module_class
        model = module_class.from_pretrained(model_path, local_files_only=True)
        model.to(device).eval()
        return MoiraiHiddenStateAdapter(model, device)

    if backend == "timesfm1":
        if timesfm1_source is None:
            raise ValueError("timesfm1_source is required for the TimesFM 1.0 backend")
        return _load_timesfm1(model_path, device, timesfm1_source)

    if backend == "timesfm25":
        import timesfm
        wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_path, torch_compile=False, local_files_only=True
        )
        return TimesFm25HiddenStateAdapter(wrapper)

    raise ValueError(f"Unknown foundation-model backend: {backend}")
