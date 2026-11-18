from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any
from ramp_hsr.routes import (
    neuron_chain_feature_keys,
    neuron_pair_feature_keys,
    load_neuron_risk_policy,
    neuron_risk_default_metrics,
    neuron_risk_feature_spec,
)
from ramp_hsr.tokenizer_context import (
    assert_tokenizer_context_policy,
    configure_tokenizer_context_policy,
    tokenizer_context_policy_metadata,
    validate_max_context_length,
)
from ramp_hsr.devices import normalize_torch_device

CHINESE_MIRRORS = [
    "https://hf-mirror.com",
    "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
]
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
DEFAULT_TOP_K = 20


def _is_truthy_env(name: str) -> bool:
    """Return True when an environment variable is set to a truthy value."""
    value = os.getenv(name)
    return value is not None and value.strip().lower() in _TRUTHY_ENV_VALUES


def _resolve_local_model_path(model_name: str) -> Path | None:
    """Resolve a local model path if the provided model name exists on disk."""
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return None


def _configure_hf_environment(
    *, use_mirror: bool, cache_dir: str | None, offline: bool, is_local_model: bool
) -> None:
    """Configure Hugging Face environment variables before importing the stack."""
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if (
        use_mirror
        and (not offline)
        and (not is_local_model)
        and (not os.getenv("HF_ENDPOINT"))
    ):
        os.environ["HF_ENDPOINT"] = CHINESE_MIRRORS[0]


def _import_hf_stack() -> dict[str, Any]:
    """Import the Hugging Face stack after environment configuration."""
    import torch
    from huggingface_hub import constants as hf_constants
    from huggingface_hub import snapshot_download
    from transformers.utils import import_utils

    import_utils._torchvision_available = False
    import_utils._librosa_available = False
    import_utils._torchaudio_available = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return {
        "torch": torch,
        "hf_constants": hf_constants,
        "snapshot_download": snapshot_download,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _batch_neuron_chain_risks(
    *, torch: Any, stacked_layers: Any, sample_layers: list[int]
) -> dict[str, Any]:
    policy = load_neuron_risk_policy()
    batch_size = int(stacked_layers.shape[1]) if stacked_layers.ndim == 3 else 0
    zeros = torch.zeros(batch_size, device=stacked_layers.device)
    if policy is None or batch_size <= 0:
        return {key: zeros for key in neuron_risk_default_metrics(policy)}
    layer_to_position = {
        int(layer_idx): int(position)
        for (position, layer_idx) in enumerate(sample_layers)
    }
    signed_means = stacked_layers.mean(dim=-1)
    signed_stds = stacked_layers.std(dim=-1, unbiased=False).clamp_min(1e-06)
    abs_layers = stacked_layers.abs()
    abs_means = abs_layers.mean(dim=-1)
    abs_stds = abs_layers.std(dim=-1, unbiased=False).clamp_min(1e-06)
    chain_rows: dict[str, dict[str, Any]] = {}
    weighted_scores: list[Any] = []
    active_masks: list[Any] = []
    for chain in policy.chains:
        node_scores: list[Any] = []
        node_active_masks: list[Any] = []
        for node in chain.nodes:
            position = layer_to_position.get(int(node.layer_index))
            if position is None:
                continue
            if node.dimension < 0 or node.dimension >= stacked_layers.shape[-1]:
                continue
            values = stacked_layers[position, :, node.dimension]
            direction = str(node.direction or "absolute").lower()
            if direction == "positive":
                z_scores = (values - signed_means[position]) / signed_stds[position]
            elif direction == "negative":
                z_scores = (signed_means[position] - values) / signed_stds[position]
            else:
                z_scores = (values.abs() - abs_means[position]) / abs_stds[position]
            node_scores.append(torch.clamp(z_scores, min=0.0))
            node_active_masks.append(z_scores >= float(policy.node_z_threshold))
        if not node_scores:
            score = zeros
            active_fraction = zeros
            active_mask = zeros > 0
        else:
            score = torch.stack(node_scores, dim=0).mean(dim=0) * float(chain.weight)
            active_fraction = torch.stack(node_active_masks, dim=0).float().mean(dim=0)
            active_mask = active_fraction >= float(policy.chain_active_fraction)
        chain_rows[chain.name] = {
            "score": score,
            "active_fraction": active_fraction,
            "active_mask": active_mask,
        }
        weighted_scores.append(score)
        active_masks.append(active_mask.float())
    max_risk = (
        torch.stack(weighted_scores, dim=0).max(dim=0).values
        if weighted_scores
        else zeros
    )
    mean_risk = (
        torch.stack(weighted_scores, dim=0).mean(dim=0) if weighted_scores else zeros
    )
    active_fraction = (
        torch.stack(active_masks, dim=0).mean(dim=0) if active_masks else zeros
    )
    pair_scores: list[Any] = []
    pair_active_masks: list[Any] = []
    pair_rows: dict[str, dict[str, Any]] = {}
    for pair in policy.pairs:
        left = chain_rows.get(pair.left)
        right = chain_rows.get(pair.right)
        if left is None or right is None:
            continue
        synergy = torch.minimum(left["active_fraction"], right["active_fraction"])
        pair_score = (
            0.5 * (left["score"] + right["score"]) * synergy * float(pair.weight)
        )
        pair_scores.append(pair_score)
        pair_active_mask = left["active_mask"] & right["active_mask"]
        pair_active_masks.append(pair_active_mask)
        pair_rows[pair.name] = {"score": pair_score, "active_mask": pair_active_mask}
    pair_risk = (
        torch.stack(pair_scores, dim=0).max(dim=0).values if pair_scores else zeros
    )
    pair_active_fraction = (
        torch.stack(pair_active_masks, dim=0).float().mean(dim=0)
        if pair_active_masks
        else zeros
    )
    composite = (
        0.5 * max_risk
        + 0.2 * mean_risk
        + 0.2 * pair_risk
        + 0.05 * active_fraction
        + 0.05 * pair_active_fraction
    )
    metrics = {
        "neuron_chain_max_risk": max_risk,
        "neuron_chain_mean_risk": mean_risk,
        "neuron_chain_active_fraction": active_fraction,
        "neuron_chain_pair_risk": pair_risk,
        "neuron_chain_pair_active_fraction": pair_active_fraction,
        "neuron_chain_composite_risk": composite,
    }
    feature_spec = neuron_risk_feature_spec(policy)
    for feature_name in feature_spec["numeric"]:
        if feature_name in metrics:
            continue
        metrics[feature_name] = zeros
    for feature_name in feature_spec["bool"]:
        metrics[feature_name] = zeros
    for chain in policy.chains:
        keys = neuron_chain_feature_keys(chain.name)
        row = chain_rows.get(chain.name)
        if row is None:
            continue
        metrics[keys["score"]] = row["score"]
        metrics[keys["active_fraction"]] = row["active_fraction"]
        metrics[keys["active"]] = row["active_mask"].float()
    for pair in policy.pairs:
        keys = neuron_pair_feature_keys(pair.name)
        row = pair_rows.get(pair.name)
        if row is None:
            continue
        metrics[keys["score"]] = row["score"]
        metrics[keys["active"]] = row["active_mask"].float()
    return metrics


class QwenCoderLLM:
    """Real LLM interface using Qwen2.5-Coder-1.5B.

    This class provides:
    - Hidden state extraction from multiple layers
    - Logits computation for next token prediction
    - Uncertainty feature calculation

    For repo identifiers, the model is first resolved to a local Hugging Face
    snapshot path. The tokenizer and model are then loaded from that local
    path to avoid follow-up remote metadata requests during initialization.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B",
        device: str | None = None,
        use_mirror: bool = False,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        offline: bool | None = None,
    ):
        """Initialize Qwen Coder LLM.

        Args:
            model_name: Model name or path.
            device: Device to run model on (auto-detect if None).
            use_mirror: Whether to use Chinese mirrors for download.
            cache_dir: Custom cache directory for models.
            local_files_only: Load only from a local Hugging Face cache.
            offline: Force offline mode using only local files.
        """
        self.model_name = model_name
        self.cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None
        self.local_model_path = _resolve_local_model_path(model_name)
        env_offline = _is_truthy_env("HF_HUB_OFFLINE") or _is_truthy_env(
            "TRANSFORMERS_OFFLINE"
        )
        self.offline = env_offline if offline is None else offline
        self.local_files_only = (
            self.offline if local_files_only is None else local_files_only
        )
        _configure_hf_environment(
            use_mirror=use_mirror,
            cache_dir=self.cache_dir,
            offline=self.offline,
            is_local_model=self.local_model_path is not None,
        )
        hf_stack = _import_hf_stack()
        self._torch = hf_stack["torch"]
        self._snapshot_download = hf_stack["snapshot_download"]
        self._AutoTokenizer = hf_stack["AutoTokenizer"]
        self._AutoModelForCausalLM = hf_stack["AutoModelForCausalLM"]
        self.hf_endpoint = hf_stack["hf_constants"].ENDPOINT
        if device is None:
            self.device = "cuda" if self._torch.cuda.is_available() else "cpu"
        else:
            self.device = normalize_torch_device(device)
        self._using_cuda = self.device.startswith("cuda")
        print(f"Loading Qwen Coder model: {model_name}")
        print(f"Using device: {self.device}")
        if self.cache_dir:
            print(f"Using cache directory: {self.cache_dir}")
        if self.local_model_path is not None:
            print(f"Using local model path: {self.local_model_path}")
        else:
            print(f"Effective Hugging Face endpoint: {self.hf_endpoint}")
            if self.local_files_only:
                print("Using local-files-only mode.")
        try:
            self.model_path = self._resolve_model_path()
            self.tokenizer = self._AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=False, local_files_only=True
            )
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                if getattr(self.tokenizer, "eos_token", None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif getattr(self.tokenizer, "bos_token", None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.bos_token
            configure_tokenizer_context_policy(self.tokenizer)
            assert_tokenizer_context_policy(
                self.tokenizer, context="QwenCoderLLM tokenizer"
            )
            if self._using_cuda:
                visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if visible_devices:
                    device_map = {"": "cuda:0"}
                else:
                    device_map = {"": "cuda:0"}
                print(
                    f"Loading model on GPU with device_map={device_map} (CUDA_VISIBLE_DEVICES={visible_devices or 'not set'})"
                )
                self.model = self._AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=self._torch.float16,
                    device_map=device_map,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            else:
                self.model = self._AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=self._torch.float32,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                self.model = self.model.to(self.device)
            self.model.eval()
            self._supports_logits_to_keep: bool | None = None
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _resolve_model_path(self) -> str:
        """Resolve a model name to a local directory before loading."""
        if self.local_model_path is not None:
            return str(self.local_model_path)
        snapshot_kwargs: dict[str, Any] = {"repo_id": self.model_name}
        if self.cache_dir:
            snapshot_kwargs["cache_dir"] = self.cache_dir
        try:
            snapshot_path = self._snapshot_download(
                local_files_only=True, **snapshot_kwargs
            )
            print(f"Using cached model snapshot: {snapshot_path}")
            return snapshot_path
        except Exception as local_cache_error:
            if self.local_files_only:
                raise RuntimeError(
                    f"Model '{self.model_name}' was not found in the local Hugging Face cache. Use a local model directory or pre-download the model before enabling offline/local-files-only mode."
                ) from local_cache_error
        print(
            f"No cached snapshot found for {self.model_name}, downloading via {self.hf_endpoint}"
        )
        try:
            snapshot_path = self._snapshot_download(
                local_files_only=False, **snapshot_kwargs
            )
        except Exception as download_error:
            raise RuntimeError(
                f"Failed to download model '{self.model_name}' from {self.hf_endpoint}. If the environment should stay offline, set `model_name` to a local model path or pre-download the model into the Hugging Face cache."
            ) from download_error
        print(f"Downloaded model snapshot: {snapshot_path}")
        return snapshot_path

    def _get_input_ids(self, prefix: str) -> Any:
        """Tokenize a prefix, ensuring at least one token exists.

        Some file-start samples have an empty prefix. Qwen cannot run a forward
        pass with sequence length 0, so we fall back to a single BOS/EOS token.
        """
        input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids
        if input_ids.shape[-1] > 0:
            return input_ids.to(self.device)
        for token_id in (self.tokenizer.bos_token_id, self.tokenizer.eos_token_id):
            if token_id is not None:
                return self._torch.tensor([[token_id]], device=self.device)
        fallback_ids = self.tokenizer("\n", return_tensors="pt").input_ids
        if fallback_ids.shape[-1] == 0:
            raise ValueError(
                "Tokenizer produced zero tokens for both prefix and fallback inputs"
            )
        return fallback_ids.to(self.device)

    def tokenizer_context_policy(
        self, *, max_length: int | None = None
    ) -> dict[str, Any]:
        """Return the active tokenizer context policy for audit summaries."""
        if max_length is not None:
            validate_max_context_length(max_length, context="QwenCoderLLM")
        assert_tokenizer_context_policy(
            self.tokenizer, context="QwenCoderLLM tokenizer"
        )
        return tokenizer_context_policy_metadata(self.tokenizer, max_length=max_length)

    def get_hidden_state(self, prefix: str, layer: int = -1) -> list[float]:
        """Get hidden state for a given prefix at a specific layer.

        Args:
            prefix: The code prefix.
            layer: Layer index (-1 for last layer, 0-N for specific layers).

        Returns:
            Hidden state vector.
        """
        input_ids = self._get_input_ids(prefix)
        with self._torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if layer == -1:
            selected_hidden = hidden_states[-1]
        else:
            selected_hidden = hidden_states[layer]
        last_token_hidden = selected_hidden[0, -1, :]
        return last_token_hidden.cpu().numpy().tolist()

    def get_logits(self, prefix: str) -> list[float]:
        """Get logits for next token prediction.

        Args:
            prefix: The code prefix.

        Returns:
            Logits vector for vocabulary.
        """
        input_ids = self._get_input_ids(prefix)
        with self._torch.no_grad():
            outputs = self.model(input_ids)
        logits = outputs.logits[0, -1, :]
        return logits.cpu().numpy().tolist()

    def get_layer_hidden_states(
        self, prefix: str, sample_layers: list[int] | None = None
    ) -> dict[int, list[float]]:
        """Get hidden states from multiple layers.

        Args:
            prefix: The code prefix.
            sample_layers: List of layer indices to sample.
                          Defaults to [2, 5, 8, 11] for 12-layer model.

        Returns:
            Dictionary mapping layer index to hidden state vector.
        """
        if sample_layers is None:
            total_layers = self.model.config.num_hidden_layers
            step = max(1, total_layers // 4)
            sample_layers = [step, step * 2, step * 3, total_layers - 1]
        input_ids = self._get_input_ids(prefix)
        with self._torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states
        layer_states = {}
        for layer_idx in sample_layers:
            if layer_idx < len(all_hidden_states):
                hidden = all_hidden_states[layer_idx][0, -1, :]
                layer_states[layer_idx] = hidden.cpu().numpy().tolist()
        return layer_states

    def analyze_prefix(
        self,
        prefix: str,
        k: int = DEFAULT_TOP_K,
        sample_layers: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run a single forward pass and return aligned next-token features."""
        results = self.analyze_prefix_batch([prefix], k=k, sample_layers=sample_layers)
        return results[0]

    def _resolve_sample_layers(self, sample_layers: list[int] | None) -> list[int]:
        if sample_layers is not None:
            return sample_layers
        total_layers = self.model.config.num_hidden_layers
        step = max(1, total_layers // 4)
        return [step, step * 2, step * 3, total_layers - 1]

    @staticmethod
    def _sanitize_uncertainty_value(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            return 0.0
        return value

    @classmethod
    def _sanitize_uncertainty_dict(
        cls, raw_uncertainty: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        sanitized: dict[str, float] = {}
        sanitized_fields: list[str] = []
        for key, value in raw_uncertainty.items():
            clean_value = cls._sanitize_uncertainty_value(value)
            sanitized[key] = clean_value
            if clean_value != float(value):
                sanitized_fields.append(key)
        diagnostics = {
            "sanitized_uncertainty_fields": sanitized_fields,
            "sanitized_uncertainty_count": len(sanitized_fields),
            "had_sanitization": bool(sanitized_fields),
        }
        return (sanitized, diagnostics)

    def _decode_token_ids(self, token_ids: list[int]) -> list[str]:
        if not token_ids:
            return []
        if not hasattr(self.tokenizer, "batch_decode"):
            return [self.tokenizer.decode([token_id]) for token_id in token_ids]
        return self.tokenizer.batch_decode(
            [[token_id] for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    @staticmethod
    def _build_top_k_details(
        *,
        token_ids: list[int],
        tokens: list[str],
        probs: list[float],
        logprobs: list[float],
        logits: list[float],
    ) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for rank, (token_id, token, prob, logprob, logit) in enumerate(
            zip(token_ids, tokens, probs, logprobs, logits), start=1
        ):
            details.append(
                {
                    "rank": rank,
                    "token_id": int(token_id),
                    "token": str(token),
                    "prob": float(prob),
                    "logprob": float(logprob),
                    "logit": float(logit),
                }
            )
        return details

    def _forward_batched_model(
        self,
        input_ids,
        attention_mask,
        *,
        output_hidden_states: bool,
        use_last_position_only: bool,
    ):
        supports_logits_to_keep = getattr(self, "_supports_logits_to_keep", None)
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": output_hidden_states,
        }
        if use_last_position_only and supports_logits_to_keep is not False:
            try:
                outputs = self.model(**kwargs, logits_to_keep=1)
                self._supports_logits_to_keep = True
                return outputs
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                self._supports_logits_to_keep = False
        return self.model(**kwargs)

    def extract_last_hidden_states_batch(
        self, prefixes: list[str], *, layer: int = -1, max_length: int = 512
    ) -> list[Any]:
        """Extract final-position hidden states for repair-MoE training."""
        if not prefixes:
            return []
        max_length = validate_max_context_length(max_length, context="h_last export")
        assert_tokenizer_context_policy(
            self.tokenizer, context="QwenCoderLLM.extract_last_hidden_states_batch"
        )
        torch = self._torch
        fallback_id = None
        for tid in (self.tokenizer.bos_token_id, self.tokenizer.eos_token_id):
            if tid is not None:
                fallback_id = tid
                break
        sanitized = []
        for prefix in prefixes:
            if prefix == "":
                if fallback_id is not None:
                    sanitized.append(self.tokenizer.decode([fallback_id]))
                else:
                    sanitized.append("\n")
            else:
                sanitized.append(prefix)
        encoded = self.tokenizer(
            sanitized,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        use_last_position_only = bool(torch.all(attention_mask[:, -1] == 1).item())
        with torch.inference_mode():
            outputs = self._forward_batched_model(
                input_ids,
                attention_mask,
                output_hidden_states=True,
                use_last_position_only=use_last_position_only,
            )
        hidden_states = outputs.hidden_states
        layer_index = len(hidden_states) - 1 if layer == -1 else int(layer)
        if layer_index < 0:
            layer_index = len(hidden_states) + layer_index
        if layer_index < 0 or layer_index >= len(hidden_states):
            raise IndexError(
                f"Hidden-state layer {layer} is out of range for {len(hidden_states)} available layers"
            )
        selected = hidden_states[layer_index]
        if use_last_position_only:
            last_hidden = selected[:, -1, :]
        else:
            last_hidden = selected[batch_indices, seq_lengths, :]
        return [vector.detach().cpu().float() for vector in last_hidden]

    def extract_hidden_repair_batch(
        self, prefixes: list[str], *, max_length: int = 512
    ) -> dict[str, Any]:
        """Return only the tensors needed by a hidden-only repair editor.

        This is the packed-cache fast path.  It deliberately avoids computing
        uncertainty, route, probability, and decoded top-k diagnostics.  The
        base top-1 token and post-final-norm hidden state still come from the
        same batched model forward used by :meth:`analyze_prefix_batch_fast`.
        Keeping this operation in the model wrapper makes cache construction
        reusable across languages and editor architectures without creating
        one ``torch.save`` file per sample.
        """
        if not prefixes:
            return {
                "h_last": self._torch.empty((0, 0), dtype=self._torch.float32),
                "base_top1_token_id": self._torch.empty((0,), dtype=self._torch.long),
            }
        max_length = validate_max_context_length(
            max_length, context="hidden-only repair cache extraction"
        )
        assert_tokenizer_context_policy(
            self.tokenizer, context="QwenCoderLLM.extract_hidden_repair_batch"
        )
        torch = self._torch
        fallback_id = None
        for token_id in (self.tokenizer.bos_token_id, self.tokenizer.eos_token_id):
            if token_id is not None:
                fallback_id = token_id
                break
        sanitized = []
        for prefix in prefixes:
            if prefix == "":
                if fallback_id is not None:
                    sanitized.append(self.tokenizer.decode([fallback_id]))
                else:
                    sanitized.append("\n")
            else:
                sanitized.append(prefix)
        encoded = self.tokenizer(
            sanitized,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        use_last_position_only = bool(torch.all(attention_mask[:, -1] == 1).item())
        with torch.inference_mode():
            outputs = self._forward_batched_model(
                input_ids,
                attention_mask,
                output_hidden_states=True,
                use_last_position_only=use_last_position_only,
            )
        if outputs.logits.shape[1] == 1:
            logits = outputs.logits[:, 0, :]
        else:
            logits = outputs.logits[batch_indices, seq_lengths, :]
        final_hidden_states = outputs.hidden_states[-1]
        if use_last_position_only:
            h_last = final_hidden_states[:, -1, :]
        else:
            h_last = final_hidden_states[batch_indices, seq_lengths, :]
        return {
            "h_last": h_last.detach().cpu().float().contiguous(),
            "base_top1_token_id": logits.argmax(dim=-1).detach().cpu().long(),
        }

    def _resolve_transformer_layers(self):
        candidates = [
            getattr(getattr(self.model, "model", None), "layers", None),
            getattr(getattr(self.model, "transformer", None), "h", None),
            getattr(getattr(self.model, "gpt_neox", None), "layers", None),
        ]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "__len__"):
                return candidate
        raise RuntimeError(
            "Unable to locate transformer layers for activation ablation"
        )

    def _map_hidden_state_layer_to_module_index(self, layer_index: int) -> int:
        layers = self._resolve_transformer_layers()
        num_layers = len(layers)
        candidate = int(layer_index) - 1
        if 0 <= candidate < num_layers:
            return candidate
        if 0 <= int(layer_index) < num_layers:
            return int(layer_index)
        raise IndexError(
            f"Hidden-state layer index {layer_index} does not map to a transformer layer (available layers: 0..{num_layers - 1})"
        )

    def analyze_prefix_batch_with_ablation(
        self,
        prefixes: list[str],
        *,
        ablations: list[dict[str, Any]],
        k: int = DEFAULT_TOP_K,
        sample_layers: list[int] | None = None,
        max_length: int = 512,
    ) -> list[dict[str, Any]]:
        if not ablations:
            return self.analyze_prefix_batch(
                prefixes, k=k, sample_layers=sample_layers, max_length=max_length
            )
        torch = self._torch
        grouped: dict[int, list[dict[str, Any]]] = {}
        for entry in ablations:
            layer_index = int(entry["layer_index"])
            module_index = self._map_hidden_state_layer_to_module_index(layer_index)
            grouped.setdefault(module_index, []).append(
                {
                    "dimension": int(entry["dimension"]),
                    "value": float(entry.get("value", 0.0)),
                    "op": str(entry.get("op") or "set").lower(),
                    "scale": float(entry.get("scale", 1.0)),
                }
            )
        handles = []
        layers = self._resolve_transformer_layers()
        for module_index, entries in grouped.items():
            layer_module = layers[module_index]

            def _ablate_hook(module, inputs, output, *, entries=entries):
                hidden = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(hidden):
                    return output
                if hidden.ndim != 3:
                    return output
                modified = hidden.clone()
                hidden_dim = modified.shape[-1]
                for entry in entries:
                    dimension = int(entry["dimension"])
                    if 0 <= dimension < hidden_dim:
                        value = float(entry["value"])
                        op = str(entry.get("op") or "set").lower()
                        if op == "add":
                            modified[:, -1, dimension] = (
                                modified[:, -1, dimension] + value
                            )
                        elif op == "blend":
                            scale = float(entry.get("scale", 1.0))
                            modified[:, -1, dimension] = modified[
                                :, -1, dimension
                            ] + scale * (value - modified[:, -1, dimension])
                        elif op == "scale":
                            modified[:, -1, dimension] = (
                                modified[:, -1, dimension] * value
                            )
                        else:
                            modified[:, -1, dimension] = value
                if isinstance(output, tuple):
                    updated = list(output)
                    updated[0] = modified
                    return tuple(updated)
                return modified

            handles.append(layer_module.register_forward_hook(_ablate_hook))
        try:
            return self.analyze_prefix_batch(
                prefixes, k=k, sample_layers=sample_layers, max_length=max_length
            )
        finally:
            for handle in handles:
                handle.remove()

    def analyze_prefix_batch_fast(
        self,
        prefixes: list[str],
        k: int = DEFAULT_TOP_K,
        sample_layers: list[int] | None = None,
        max_length: int = 512,
    ) -> list[dict[str, Any]]:
        """Run a batched forward pass optimized for feature extraction.

        This path computes uncertainty summaries mostly on-device and only
        materializes the compact outputs needed by the training pipeline.
        """
        if not prefixes:
            return []
        max_length = validate_max_context_length(
            max_length, context="feature extraction"
        )
        assert_tokenizer_context_policy(
            self.tokenizer, context="QwenCoderLLM.analyze_prefix_batch_fast"
        )
        torch = self._torch
        fallback_id = None
        for tid in (self.tokenizer.bos_token_id, self.tokenizer.eos_token_id):
            if tid is not None:
                fallback_id = tid
                break
        sanitized = []
        for p in prefixes:
            if p == "":
                if fallback_id is not None:
                    sanitized.append(self.tokenizer.decode([fallback_id]))
                else:
                    sanitized.append("\n")
            else:
                sanitized.append(p)
        encoded = self.tokenizer(
            sanitized,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        use_last_position_only = bool(torch.all(attention_mask[:, -1] == 1).item())
        with torch.inference_mode():
            outputs = self._forward_batched_model(
                input_ids,
                attention_mask,
                output_hidden_states=True,
                use_last_position_only=use_last_position_only,
            )
        if outputs.logits.shape[1] == 1:
            logits = outputs.logits[:, 0, :]
        else:
            logits = outputs.logits[batch_indices, seq_lengths, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        top_k_val = min(k, probs.shape[-1])
        (top_k_probs, top_k_indices) = torch.topk(probs, top_k_val, dim=-1)
        top_k_log_probs = torch.gather(log_probs, dim=-1, index=top_k_indices)
        top_k_logits = torch.gather(logits, dim=-1, index=top_k_indices)
        top2_val = min(2, logits.shape[-1])
        (top2_logits, _) = torch.topk(logits, top2_val, dim=-1)
        if top2_val >= 2:
            margins = top2_logits[:, 0] - top2_logits[:, 1]
        else:
            margins = torch.zeros(logits.shape[0], device=logits.device)
        entropies = -(probs * log_probs).sum(dim=-1)
        max_softmax_probabilities = top_k_probs[:, 0]
        energy_scores = -torch.logsumexp(logits.float(), dim=-1)
        sampling_disagreement_probabilities = 1.0 - probs.float().pow(2).sum(dim=-1)
        sample_layers = self._resolve_sample_layers(sample_layers)
        hidden_states = outputs.hidden_states
        layer_vectors: list[Any] = []
        if use_last_position_only:
            final_hidden = hidden_states[-1][:, -1, :]
        else:
            final_hidden = hidden_states[-1][batch_indices, seq_lengths, :]
        for layer_idx in sample_layers:
            if layer_idx >= len(hidden_states):
                continue
            if use_last_position_only:
                hidden = hidden_states[layer_idx][:, -1, :]
            else:
                hidden = hidden_states[layer_idx][batch_indices, seq_lengths, :]
            layer_vectors.append(hidden.float())
        stacked = (
            torch.stack(layer_vectors, dim=0)
            if layer_vectors
            else torch.empty(
                (0, logits.shape[0], final_hidden.shape[-1]),
                device=logits.device,
                dtype=torch.float32,
            )
        )
        if len(layer_vectors) >= 2:
            nonfinite_hidden_mask = ~torch.isfinite(stacked)
            hidden_nonfinite_per_sample = nonfinite_hidden_mask.any(dim=0).any(dim=-1)
            stacked = torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
            representation_variances = stacked.var(dim=0, unbiased=False).mean(dim=-1)
            normalized_layers = torch.nn.functional.normalize(
                stacked, p=2, dim=-1, eps=1e-12
            )
            similarities = torch.einsum(
                "lbd,mbd->blm", normalized_layers, normalized_layers
            )
            upper = torch.triu_indices(
                similarities.shape[1],
                similarities.shape[2],
                offset=1,
                device=similarities.device,
            )
            layer_disagreements = (1.0 - similarities[:, upper[0], upper[1]]).mean(
                dim=-1
            )
            neuron_chain_risks = _batch_neuron_chain_risks(
                torch=torch, stacked_layers=stacked, sample_layers=sample_layers
            )
        else:
            representation_variances = torch.zeros(
                logits.shape[0], device=logits.device
            )
            layer_disagreements = torch.zeros(logits.shape[0], device=logits.device)
            hidden_nonfinite_per_sample = torch.zeros(
                logits.shape[0], dtype=torch.bool, device=logits.device
            )
            neuron_chain_risks = {
                key: torch.zeros(logits.shape[0], device=logits.device)
                for key in neuron_risk_default_metrics(load_neuron_risk_policy())
            }
        top_k_probs_cpu = top_k_probs.detach().cpu().tolist()
        top_k_indices_cpu = top_k_indices.detach().cpu().tolist()
        top_k_log_probs_cpu = top_k_log_probs.detach().cpu().tolist()
        top_k_logits_cpu = top_k_logits.detach().cpu().tolist()
        margins_cpu = margins.detach().cpu().tolist()
        entropies_cpu = entropies.detach().cpu().tolist()
        max_softmax_probabilities_cpu = (
            max_softmax_probabilities.detach().cpu().tolist()
        )
        energy_scores_cpu = energy_scores.detach().cpu().tolist()
        sampling_disagreement_probabilities_cpu = (
            sampling_disagreement_probabilities.detach().cpu().tolist()
        )
        layer_disagreements_cpu = layer_disagreements.detach().cpu().tolist()
        representation_variances_cpu = representation_variances.detach().cpu().tolist()
        neuron_chain_risks_cpu = {
            key: value.detach().cpu().tolist()
            for (key, value) in neuron_chain_risks.items()
        }
        hidden_nonfinite_cpu = hidden_nonfinite_per_sample.detach().cpu().tolist()
        flat_top_k_ids = [token_id for row in top_k_indices_cpu for token_id in row]
        flat_top_k_tokens = self._decode_token_ids(flat_top_k_ids)
        results: list[dict[str, Any]] = []
        token_offset = 0
        for i, (indices_row, probs_row) in enumerate(
            zip(top_k_indices_cpu, top_k_probs_cpu)
        ):
            row_tokens = flat_top_k_tokens[
                token_offset : token_offset + len(indices_row)
            ]
            token_offset += len(indices_row)
            top_k_tokens = [
                (token, float(prob)) for (token, prob) in zip(row_tokens, probs_row)
            ]
            top_k_details = self._build_top_k_details(
                token_ids=[int(token_id) for token_id in indices_row],
                tokens=row_tokens,
                probs=[float(prob) for prob in probs_row],
                logprobs=[float(value) for value in top_k_log_probs_cpu[i]],
                logits=[float(value) for value in top_k_logits_cpu[i]],
            )
            raw_uncertainty = {
                "margin": float(margins_cpu[i]),
                "entropy": float(entropies_cpu[i]),
                "max_softmax_probability": float(max_softmax_probabilities_cpu[i]),
                "energy_score": float(energy_scores_cpu[i]),
                "sampling_disagreement_probability": float(
                    sampling_disagreement_probabilities_cpu[i]
                ),
                "layer_disagreement": float(layer_disagreements_cpu[i]),
                "representation_variance": float(representation_variances_cpu[i]),
            }
            for key, values in neuron_chain_risks_cpu.items():
                raw_uncertainty[key] = float(values[i])
            (uncertainty, diagnostics) = self._sanitize_uncertainty_dict(
                raw_uncertainty
            )
            diagnostics["nonfinite_hidden_states"] = bool(hidden_nonfinite_cpu[i])
            results.append(
                {
                    "top_k_tokens": top_k_tokens,
                    "top_k_details": top_k_details,
                    "uncertainty": uncertainty,
                    "uncertainty_diagnostics": diagnostics,
                    "_h_last_tensor": final_hidden[i].detach().cpu().float(),
                    "_input_ids_tensor": input_ids[i][attention_mask[i].bool()]
                    .detach()
                    .cpu(),
                    "_sampled_hidden_tensor": stacked[:, i, :].detach().cpu().float(),
                    "_sampled_hidden_layer_indices": list(sample_layers),
                }
            )
        return results

    def analyze_prefix_batch(
        self,
        prefixes: list[str],
        k: int = DEFAULT_TOP_K,
        sample_layers: list[int] | None = None,
        max_length: int = 512,
    ) -> list[dict[str, Any]]:
        """Run a batched forward pass for multiple prefixes.

        Tokenizes all prefixes with left-padding, runs one forward pass,
        then extracts per-sample logits, top-k tokens, layer hidden states,
        and per-layer top-k predictions from the last *non-padded* position.
        """
        if not prefixes:
            return []
        max_length = validate_max_context_length(
            max_length, context="feature extraction"
        )
        assert_tokenizer_context_policy(
            self.tokenizer, context="QwenCoderLLM.analyze_prefix_batch"
        )
        torch = self._torch
        fallback_id = None
        for tid in (self.tokenizer.bos_token_id, self.tokenizer.eos_token_id):
            if tid is not None:
                fallback_id = tid
                break
        sanitized = []
        for p in prefixes:
            if p == "":
                if fallback_id is not None:
                    sanitized.append(self.tokenizer.decode([fallback_id]))
                else:
                    sanitized.append("\n")
            else:
                sanitized.append(p)
        encoded = self.tokenizer(
            sanitized,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        seq_lengths = attention_mask.sum(dim=1) - 1
        use_last_position_only = bool(torch.all(attention_mask[:, -1] == 1).item())
        with torch.no_grad():
            outputs = self._forward_batched_model(
                input_ids,
                attention_mask,
                output_hidden_states=True,
                use_last_position_only=use_last_position_only,
            )
        sample_layers = self._resolve_sample_layers(sample_layers)
        batch_size = input_ids.shape[0]
        results: list[dict[str, Any]] = []
        for i in range(batch_size):
            sl = seq_lengths[i]
            if outputs.logits.shape[1] == 1:
                logits_i = outputs.logits[i, 0, :]
            else:
                logits_i = outputs.logits[i, sl, :]
            log_probs_i = torch.log_softmax(logits_i, dim=0)
            probs_i = log_probs_i.exp()
            top_k_val = min(k, probs_i.shape[0])
            (top_k_probs, top_k_indices) = torch.topk(probs_i, top_k_val)
            top_k_log_probs = torch.gather(log_probs_i, dim=0, index=top_k_indices)
            top_k_logits = torch.gather(logits_i, dim=0, index=top_k_indices)
            top_k_tokens: list[tuple[str, float]] = [
                (self.tokenizer.decode([idx.item()]), float(prob.item()))
                for (prob, idx) in zip(top_k_probs, top_k_indices)
            ]
            top_k_details = self._build_top_k_details(
                token_ids=[int(idx.item()) for idx in top_k_indices],
                tokens=[token for (token, _prob) in top_k_tokens],
                probs=[float(prob.item()) for prob in top_k_probs],
                logprobs=[float(value.item()) for value in top_k_log_probs],
                logits=[float(value.item()) for value in top_k_logits],
            )
            layer_states: dict[int, list[float]] = {}
            for layer_idx in sample_layers:
                if layer_idx < len(outputs.hidden_states):
                    if use_last_position_only:
                        hidden = outputs.hidden_states[layer_idx][i, -1, :]
                    else:
                        hidden = outputs.hidden_states[layer_idx][i, sl, :]
                    layer_states[layer_idx] = hidden.cpu().numpy().tolist()
            results.append(
                {
                    "logits": logits_i.cpu().numpy().tolist(),
                    "top_k_tokens": top_k_tokens,
                    "top_k_details": top_k_details,
                    "layer_hidden_states": layer_states,
                }
            )
        return results

    def get_top_k_tokens(
        self, prefix: str, k: int = DEFAULT_TOP_K
    ) -> list[tuple[str, float]]:
        """Get top-k predicted tokens with probabilities.

        Args:
            prefix: The code prefix.
            k: Number of top tokens to return.

        Returns:
            List of (token, probability) tuples.
        """
        logits = self.get_logits(prefix)
        logits_tensor = self._torch.tensor(logits)
        probs = self._torch.softmax(logits_tensor, dim=0)
        (top_k_probs, top_k_indices) = self._torch.topk(probs, k)
        result = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            token = self.tokenizer.decode([idx.item()])
            result.append((token, prob.item()))
        return result
