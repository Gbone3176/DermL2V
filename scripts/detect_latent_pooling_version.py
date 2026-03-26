#!/usr/bin/env python3
"""
Detect which local latent pooling implementation best matches a checkpoint.

Usage:
  python scripts/detect_latent_pooling_version.py /path/to/latent_attn.pt
  python scripts/detect_latent_pooling_version.py /path/to/model_dir
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Candidate:
    name: str
    file_path: Path


@dataclass
class MatchResult:
    name: str
    score: float
    key_f1: float
    shape_acc: float
    n_ckpt: int
    n_model: int
    n_common: int
    n_shape_match: int
    n_shape_mismatch: int
    n_missing: int
    n_unexpected: int
    missing_examples: List[str]
    unexpected_examples: List[str]
    mismatch_examples: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the most likely latent_pooling code version for a latent_attn checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to latent_attn.pt / latent_attn.safetensors, or a directory containing one of them.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top candidates to print (default: 5).",
    )
    return parser.parse_args()


def resolve_checkpoint_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        st_path = path / "latent_attn.safetensors"
        pt_path = path / "latent_attn.pt"
        if st_path.exists():
            return st_path
        if pt_path.exists():
            return pt_path
        raise FileNotFoundError(
            f"Directory {path} does not contain latent_attn.safetensors or latent_attn.pt"
        )
    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Need safetensors installed to read .safetensors checkpoints."
            ) from exc
        raw_obj = load_file(str(path))
    else:
        raw_obj = torch.load(str(path), map_location="cpu")

    state = extract_tensor_state_dict(raw_obj)
    if not state:
        raise ValueError(f"Could not find tensor state_dict in checkpoint: {path}")
    state = normalize_state_dict_keys(state)
    if not state:
        raise ValueError(f"No usable latent_attn keys after normalization: {path}")
    return state


def extract_tensor_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()

    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()):
            return obj

        for key in ("state_dict", "model_state_dict", "model", "latent_attn"):
            if key in obj and isinstance(obj[key], dict):
                sub = extract_tensor_state_dict(obj[key])
                if sub:
                    return sub

        tensor_items = {k: v for k, v in obj.items() if isinstance(k, str) and torch.is_tensor(v)}
        if tensor_items:
            return tensor_items

    return {}


def normalize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state.keys())
    latent_related = [
        k for k in keys if k.startswith("latent_attn.") or ".latent_attn." in k
    ]
    if latent_related:
        filtered = {}
        for k in latent_related:
            idx = k.find("latent_attn.")
            new_k = k[idx + len("latent_attn.") :]
            filtered[new_k] = state[k]
        state = filtered

    state = strip_uniform_prefix(state, "module.")
    state = strip_uniform_prefix(state, "model.")
    return state


def strip_uniform_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if state and all(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix) :]: v for k, v in state.items()}
    return state


def infer_dims(state: Dict[str, torch.Tensor]) -> Tuple[int, int, Optional[int]]:
    if "latents" not in state:
        raise KeyError("Checkpoint is missing required key 'latents'.")
    latents = state["latents"]
    if latents.ndim != 2:
        raise ValueError(f"Expected 'latents' to be 2D, got shape {tuple(latents.shape)}")
    num_latents, d_model = int(latents.shape[0]), int(latents.shape[1])

    inner = None
    q_key = "cross_attn.fn.to_q.weight"
    if q_key in state and state[q_key].ndim == 2:
        inner = int(state[q_key].shape[0])

    return d_model, num_latents, inner


def load_latent_class(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "LatentAttentionPooling"):
        raise AttributeError(f"{module_path} does not define LatentAttentionPooling")
    return mod.LatentAttentionPooling


def instantiate_state_dict(
    candidate: Candidate, d_model: int, num_latents: int, inner_dim: Optional[int]
) -> Dict[str, torch.Tensor]:
    latent_cls = load_latent_class(candidate.file_path)

    kwargs = {"d_model": d_model, "num_latents": num_latents}
    name = candidate.name.lower()

    if "v0" in name or "v1" in name:
        kwargs["num_heads"] = 1
    else:
        kwargs["num_heads"] = 1
        if inner_dim is not None:
            kwargs["head_dim"] = inner_dim

    model = latent_cls(**kwargs)
    return model.state_dict()


def f1_overlap(set_a: Iterable[str], set_b: Iterable[str]) -> float:
    a = set(set_a)
    b = set(set_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = len(a & b)
    return 2.0 * common / (len(a) + len(b))


def compare_state_dicts(
    ckpt: Dict[str, torch.Tensor], model: Dict[str, torch.Tensor], name: str
) -> MatchResult:
    ckpt_keys = set(ckpt.keys())
    model_keys = set(model.keys())
    common = sorted(ckpt_keys & model_keys)
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    shape_match = []
    shape_mismatch = []
    for k in common:
        if tuple(ckpt[k].shape) == tuple(model[k].shape):
            shape_match.append(k)
        else:
            shape_mismatch.append(k)

    key_f1 = f1_overlap(ckpt_keys, model_keys)
    shape_acc = len(shape_match) / max(1, len(common))
    mismatch_penalty = 0.02 * len(shape_mismatch)
    score = 0.6 * key_f1 + 0.4 * shape_acc - mismatch_penalty

    return MatchResult(
        name=name,
        score=score,
        key_f1=key_f1,
        shape_acc=shape_acc,
        n_ckpt=len(ckpt_keys),
        n_model=len(model_keys),
        n_common=len(common),
        n_shape_match=len(shape_match),
        n_shape_mismatch=len(shape_mismatch),
        n_missing=len(missing),
        n_unexpected=len(unexpected),
        missing_examples=missing[:5],
        unexpected_examples=unexpected[:5],
        mismatch_examples=shape_mismatch[:5],
    )


def build_candidates() -> List[Candidate]:
    base = ROOT / "llm2vec"
    return [
        Candidate("V0", base / "pooling_latent_V0.py"),
        Candidate("V1", base / "pooling_latent_V1.py"),
        Candidate("V2", base / "pooling_latent_V2.py"),
        Candidate("V3", base / "pooling_latent_V3.py"),
        Candidate("current(pooling_latent.py)", base / "pooling_latent.py"),
    ]


def main() -> None:
    args = parse_args()
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt_sd = load_checkpoint(ckpt_path)
    d_model, num_latents, inner_dim = infer_dims(ckpt_sd)

    candidates = build_candidates()
    results: List[MatchResult] = []

    for c in candidates:
        try:
            model_sd = instantiate_state_dict(c, d_model, num_latents, inner_dim)
            results.append(compare_state_dicts(ckpt_sd, model_sd, c.name))
        except Exception as exc:
            results.append(
                MatchResult(
                    name=c.name,
                    score=float("-inf"),
                    key_f1=0.0,
                    shape_acc=0.0,
                    n_ckpt=len(ckpt_sd),
                    n_model=0,
                    n_common=0,
                    n_shape_match=0,
                    n_shape_mismatch=0,
                    n_missing=0,
                    n_unexpected=0,
                    missing_examples=[],
                    unexpected_examples=[],
                    mismatch_examples=[f"instantiate error: {exc}"],
                )
            )

    results.sort(key=lambda x: (x.score, x.n_shape_match, x.n_common), reverse=True)
    top_k = max(1, min(args.top_k, len(results)))

    print(f"Checkpoint: {ckpt_path}")
    print(f"Inferred dims: d_model={d_model}, num_latents={num_latents}, inner_dim={inner_dim}")
    print("")
    print("Ranking:")
    for idx, r in enumerate(results[:top_k], start=1):
        print(
            f"{idx}. {r.name}: score={r.score:.4f}, key_f1={r.key_f1:.4f}, "
            f"shape_acc={r.shape_acc:.4f}, common={r.n_common}, "
            f"shape_match={r.n_shape_match}, shape_mismatch={r.n_shape_mismatch}, "
            f"missing={r.n_missing}, unexpected={r.n_unexpected}"
        )
        if r.mismatch_examples:
            print(f"   mismatch_examples: {', '.join(r.mismatch_examples)}")
        if r.missing_examples:
            print(f"   missing_examples: {', '.join(r.missing_examples)}")
        if r.unexpected_examples:
            print(f"   unexpected_examples: {', '.join(r.unexpected_examples)}")

    print("")
    best = results[0]
    ties = [
        r for r in results
        if abs(r.score - best.score) < 1e-9 and r.n_shape_match == best.n_shape_match and r.n_common == best.n_common
    ]
    if len(ties) > 1:
        tied_names = ", ".join(t.name for t in ties)
        print(f"Result: ambiguous best match among [{tied_names}]")
        print("Reason: key and shape evidence are identical for these candidates.")
    else:
        print(f"Result: most likely = {best.name}")


if __name__ == "__main__":
    main()
