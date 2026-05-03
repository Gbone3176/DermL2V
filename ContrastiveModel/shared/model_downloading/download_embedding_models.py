#!/usr/bin/env python
"""Download embedding model snapshots for migration to another machine.

The default mode downloads full Hugging Face snapshots, including model weights,
tokenizers, configs, README files, and remote-code Python files such as
NV-Embed-v2's custom modeling/configuration modules.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


DEFAULT_OUTPUT_DIR = Path("./share/model_snapshots")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    note: str
    needs_trust_remote_code: bool = False


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="pubmedbert",
        repo_id="NeuML/pubmedbert-base-embeddings",
        note="PubMedBERT sentence embedding model; original pooling is mean pooling.",
    ),
    ModelSpec(
        key="bioclinicalbert",
        repo_id="emilyalsentzer/Bio_ClinicalBERT",
        note="BioClinicalBERT backbone; wrap with fixed pooling for embedding training.",
    ),
    ModelSpec(
        key="bmretriever7b",
        repo_id="BMRetriever/BMRetriever-7B",
        note="Biomedical retrieval LLM; later fine-tuning should use LoRA/QLoRA.",
    ),
    ModelSpec(
        key="qwen3embedding8b",
        repo_id="Qwen/Qwen3-Embedding-8B",
        note="Qwen3 embedding model; use a modern transformers environment such as qwen3.",
    ),
    ModelSpec(
        key="nvembedv2",
        repo_id="nvidia/NV-Embed-v2",
        note="NV-Embed-v2 custom embedding model; loading usually requires trust_remote_code=True.",
        needs_trust_remote_code=True,
    ),
)


CODE_AND_CONFIG_PATTERNS = (
    "*.py",
    "*.json",
    "*.md",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "*/config.json",
)


WEIGHT_PATTERNS = (
    "*.bin",
    "*.safetensors",
    "*.msgpack",
    "*.h5",
    "*.ckpt",
    "*.pt",
    "*.pth",
    "*.index.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where snapshots are stored. Default: ./share/model_snapshots",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Model keys or HF repo ids to download. Use 'all' for the five target models. "
            f"Known keys: {', '.join(spec.key for spec in MODEL_SPECS)}"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["full", "code-only", "weights-only"],
        default="full",
        help="Download full snapshots, remote-code/config files only, or weights only.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/tag/commit. Applies to all selected models.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, HF_TOKEN/HUGGINGFACE_HUB_TOKEN is used when present.",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="Resume partial downloads when supported by huggingface_hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without downloading files.",
    )
    parser.add_argument(
        "--manifest-name",
        default="download_manifest.json",
        help="Manifest filename written under --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = resolve_models(args.models)
    allow_patterns = patterns_for_mode(args.mode)
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print_download_plan(args, selected, allow_patterns)
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in selected:
        local_dir = args.output_dir / safe_dir_name(spec.repo_id)
        print(f"\n==> Downloading {spec.repo_id} -> {local_dir}")
        snapshot_path = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="model",
            revision=args.revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            token=token,
            resume_download=args.resume_download,
        )
        results.append(
            {
                "key": spec.key,
                "repo_id": spec.repo_id,
                "snapshot_path": snapshot_path,
                "local_dir": str(local_dir),
                "needs_trust_remote_code": spec.needs_trust_remote_code,
                "note": spec.note,
            }
        )
        print(f"Downloaded: {snapshot_path}")

    write_manifest(args.output_dir / args.manifest_name, args, results)


def resolve_models(requested: Iterable[str]) -> list[ModelSpec]:
    requested = list(requested)
    if requested == ["all"] or "all" in requested:
        return list(MODEL_SPECS)

    by_key = {spec.key.lower(): spec for spec in MODEL_SPECS}
    by_repo = {spec.repo_id.lower(): spec for spec in MODEL_SPECS}
    selected = []
    for item in requested:
        normalized = item.lower()
        if normalized in by_key:
            selected.append(by_key[normalized])
        elif normalized in by_repo:
            selected.append(by_repo[normalized])
        elif "/" in item:
            selected.append(ModelSpec(key=safe_dir_name(item), repo_id=item, note="User-specified repo id."))
        else:
            raise ValueError(f"Unknown model key or repo id: {item}")
    return selected


def patterns_for_mode(mode: str) -> tuple[str, ...] | None:
    if mode == "full":
        return None
    if mode == "code-only":
        return CODE_AND_CONFIG_PATTERNS
    if mode == "weights-only":
        return WEIGHT_PATTERNS
    raise ValueError(f"Unsupported mode: {mode}")


def print_download_plan(
    args: argparse.Namespace,
    selected: list[ModelSpec],
    allow_patterns: tuple[str, ...] | None,
) -> None:
    print("Download plan")
    print(f"- output_dir: {args.output_dir}")
    print(f"- mode: {args.mode}")
    print(f"- revision: {args.revision or 'default branch'}")
    print(f"- cache_dir: {args.cache_dir or 'huggingface default'}")
    print(f"- allow_patterns: {'full snapshot' if allow_patterns is None else ', '.join(allow_patterns)}")
    for spec in selected:
        trust_note = " requires trust_remote_code when loading" if spec.needs_trust_remote_code else ""
        print(f"- {spec.key}: {spec.repo_id}{trust_note}")


def write_manifest(path: Path, args: argparse.Namespace, results: list[dict]) -> None:
    payload = {
        "mode": args.mode,
        "revision": args.revision,
        "output_dir": str(args.output_dir),
        "models": results,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"\nWrote manifest: {path}")


def safe_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


if __name__ == "__main__":
    main()
