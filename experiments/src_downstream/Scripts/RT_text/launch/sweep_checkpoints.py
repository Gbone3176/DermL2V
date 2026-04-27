from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate RT-nonhomo checkpoint sweeps with clear upper/lower separation")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--method-key", required=True)
    parser.add_argument("--mode", choices=["full", "single-dataset"], default="full")
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--steps", nargs="*", type=int, default=[])
    parser.add_argument("--devices", default="")
    parser.add_argument("--max-gpus", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_steps(run_root: Path, requested_steps: list[int]) -> list[int]:
    available = []
    for path in sorted(run_root.glob("checkpoint-*")):
        try:
            available.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    if requested_steps:
        available = [step for step in available if step in set(requested_steps)]
    if not available:
        raise FileNotFoundError(f"No matching checkpoints under {run_root}")
    return available


def detect_free_gpus(max_memory_used_mb: int = 1024) -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    for line in out.strip().splitlines():
        gpu_str, mem_str = [part.strip() for part in line.split(",")]
        if int(mem_str) <= max_memory_used_mb:
            free.append(int(gpu_str))
    return free


def selected_gpus(devices_arg: str, max_gpus: int) -> list[int]:
    if devices_arg.strip():
        return [int(part.strip()) for part in devices_arg.split(",") if part.strip()]
    return detect_free_gpus()[:max_gpus]


def lower_script(mode: str) -> str:
    if mode == "full":
        return "run_full_checkpoint.py"
    return "run_single_checkpoint.py"


def build_lower_cmd(base_dir: Path, config_path: Path, method_key: str, checkpoint_dir: Path, batch_size: int, device: int, mode: str, dataset_file: str | None) -> list[str]:
    cfg = load_json(config_path)
    cmd = [
        cfg.get("python_bin", "python"),
        str(base_dir / lower_script(mode)),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--config-path",
        str(config_path),
        "--method-key",
        method_key,
        "--batch-size",
        str(batch_size),
        "--device",
        str(device),
    ]
    if mode == "single-dataset":
        if not dataset_file:
            raise ValueError("--dataset-file is required for single-dataset mode")
        cmd.extend(["--dataset-file", dataset_file])
    return cmd


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path)
    cfg = load_json(config_path)
    method = dict(cfg["methods"][args.method_key])
    run_root = Path(method["run_root"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 64))
    steps = checkpoint_steps(run_root, args.steps)
    gpu_ids = selected_gpus(args.devices, args.max_gpus)
    if not gpu_ids:
        raise RuntimeError("No GPU available for sweep")

    base_dir = Path(__file__).resolve().parent
    pending = [run_root / f"checkpoint-{step}" for step in steps]

    for start in range(0, len(pending), len(gpu_ids)):
        batch = pending[start : start + len(gpu_ids)]
        processes: list[tuple[Path, int, subprocess.Popen[str]]] = []
        for gpu_id, checkpoint_dir in zip(gpu_ids, batch):
            cmd = build_lower_cmd(
                base_dir=base_dir,
                config_path=config_path,
                method_key=args.method_key,
                checkpoint_dir=checkpoint_dir,
                batch_size=batch_size,
                device=gpu_id,
                mode=args.mode,
                dataset_file=args.dataset_file,
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(cmd, cwd=cfg.get("repo_root"), env=env, text=True)
            processes.append((checkpoint_dir, gpu_id, proc))

        failures = []
        for checkpoint_dir, gpu_id, proc in processes:
            return_code = proc.wait()
            if return_code != 0:
                failures.append((str(checkpoint_dir), gpu_id, return_code))
        if failures:
            raise RuntimeError(f"Sweep failed: {failures}")


if __name__ == "__main__":
    main()
