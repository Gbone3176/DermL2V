from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
import sys
from typing import Dict, List

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
sys.path.insert(0, str(LIB_DIR))

from checkpoint_common import DEFAULT_CONFIG_PATHS, dataset_file_paths, resolve_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the 4-dataset RT-nonhomo full suite for one SA_Fusion checkpoint"
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--method-key", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--config-path", action="append", default=[])
    return parser.parse_args()


def extend_structured_selfattn_args(cmd: List[str], method: Dict[str, object]) -> None:
    if method["pooling_mode"] not in {"structured_selfattn", "structured_selfattn_fusion"}:
        return
    cmd.extend(
        [
            "--selfattn_attn_hidden_dim",
            str(method.get("selfattn_attn_hidden_dim", 512)),
            "--selfattn_num_hops",
            str(method.get("selfattn_num_hops", 8)),
            "--selfattn_output_dropout",
            str(method.get("selfattn_output_dropout", 0.0)),
        ]
    )
    if "selfattn_output_norm" in method:
        cmd.extend(["--selfattn_output_norm", str(method["selfattn_output_norm"])])
    if "selfattn_gamma_init" in method:
        cmd.extend(["--selfattn_gamma_init", str(method["selfattn_gamma_init"])])
    if "selfattn_gamma_learnable" in method:
        cmd.extend(
            [
                "--selfattn_gamma_learnable",
                str(method["selfattn_gamma_learnable"]).lower(),
            ]
        )
    if "selfattn_merge_mode" in method:
        cmd.extend(["--selfattn_merge_mode", str(method["selfattn_merge_mode"])])
    if "selfattn_merge_temperature" in method:
        cmd.extend(
            [
                "--selfattn_merge_temperature",
                str(method["selfattn_merge_temperature"]),
            ]
        )
    if "selfattn_merge_hidden_dim" in method:
        cmd.extend(
            ["--selfattn_merge_hidden_dim", str(method["selfattn_merge_hidden_dim"])]
        )
    if "selfattn_merge_input_norm" in method:
        cmd.extend(
            ["--selfattn_merge_input_norm", str(method["selfattn_merge_input_norm"])]
        )

def main() -> None:
    args = parse_args()
    config_paths = [Path(p) for p in args.config_path] if args.config_path else DEFAULT_CONFIG_PATHS
    resolved = resolve_checkpoint(
        Path(args.checkpoint_dir), method_key=args.method_key, config_paths=config_paths
    )
    if resolved.method["pooling_mode"] != "structured_selfattn_fusion":
        raise ValueError(
            f"Resolved method {resolved.method_key} is not SA_Fusion: "
            f"{resolved.method['pooling_mode']}"
        )
    resolved.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        resolved.config["python_bin"],
        "-m",
        "experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_l2v_multi_full",
        "--instruction",
        resolved.config["instruction_rt"],
        "--model_name",
        resolved.model_name,
        "--pooling_mode",
        resolved.method["pooling_mode"],
        "--max_length",
        "512",
        "--batch_size",
        str(args.batch_size),
        "--enable_bidirectional",
        "True",
        "--base_model_name_or_path",
        resolved.config["base_model_name_or_path"],
        "--peft_model_name_or_path",
        resolved.config["peft_model_name_or_path"],
        "--extra_model_name_or_path",
        resolved.config["supervised_model_name_or_path"],
        str(resolved.checkpoint_dir),
        "--output",
        str(resolved.checkpoint_output_dir),
    ]
    extend_structured_selfattn_args(cmd, resolved.method)
    for dataset_file in dataset_file_paths(resolved.config):
        cmd.extend(["--dataset_file_path", str(dataset_file)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, env=env, cwd=resolved.config.get("repo_root"))


if __name__ == "__main__":
    main()
