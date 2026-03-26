import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from llm2vec import LLM2Vec
from sklearn.decomposition import PCA


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOSNES_ROOT = os.path.join(CURRENT_DIR, "DOSNES")
if DOSNES_ROOT not in sys.path:
    sys.path.insert(0, DOSNES_ROOT)

from dosnes import dosnes  # noqa: E402


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _float_for_filename(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if s else "0"


def infer_label_column(input_file: str, label_column: str | None) -> str:
    if label_column:
        return label_column
    lowered = input_file.lower()
    if "/l4/" in lowered or "level4" in lowered:
        return "hierarchical_Level4_label"
    return "hierarchical_Level1_label"


def save_plot(embedding_3d, labels, output_path, title, point_size):
    label_codes, uniques = pd.factorize(np.asarray(labels))
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        embedding_3d[:, 0],
        embedding_3d[:, 1],
        embedding_3d[:, 2],
        c=label_codes,
        cmap="turbo",
        s=point_size,
        alpha=0.8,
    )
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [str(x) for x in uniques], loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Encode captions using LLM2Vec, reduce with DOSNES, and visualize.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="dosnes_outputs")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--peft_model_name_or_path", type=str, default=None)
    parser.add_argument("--extra_model_name_or_path", type=str, nargs="+", default=None)
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--enable_multiprocessing", action="store_true")
    parser.add_argument("--point_size", type=int, default=10)
    parser.add_argument("--pca_n_components", type=int, default=50)
    parser.add_argument("--pca_random_state", type=int, default=42)
    parser.add_argument("--pooling_mode", type=str, default="mean")
    parser.add_argument("--dosnes_metric", type=str, default="cosine")
    parser.add_argument("--dosnes_max_iter", type=int, default=1000)
    parser.add_argument("--dosnes_learning_rate", type=float, default=500.0)
    parser.add_argument("--dosnes_momentum", type=float, default=0.5)
    parser.add_argument("--dosnes_final_momentum", type=float, default=0.3)
    parser.add_argument("--dosnes_mom_switch_iter", type=int, default=250)
    parser.add_argument("--dosnes_min_gain", type=float, default=0.01)
    parser.add_argument("--dosnes_random_state", type=int, default=42)
    parser.add_argument("--dosnes_verbose", type=int, default=0)
    parser.add_argument("--dosnes_verbose_freq", type=int, default=10)
    parser.add_argument("--render_training", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    label_column = infer_label_column(args.input_file, args.label_column)
    logger.info(f"Using device: {device}")
    logger.info(f"Using label column: {label_column}")

    df = pd.read_csv(args.input_file)
    if "caption" not in df.columns or label_column not in df.columns:
        raise ValueError(f"CSV must contain 'caption' and '{label_column}' columns.")
    df = df[["caption", label_column]].dropna()
    captions = df["caption"].astype(str).tolist()
    labels = df[label_column].astype(str).tolist()

    llm2vec_model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        extra_model_name_or_path=args.extra_model_name_or_path,
        merge_peft=True,
        pooling_mode=args.pooling_mode,
        max_length=args.max_length,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    encode_inputs = [[args.instruction, c] for c in captions] if args.instruction else captions
    logger.info("Encoding captions...")
    if args.enable_multiprocessing:
        embeddings = llm2vec_model.encode(
            encode_inputs,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=device,
        )
    else:
        orig_device_count = torch.cuda.device_count
        try:
            torch.cuda.device_count = lambda: 1  # type: ignore
            embeddings = llm2vec_model.encode(
                encode_inputs,
                batch_size=args.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=device,
            )
        finally:
            torch.cuda.device_count = orig_device_count  # type: ignore

    embeddings = np.asarray(embeddings).astype(np.float32, copy=False)

    pca_components_used = 0
    if args.pca_n_components and args.pca_n_components > 0:
        n_samples, n_features = embeddings.shape
        pca_components_used = min(args.pca_n_components, n_samples, n_features)
        logger.info(f"Running PCA (n_components={pca_components_used})...")
        pca = PCA(n_components=pca_components_used, random_state=args.pca_random_state)
        embeddings = pca.fit_transform(embeddings).astype(np.float32, copy=False)

    logger.info("Running DOSNES...")
    model_dosnes = dosnes.DOSNES(
        metric=args.dosnes_metric,
        max_iter=args.dosnes_max_iter,
        learning_rate=args.dosnes_learning_rate,
        momentum=args.dosnes_momentum,
        final_momentum=args.dosnes_final_momentum,
        mom_switch_iter=args.dosnes_mom_switch_iter,
        min_gain=args.dosnes_min_gain,
        verbose=args.dosnes_verbose,
        verbose_freq=args.dosnes_verbose_freq,
        random_state=args.dosnes_random_state,
        render_training=args.render_training,
    )

    os.makedirs(args.output_folder, exist_ok=True)
    pca_part = f"_pca{pca_components_used}" if pca_components_used > 0 else ""
    run_part = f"_run{args.run_name}" if args.run_name else ""
    stem = (
        f"dosnes_metric{args.dosnes_metric}"
        f"_iter{args.dosnes_max_iter}"
        f"_lr{_float_for_filename(args.dosnes_learning_rate)}"
        f"_rs{args.dosnes_random_state}"
        f"{pca_part}"
        f"_bs{args.batch_size}"
        f"{run_part}"
        f"_pool{args.pooling_mode}"
    )
    gif_path = os.path.join(args.output_folder, f"{stem}.gif") if args.render_training else None
    embedding_3d = model_dosnes.fit_transform(embeddings, y=pd.factorize(np.asarray(labels))[0], filename=gif_path)

    plot_path = os.path.join(args.output_folder, f"{stem}.png")
    coords_path = os.path.join(args.output_folder, f"{stem}.csv")
    cost_path = os.path.join(args.output_folder, f"{stem}_cost.json")

    save_plot(
        embedding_3d=embedding_3d,
        labels=labels,
        output_path=plot_path,
        title="DOSNES Projection of Caption Embeddings (LLM2Vec)",
        point_size=args.point_size,
    )
    pd.DataFrame(
        {
            "x": embedding_3d[:, 0],
            "y": embedding_3d[:, 1],
            "z": embedding_3d[:, 2],
            "label": labels,
            "caption": captions,
        }
    ).to_csv(coords_path, index=False)
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_file": args.input_file,
                "label_column": label_column,
                "model_name_or_path": args.model_name_or_path,
                "peft_model_name_or_path": args.peft_model_name_or_path,
                "extra_model_name_or_path": args.extra_model_name_or_path,
                "pooling_mode": args.pooling_mode,
                "instruction": args.instruction,
                "pca_n_components": pca_components_used,
                "dosnes_metric": args.dosnes_metric,
                "dosnes_max_iter": args.dosnes_max_iter,
                "dosnes_learning_rate": args.dosnes_learning_rate,
                "dosnes_random_state": args.dosnes_random_state,
                "cost": model_dosnes.cost,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"Plot saved to {plot_path}")
    logger.info(f"Coordinates saved to {coords_path}")
    logger.info(f"Training cost saved to {cost_path}")


if __name__ == "__main__":
    main()
