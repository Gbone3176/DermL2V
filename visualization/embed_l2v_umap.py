import argparse
import logging
import os
import sys

VENDOR_DIR = os.path.join(os.path.dirname(__file__), ".vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.append(VENDOR_DIR)

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from llm2vec import LLM2Vec
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from visualization.plot_utils import draw_kde_envelopes


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


def main():
    parser = argparse.ArgumentParser(description="Encode captions using LLM2Vec, reduce with UMAP, and visualize.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_folder", type=str, default="umap_outputs", help="Folder to save the UMAP plot.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the LLM2Vec base model.")
    parser.add_argument("--peft_model_name_or_path", type=str, default=None, help="Path to the PEFT model.")
    parser.add_argument("--extra_model_name_or_path", type=str, nargs="+", default=None, help="Path to extra PEFT models (list).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--instruction", type=str, default="", help="Instruction to prepend to text (optional).")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    parser.add_argument("--label_column", type=str, default=None, help="Optional label column override.")
    parser.add_argument("--enable_multiprocessing", action="store_true", help="Enable multi-GPU multiprocessing in LLM2Vec.encode.")

    parser.add_argument("--pca_n_components", type=int, default=50, help="PCA n_components before UMAP (set 0 to disable).")
    parser.add_argument("--pca_random_state", type=int, default=42, help="PCA random_state (used for randomized solver).")

    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min_dist", type=float, default=0.0, help="UMAP min_dist.")
    parser.add_argument("--random_state", type=int, default=42, help="UMAP random_state.")
    parser.add_argument("--point_size", type=int, default=10, help="Size of the points in the scatter plot.")
    parser.add_argument("--disable_kde_envelope", action="store_true", help="Disable KDE envelopes behind each class cluster.")
    parser.add_argument("--kde_alpha", type=float, default=0.18, help="Alpha used for KDE envelopes.")
    parser.add_argument("--kde_levels", type=int, default=1, help="Number of filled KDE levels.")
    parser.add_argument("--kde_thresh", type=float, default=0.25, help="Density threshold for KDE envelopes.")
    parser.add_argument("--kde_bw_adjust", type=float, default=0.9, help="Bandwidth adjustment for KDE envelopes.")
    parser.add_argument("--kde_min_points", type=int, default=25, help="Minimum points required in a local cluster to draw a KDE envelope.")
    parser.add_argument("--kde_cluster_eps", type=float, default=None, help="Optional DBSCAN eps for local cluster filtering before KDE.")
    parser.add_argument("--kde_cluster_min_samples", type=int, default=8, help="DBSCAN min_samples used to suppress tiny outlier groups.")

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    logger.info(f"Loading data from {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        return

    label_column = infer_label_column(args.input_file, args.label_column)
    logger.info(f"Using label column: {label_column}")

    if "caption" not in df.columns or label_column not in df.columns:
        logger.error(f"CSV must contain 'caption' and '{label_column}' columns.")
        return

    df = df[["caption", label_column]].dropna()
    captions = df["caption"].astype(str).tolist()
    labels = df[label_column].astype(str).tolist()

    if len(captions) == 0:
        logger.error("No valid rows found after dropping NaNs.")
        return

    logger.info(f"Loading LLM2Vec model from {args.model_name_or_path}...")
    llm2vec_model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        extra_model_name_or_path=args.extra_model_name_or_path,
        merge_peft=True,
        pooling_mode="mean",
        max_length=args.max_length,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    if args.instruction:
        encode_inputs = [[args.instruction, c] for c in captions]
    else:
        encode_inputs = captions

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
    if embeddings.ndim != 2 or embeddings.shape[0] != len(captions):
        logger.error(f"Unexpected embeddings shape: {embeddings.shape}")
        return

    pca_components_used = 0
    if args.pca_n_components and args.pca_n_components > 0:
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]
        pca_components_used = min(args.pca_n_components, n_samples, n_features)
        logger.info(f"Running PCA (n_components={pca_components_used})...")
        pca = PCA(n_components=pca_components_used, random_state=args.pca_random_state)
        embeddings = pca.fit_transform(embeddings).astype(np.float32, copy=False)

    logger.info("Running UMAP...")
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        logger.warning("Not enough samples for UMAP (need at least 2). Skipping UMAP.")
        return

    n_neighbors = min(max(2, args.n_neighbors), n_samples - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        metric="cosine"
    )
    embedding_2d = reducer.fit_transform(embeddings)

    logger.info("Plotting...")
    plot_df = pd.DataFrame(
        {
            "x": embedding_2d[:, 0],
            "y": embedding_2d[:, 1],
            "label": labels,
        }
    )

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    unique_labels = pd.unique(plot_df["label"])
    palette = dict(zip(unique_labels, sns.color_palette("turbo", n_colors=len(unique_labels))))

    draw_kde_envelopes(
        ax=ax,
        plot_df=plot_df,
        palette=palette,
        enabled=not args.disable_kde_envelope,
        alpha=args.kde_alpha,
        levels=args.kde_levels,
        thresh=args.kde_thresh,
        bw_adjust=args.kde_bw_adjust,
        min_points=args.kde_min_points,
        cluster_eps=args.kde_cluster_eps,
        cluster_min_samples=args.kde_cluster_min_samples,
    )

    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="label",
        palette=palette,
        s=args.point_size,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    plt.axis("off")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()

    os.makedirs(args.output_folder, exist_ok=True)

    pca_part = f"_pca{pca_components_used}" if pca_components_used > 0 else ""
    filename = (
        f"umap_nc2"
        f"_nn{n_neighbors}"
        f"_md{_float_for_filename(args.min_dist)}"
        f"_rs{args.random_state}"
        f"{pca_part}"
        f"_bs{args.batch_size}"
        ".png"
    )
    output_path = os.path.join(args.output_folder, filename)
    plt.savefig(output_path, dpi=300)
    logger.info(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
