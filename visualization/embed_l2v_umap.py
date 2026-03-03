import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from llm2vec import LLM2Vec
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


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
    parser.add_argument("--enable_multiprocessing", action="store_true", help="Enable multi-GPU multiprocessing in LLM2Vec.encode.")

    parser.add_argument("--pca_n_components", type=int, default=50, help="PCA n_components before UMAP (set 0 to disable).")
    parser.add_argument("--pca_random_state", type=int, default=42, help="PCA random_state (used for randomized solver).")

    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min_dist", type=float, default=0.0, help="UMAP min_dist.")
    parser.add_argument("--random_state", type=int, default=42, help="UMAP random_state.")
    parser.add_argument("--point_size", type=int, default=10, help="Size of the points in the scatter plot.")

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

    if "caption" not in df.columns or "hierarchical_Level1_label" not in df.columns:
        logger.error("CSV must contain 'caption' and 'hierarchical_Level1_label' columns.")
        return

    df = df[["caption", "hierarchical_Level1_label"]].dropna()
    captions = df["caption"].astype(str).tolist()
    labels = df["hierarchical_Level1_label"].astype(str).tolist()

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
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="label",
        palette="turbo",
        s=args.point_size,
        alpha=0.7,
    )
    plt.title("UMAP Projection of Caption Embeddings (LLM2Vec)")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
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
