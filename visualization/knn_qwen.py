import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def calculate_neighborhood_purity(embeddings, labels, k_values=None):
    if k_values is None:
        k_values = [10, 30]
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        logger.warning("Not enough samples to calculate Neighborhood Purity.")
        return {k: 0.0 for k in k_values}

    labels = np.array(labels)
    unique_labels = np.unique(labels)
    logger.info(f"Number of unique labels: {len(unique_labels)}")

    max_k = max(k_values)
    n_neighbors_fit = min(n_samples, max_k + 1)

    logger.info(f"Fitting NearestNeighbors with max_k={max_k}...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors_fit, algorithm="auto", metric="cosine", n_jobs=-1)
    nbrs.fit(embeddings)

    _, indices = nbrs.kneighbors(embeddings)

    results = {}
    for k in k_values:
        if k >= n_samples:
            logger.warning(f"k={k} is larger than n_samples={n_samples}. Skipping.")
            results[k] = None
            continue

        logger.info(f"Calculating Purity for k={k}...")
        current_indices = indices[:, 1 : k + 1]
        neighbor_labels = labels[current_indices]
        true_labels = labels.reshape(-1, 1)
        matches = neighbor_labels == true_labels
        point_purities = np.mean(matches, axis=1)
        avg_purity = np.mean(point_purities)
        results[k] = avg_purity
        logger.info(f"Neighborhood Purity (k={k}): {avg_purity:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Encode captions using Qwen3-Embedding and calculate Neighborhood Purity.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Path to the Qwen3-Embedding model.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save JSON results.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for encoding.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation for Qwen3.",
    )
    parser.add_argument("--prompt_name", type=str, default="query", help="Prompt name for Qwen3 embedding.")

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

    if "caption" not in df.columns or "hierarchical_Level4_label" not in df.columns:
        logger.error("CSV must contain 'caption' and 'hierarchical_Level4_label' columns.")
        return

    df = df[["caption", "hierarchical_Level4_label"]].dropna()

    captions = df["caption"].tolist()
    labels = df["hierarchical_Level4_label"].astype(str).tolist()

    if len(captions) == 0:
        logger.error("No valid data found.")
        return

    logger.info(f"Loading model from {args.model_name_or_path}...")
    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    tokenizer_kwargs = {"padding_side": "left", "trust_remote_code": True}
    try:
        model = SentenceTransformer(
            args.model_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
            trust_remote_code=True,
        )
    except TypeError:
        model = SentenceTransformer(
            args.model_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
        )

    model.max_seq_length = args.max_length

    logger.info("Encoding captions...")
    embeddings_list = []
    for i in tqdm(range(0, len(captions), args.batch_size), desc="Encoding"):
        batch_texts = captions[i : i + args.batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        embeddings_list.append(batch_embeddings.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)

    logger.info("Calculating Neighborhood Purity...")
    purity_scores = calculate_neighborhood_purity(embeddings, labels, k_values=[10, 30])
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_basename = os.path.basename(args.model_name_or_path.rstrip("/"))
        output_filename = f"{model_basename}.json" if model_basename else "knn_purity.json"
        output_path = os.path.join(args.output_dir, output_filename)
        result_payload = {
            "model_name_or_path": args.model_name_or_path,
            "num_samples": len(labels),
            "k_values": [10, 30],
            "purity_scores": {str(k): v for k, v in purity_scores.items()},
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results to {output_path}")

    print("\n" + "=" * 40)
    print("Neighborhood Purity Results:")
    print("=" * 40)
    for k, score in purity_scores.items():
        if score is not None:
            print(f"k={k}: {score:.4f}")
    print("=" * 40 + "\n")

    print("Interpretation Hint:")
    print(" - ~0.1 (random baseline for 10 balanced classes): High overlap.")
    print(" - >> 0.1: Natural clustering/separation exists.")


if __name__ == "__main__":
    main()
