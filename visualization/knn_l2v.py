import argparse
import json
import logging
import sys
import os

import numpy as np
import pandas as pd
import torch
from llm2vec import LLM2Vec
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def calculate_neighborhood_purity(embeddings, labels, k_values=[10, 30]):
    """
    Calculate kNN Neighborhood Purity for given k values.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        labels: list of labels corresponding to each sample
        k_values: list of integers, values of k for kNN
        
    Returns:
        dict: {k: purity_score}
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        logger.warning("Not enough samples to calculate Neighborhood Purity.")
        return {k: 0.0 for k in k_values}

    labels = np.array(labels)
    unique_labels = np.unique(labels)
    logger.info(f"Number of unique labels: {len(unique_labels)}")
    
    # Fit Nearest Neighbors
    # Use max k to fit once
    max_k = max(k_values)
    # We need k+1 neighbors because the first one is the point itself
    n_neighbors_fit = min(n_samples, max_k + 1)
    
    logger.info(f"Fitting NearestNeighbors with max_k={max_k}...")
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors_fit, algorithm='auto', metric='cosine', n_jobs=-1)
    nbrs.fit(embeddings)
    
    distances, indices = nbrs.kneighbors(embeddings)
    
    results = {}
    
    for k in k_values:
        if k >= n_samples:
            logger.warning(f"k={k} is larger than n_samples={n_samples}. Skipping.")
            results[k] = None
            continue
            
        logger.info(f"Calculating Purity for k={k}...")
        
        # indices has shape (n_samples, n_neighbors_fit)
        # We take columns 1 to k+1 (skipping column 0 which is the point itself)
        
        current_indices = indices[:, 1:k+1] # shape (n_samples, k)
        
        # Get labels of neighbors
        neighbor_labels = labels[current_indices] # shape (n_samples, k)
        
        # Get true labels of query points
        true_labels = labels.reshape(-1, 1) # shape (n_samples, 1)
        
        # Check matches
        matches = (neighbor_labels == true_labels) # shape (n_samples, k) boolean
        
        # Calculate purity per point: fraction of neighbors with same label
        point_purities = np.mean(matches, axis=1) # shape (n_samples,)
        
        # Average over all points
        avg_purity = np.mean(point_purities)
        
        results[k] = avg_purity
        logger.info(f"Neighborhood Purity (k={k}): {avg_purity:.4f}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Encode captions using LLM2Vec and calculate Neighborhood Purity.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the LLM2Vec base model.")
    parser.add_argument("--peft_model_name_or_path", type=str, default=None, help="Path to the PEFT model.")
    parser.add_argument("--extra_model_name_or_path", type=str, nargs="+", default=None, help="Path to extra PEFT models (list).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument(
        "--pooling_mode",
        type=str,
        default="mean",
        choices=["mean", "weighted_mean", "eos_token", "last_token", "bos_token", "latent_pooling"],
        help="Pooling strategy for sentence embeddings.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save JSON results. If a directory, a filename will be generated.",
    )
    parser.add_argument("--instruction", type=str, default="", help="Instruction to prepend to text (optional).")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    parser.add_argument("--enable_multiprocessing", action="store_true", help="Enable multi-GPU multiprocessing in LLM2Vec.encode.")
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Column name to use as labels for purity calculation. If None, inferred from input file path.",
    )
    
    args = parser.parse_args()

    if args.label_column is None:
        if "/L4/" in args.input_file or "level4" in args.input_file.lower():
            args.label_column = "hierarchical_Level4_label"
        else:
            args.label_column = "hierarchical_Level1_label"
    
    logger.info(f"Using label column: {args.label_column}")

    # Determine output path early to check if it exists
    if args.output_file is None:
        base = os.path.splitext(os.path.basename(args.input_file))[0]
        output_path = os.path.join(os.getcwd(), f"{base}_knn_purity.json")
    else:
        output_path = args.output_file
        if os.path.isdir(output_path) or output_path.endswith(os.sep):
            base = os.path.splitext(os.path.basename(args.input_file))[0]
            output_path = os.path.join(output_path, f"{base}_knn_purity.json")

    if os.path.exists(output_path):
        logger.info(f"Output file {output_path} already exists. Exiting.")
        return

    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")

    # Load Data
    logger.info(f"Loading data from {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        return

    if "caption" not in df.columns or args.label_column not in df.columns:
        logger.error(f"CSV must contain 'caption' and '{args.label_column}' columns.")
        return

    # Filter NaNs
    df = df[["caption", args.label_column]].dropna()
    
    captions = df["caption"].astype(str).tolist()
    labels = df[args.label_column].astype(str).tolist()
    
    if len(captions) == 0:
        logger.error("No valid rows found after dropping NaNs.")
        return

    # Load Model
    logger.info(f"Loading LLM2Vec model from {args.model_name_or_path}...")
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

    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(captions):
        logger.error(f"Unexpected embeddings shape: {embeddings.shape}")
        return

    # Calculate Neighborhood Purity
    logger.info("Calculating Neighborhood Purity...")
    k_values = [10, 30]
    purity_scores = calculate_neighborhood_purity(embeddings, labels, k_values=k_values)
    
    print("\n" + "="*40)
    print("Neighborhood Purity Results (LLM2Vec):")
    print("="*40)
    for k, score in purity_scores.items():
        if score is not None:
            print(f"k={k}: {score:.4f}")
    print("="*40 + "\n")

    # Interpretation hint
    print("Interpretation Hint:")
    print(" - ~0.1 (random baseline for 10 balanced classes): High overlap.")
    print(" - >> 0.1: Natural clustering/separation exists.")

    if args.output_file is None:
        base = os.path.splitext(os.path.basename(args.input_file))[0]
        output_path = os.path.join(os.getcwd(), f"{base}_knn_purity.json")
    else:
        output_path = args.output_file
        if os.path.isdir(output_path) or output_path.endswith(os.sep):
            base = os.path.splitext(os.path.basename(args.input_file))[0]
            output_path = os.path.join(output_path, f"{base}_knn_purity.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = {
        "input_file": args.input_file,
        "model_name_or_path": args.model_name_or_path,
        "peft_model_name_or_path": args.peft_model_name_or_path,
        "extra_model_name_or_path": args.extra_model_name_or_path,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "pooling_mode": args.pooling_mode,
        "instruction": args.instruction,
        "device": str(device),
        "enable_multiprocessing": bool(args.enable_multiprocessing),
        "n_samples": int(len(captions)),
        "k_values": k_values,
        "purity_scores": {str(k): v for k, v in purity_scores.items()},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved results to: {output_path}")

if __name__ == "__main__":
    main()
