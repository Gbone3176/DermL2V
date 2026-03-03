import argparse
import torch
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import logging
import sys
import os
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
    # Using cosine distance (metric='cosine')
    # Note: sklearn NearestNeighbors with metric='cosine' expects inputs, and distances will be cosine distances.
    # Alternatively, since embeddings are usually normalized, Euclidean distance on normalized vectors 
    # ranks same as cosine similarity. But let's use metric='cosine' explicitly or normalize and use euclidean.
    # embed_bert_umap.py does normalization. So we can use 'cosine' or 'euclidean' on normalized data.
    # 'cosine' metric in sklearn might be slower than 'euclidean' with optimized kd_tree/ball_tree?
    # Actually, for normalized vectors, Euclidean distance is proportional to Cosine distance.
    # Let's stick to metric='cosine' for clarity, using 'brute' force algorithm if 'auto' fails or for exactness.
    # For speed on large datasets, normalized + euclidean + 'ball_tree' is often good.
    # Here we use 'cosine' with default algorithm.
    
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
        # Actually indices include the point itself at index 0 usually (distance 0)
        
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
    parser = argparse.ArgumentParser(description="Encode captions using BERT and calculate Neighborhood Purity.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="Path to the BERT model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save JSON results.")
    
    args = parser.parse_args()

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

    if "caption" not in df.columns or "hierarchical_Level1_label" not in df.columns:
        logger.error("CSV must contain 'caption' and 'hierarchical_Level1_label' columns.")
        return

    # Filter NaNs
    df = df[["caption", "hierarchical_Level1_label"]].dropna()
    
    captions = df["caption"].tolist()
    # Convert labels to string as requested
    labels = df["hierarchical_Level1_label"].astype(str).tolist()
    
    if len(captions) == 0:
        logger.error("No valid data found.")
        return

    # Load Model
    logger.info(f"Loading model from {args.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            use_safetensors=True,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    model.to(device)
    model.eval()

    # Encode
    logger.info("Encoding captions...")
    embeddings_list = []
    
    for i in tqdm(range(0, len(captions), args.batch_size), desc="Encoding"):
        batch_texts = captions[i : i + args.batch_size]
        
        # Tokenize
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        # Pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        embeddings_list.append(sentence_embeddings.cpu().numpy())
        
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Calculate Neighborhood Purity
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
    
    print("\n" + "="*40)
    print("Neighborhood Purity Results:")
    print("="*40)
    for k, score in purity_scores.items():
        if score is not None:
            print(f"k={k}: {score:.4f}")
    print("="*40 + "\n")

    # Interpretation hint
    print("Interpretation Hint:")
    print(" - ~0.1 (random baseline for 10 balanced classes): High overlap.")
    print(" - >> 0.1: Natural clustering/separation exists.")

if __name__ == "__main__":
    main()
