import argparse
import torch
import pandas as pd
import numpy as np
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
    
    max_k = max(k_values)
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
        current_indices = indices[:, 1:k+1]
        neighbor_labels = labels[current_indices]
        true_labels = labels.reshape(-1, 1)
        matches = (neighbor_labels == true_labels)
        point_purities = np.mean(matches, axis=1)
        avg_purity = np.mean(point_purities)
        
        results[k] = avg_purity
        logger.info(f"Neighborhood Purity (k={k}): {avg_purity:.4f}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Encode captions using GPT-2 and calculate Neighborhood Purity.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2", help="Path to the GPT-2 model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    
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
    
    captions = df["caption"].tolist()
    labels = df["hierarchical_Level1_label"].astype(str).tolist()
    
    if len(captions) == 0:
        logger.error("No valid data found.")
        return

    logger.info(f"Loading model from {args.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    logger.info("Encoding captions...")
    embeddings_list = []
    
    for i in tqdm(range(0, len(captions), args.batch_size), desc="Encoding"):
        batch_texts = captions[i : i + args.batch_size]
        
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=args.max_length, 
            return_tensors='pt'
        ).to(device)
        if model.config.model_type == "gpt2" and "token_type_ids" in encoded_input:
            encoded_input.pop("token_type_ids")
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        embeddings_list.append(sentence_embeddings.cpu().numpy())
        
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    logger.info("Calculating Neighborhood Purity...")
    purity_scores = calculate_neighborhood_purity(embeddings, labels, k_values=[10, 30])
    
    print("\n" + "="*40)
    print("Neighborhood Purity Results (GPT-2):")
    print("="*40)
    for k, score in purity_scores.items():
        if score is not None:
            print(f"k={k}: {score:.4f}")
    print("="*40 + "\n")

    print("Interpretation Hint:")
    print(" - ~0.1 (random baseline for 10 balanced classes): High overlap.")
    print(" - >> 0.1: Natural clustering/separation exists.")

if __name__ == "__main__":
    main()
