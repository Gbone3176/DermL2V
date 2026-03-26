import argparse
import os
import sys

VENDOR_DIR = os.path.join(os.path.dirname(__file__), ".vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.append(VENDOR_DIR)

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import logging
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def _sanitize_for_filename(value: str) -> str:
    value = value.replace("/", "-")
    value = value.replace(os.sep, "-")
    value = value.replace(" ", "-")
    sanitized = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            sanitized.append(ch)
        else:
            sanitized.append("-")
    while "--" in "".join(sanitized):
        sanitized = "".join(sanitized).replace("--", "-")
        sanitized = list(sanitized)
    return "".join(sanitized).strip("-")

def _float_for_filename(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if s else "0"

def main():
    parser = argparse.ArgumentParser(description="Encode captions using GPT-2, reduce with UMAP, and visualize.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_folder", type=str, default="umap_outputs", help="Folder to save the UMAP plot.")
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2", help="Path to the GPT-2 model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    parser.add_argument("--point_size", type=int, default=10, help="Size of the points in the scatter plot.")
    parser.add_argument("--pca_n_components", type=int, default=50, help="PCA n_components before UMAP (set 0 to disable).")
    parser.add_argument("--pca_random_state", type=int, default=42, help="PCA random_state (used for randomized solver).")
    parser.add_argument("--umap_n_components", type=int, default=2, help="UMAP n_components.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap_min_dist", type=float, default=0.0, help="UMAP min_dist.")
    parser.add_argument("--umap_random_state", type=int, default=42, help="UMAP random_state.")
    parser.add_argument("--umap_metric", type=str, default="cosine", help="UMAP metric.")
    
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

    captions = df["caption"].tolist()
    labels = df["hierarchical_Level4_label"].tolist()
    
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
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        if model.config.model_type == "gpt2" and "token_type_ids" in encoded_input:
            encoded_input.pop("token_type_ids")
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        embeddings_list.append(sentence_embeddings.cpu().numpy())
        
    embeddings = np.concatenate(embeddings_list, axis=0)
    embeddings = embeddings.astype(np.float32, copy=False)

    pca_components_used = 0
    if args.pca_n_components and args.pca_n_components > 0:
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]
        pca_components_used = min(args.pca_n_components, n_samples, n_features)
        if pca_components_used < args.pca_n_components:
            logger.info(f"PCA n_components adjusted to {pca_components_used} based on data shape {embeddings.shape}.")
        logger.info(f"Running PCA (n_components={pca_components_used})...")
        pca = PCA(n_components=pca_components_used, random_state=args.pca_random_state)
        embeddings = pca.fit_transform(embeddings).astype(np.float32, copy=False)
    
    logger.info("Running UMAP...")
    n_samples = len(embeddings)
    n_neighbors = min(max(2, args.umap_n_neighbors), n_samples - 1) if n_samples > 1 else 1
    
    if n_samples < 2:
         logger.warning("Not enough samples for UMAP (need at least 2). Skipping UMAP.")
         return

    reducer = umap.UMAP(
        n_components=args.umap_n_components,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.umap_random_state,
        metric=args.umap_metric,
    )
    embedding_2d = reducer.fit_transform(embeddings)
    
    logger.info("Plotting...")
    plt.figure(figsize=(12, 8))
    
    plot_df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'label': labels
    })
    
    sns.scatterplot(
        data=plot_df,
        x='x',
        y='y',
        hue='label',
        palette='turbo',
        s=args.point_size,
        alpha=0.7
    )
    
    plt.axis("off")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    os.makedirs(args.output_folder, exist_ok=True)

    model_part = _sanitize_for_filename(args.model_name_or_path)
    metric_part = _sanitize_for_filename(args.umap_metric)
    pca_part = f"_pca{pca_components_used}" if pca_components_used > 0 else ""
    filename = (
        f"umap_nc{args.umap_n_components}"
        f"_nn{n_neighbors}"
        f"_md{_float_for_filename(args.umap_min_dist)}"
        f"_rs{args.umap_random_state}"
        f"_metric{metric_part}"
        f"{pca_part}"
        f"_bs{args.batch_size}"
        f"_model{model_part}"
        ".png"
    )
    output_path = os.path.join(args.output_folder, filename)
    plt.savefig(output_path, dpi=300)
    logger.info(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
