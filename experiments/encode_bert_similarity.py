import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
import sys

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
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    parser = argparse.ArgumentParser(description="Encode text using a BERT-type model and calculate cosine similarity.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the BERT model (e.g. bert-base-uncased).")
    parser.add_argument("--sentence1", type=str, required=True, help="First sentence.")
    parser.add_argument("--sentence2", type=str, required=True, help="Second sentence.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    
    args = parser.parse_args()

    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")

    # Load Model
    logger.info(f"Loading model from {args.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModel.from_pretrained(args.model_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    model.to(device)
    model.eval()

    logger.info("Encoding...")
    sentences = [args.sentence1, args.sentence2]
    
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    # Convert to numpy
    embeddings = sentence_embeddings.cpu().numpy()
    
    vec1 = embeddings[0]
    vec2 = embeddings[1]
    
    # Calculate Cosine Similarity
    similarity = np.dot(vec1, vec2)
    
    print(f"Sentence 1: {args.sentence1}")
    print(f"Sentence 2: {args.sentence2}")
    print(f"Cosine Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main()
