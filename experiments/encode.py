import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from llm2vec import LLM2Vec
import logging
import sys
from transformers import AutoTokenizer, AutoConfig, AutoModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    parser = argparse.ArgumentParser(description="Encode text dataset using LLM2Vec and save as .npy files.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the LLM2Vec model.")
    parser.add_argument("--peft_model_name_or_path", type=str, default=None, help="Path to the PEFT model.")
    parser.add_argument("--extra_model_name_or_path", type=str, nargs="+", default=None, help="Path to extra PEFT models (list).")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .jsonl file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to save .npy files.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--pooling_mode",type=str,default="mean",choices=["mean", "weighted_mean", "eos_token", "last_token", "bos_token", "latent_pooling"], help="Pooling strategy for sentence embeddings.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    parser.add_argument("--instruction", type=str, default="", help="Instruction to prepend to text (optional).")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for distributed processing.")
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard ID (0 to num_shards-1).")
    
    args = parser.parse_args()

    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # Load Model
    logger.info(f"Loading model from {args.model_name_or_path}...")
    
    # Use LLM2Vec.from_pretrained to handle model loading, config, and PEFT integration automatically
    llm2vec_model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        extra_model_name_or_path=args.extra_model_name_or_path,
        merge_peft=True,  # Merge weights for efficiency
        pooling_mode=args.pooling_mode,
        max_length=args.max_length,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    # Prepare for batch processing
    batch_data = []
    
    logger.info(f"Processing {args.input_file}...")
    
    # Count total lines for progress bar (optional, might take time for huge files)
    total_lines = sum(1 for _ in open(args.input_file))
    
    iterator = load_jsonl(args.input_file)
    
    for i, item in enumerate(tqdm(iterator, total=total_lines, desc=f"Encoding (Shard {args.shard_id}/{args.num_shards})")):
        # Sharding logic: skip items not belonging to this shard
        if i % args.num_shards != args.shard_id:
            continue

        caption = item.get("caption", "")
        image_path = item.get("image_path", "")
        
        if not caption or not image_path:
            continue
            
        # Determine output path
        rel_path = image_path
        # Replace extension with .npy
        base_name = os.path.splitext(rel_path)[0]
        output_rel_path = base_name + ".npy"
        output_full_path = os.path.join(args.output_dir, output_rel_path)
        
        # Check if exists
        if os.path.exists(output_full_path) and not args.overwrite:
            continue
            
        batch_data.append({
            "text": caption,
            "output_path": output_full_path
        })
        
        if len(batch_data) >= args.batch_size:
            process_batch(llm2vec_model, batch_data, args.batch_size, device, args.instruction)
            batch_data = []
            
    # Process remaining
    if batch_data:
        process_batch(llm2vec_model, batch_data, args.batch_size, device, args.instruction)
        
    logger.info("Done!")

def process_batch(model, batch_data, batch_size, device, instruction=""):
    texts = [item["text"] for item in batch_data]
    
    # If instruction is provided, format as [[instruction, text], ...]
    if instruction:
        inputs = [[instruction, text] for text in texts]
    else:
        inputs = texts # llm2vec handles list of strings automatically
        
    # Encode
    # Using convert_to_numpy=True to get numpy arrays directly
    embeddings = model.encode(
        inputs, 
        batch_size=batch_size, 
        show_progress_bar=False, 
        convert_to_numpy=True, 
        device=device
    )
    
    # Save
    for i, item in enumerate(batch_data):
        output_path = item["output_path"]
        emb = embeddings[i]
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        np.save(output_path, emb)

if __name__ == "__main__":
    main()
