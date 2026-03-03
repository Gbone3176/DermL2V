import argparse
import json
import os
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def main():
    parser = argparse.ArgumentParser(description="Task4: Accuracy evaluation using Qwen3-Embedding")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Qwen3 model using SentenceTransformer
    # Qwen3 models are large, use float16 to save memory
    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": True,
        "torch_dtype": torch.float16
    }
    # Qwen3-embedding uses left padding for batch generation
    tokenizer_kwargs = {"padding_side": "left", "trust_remote_code": True}
    
    print(f"Loading model from {args.model_name_or_path}...")
    try:
        model = SentenceTransformer(
            args.model_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
            trust_remote_code=True
        )
    except TypeError:
        # Fallback for older sentence-transformers versions that don't accept trust_remote_code in __init__
        # It might be passed via model_kwargs
        model = SentenceTransformer(
            args.model_name_or_path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device
        )
    
    # Set max sequence length
    model.max_seq_length = args.max_length

    dataset = load_jsonl(args.input)
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]

    # Qwen3-Embedding typically doesn't need complex instruction formatting for retrieval tasks
    # when used via SentenceTransformer with prompt_name="query" for queries.
    # But for Task4 (matching), we treat it as query-candidate matching.
    
    # Prepare data for batch encoding
    all_queries = []
    all_candidate_sets = []
    valid_indices = []

    for i, item in enumerate(dataset):
        query = (item.get("original") or "").strip()
        pos_variant = (item.get("positive_variant") or "").strip()
        neg_variants = item.get("hard_negative_variants") or []
        
        if not isinstance(neg_variants, list):
            continue
            
        neg_variants = [str(n).strip() for n in neg_variants if str(n).strip()]
        candidates = [pos_variant] + neg_variants
        
        if not query or not pos_variant or len(candidates) < 2:
            continue
            
        all_queries.append(query)
        all_candidate_sets.append(candidates)
        valid_indices.append(i)

    if not all_queries:
        print("No valid samples found.")
        return

    print(f"Encoding {len(all_queries)} queries...")
    # Batch encode queries
    # Qwen3-Embedding uses prompt_name="query" for queries to apply specific instruction/masking
    q_reps = model.encode(
        all_queries, 
        prompt_name="query", 
        batch_size=args.batch_size, 
        convert_to_tensor=True,
        show_progress_bar=True
    )

    print(f"Encoding candidates for {len(all_candidate_sets)} sets...")
    cand_counts = [len(c) for c in all_candidate_sets]
    flat_candidates = [c for set_c in all_candidate_sets for c in set_c]
    
    # Encode candidates without specific prompt_name (or default behavior)
    flat_cand_reps = model.encode(
        flat_candidates, 
        batch_size=args.batch_size, 
        convert_to_tensor=True,
        show_progress_bar=True
    )

    # Calculate accuracy
    total = 0
    correct = 0
    
    import torch.nn.functional as F
    
    start_idx = 0
    for i, count in enumerate(tqdm(cand_counts, desc="Calculating similarities")):
        q_rep = q_reps[i].unsqueeze(0)  # (1, hidden)
        
        end_idx = start_idx + count
        c_reps_set = flat_cand_reps[start_idx:end_idx]  # (count, hidden)
        start_idx = end_idx
        
        sims = F.cosine_similarity(q_rep, c_reps_set, dim=1)
        pred = int(torch.argmax(sims).item())
        
        total += 1
        if pred == 0:
            correct += 1

    accuracy = (correct / total) if total > 0 else 0.0

    results = {
        "input": args.input,
        "model_path": args.model_name_or_path,
        "max_length": args.max_length,
        "count": total,
        "accuracy": accuracy,
    }

    if args.output:
        model_name_str = os.path.basename(args.model_name_or_path.rstrip("/")).replace("/", "-")
        output_path = os.path.join(args.output, f"task4_accuracy_{model_name_str}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Accuracy: {accuracy:.4f} (count={total})")

if __name__ == "__main__":
    main()
