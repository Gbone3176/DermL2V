import argparse
import json
import os
import torch
from llm2vec import LLM2Vec
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
    parser = argparse.ArgumentParser(description="Task4: Accuracy evaluation using LLM2Vec")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_model_name_or_path", type=str, default=None)
    parser.add_argument("--extra_model_name_or_path", type=str, nargs='*', default=None, help="One or more paths to extra models")
    parser.add_argument("--pooling_mode", type=str, default="mean")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--enable_bidirectional", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--safetensors", action="store_true", default=False)
    args = parser.parse_args()

    output_path = None
    if args.output:
        output_path = os.path.join(args.output, f"homo_RT_accuracy_{args.model_name}.json")
        if os.path.exists(output_path):
            print(f"Output already exists, skipping: {output_path}")
            return

    extra_models = []
    if args.extra_model_name_or_path:
        for m in args.extra_model_name_or_path:
            # Simple cleanup to be robust against "['path']" style inputs if passed as single string
            clean_m = m.strip("[]\"' ")
            if clean_m:
                extra_models.append(clean_m)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        extra_model_name_or_path=extra_models,
        pooling_mode=args.pooling_mode,
        max_length=args.max_length,
        enable_bidirectional=args.enable_bidirectional,
        torch_dtype=torch.bfloat16,
        use_safetensors=args.safetensors,
    ).to(device).eval()
    model.tokenizer.padding_side = "left"

    dataset = load_jsonl(args.input)
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]

    instruction = "Given a dermatology question, retrieve the most relevant dermatology text passage"
    
    # Prepare data for batch encoding
    all_queries = []
    all_candidate_sets = []
    valid_indices = []

    for i, item in enumerate(dataset):
        # New format parsing
        query = (item.get("original") or "").strip()
        pos_variant = (item.get("positive_variant") or "").strip()
        neg_variants = item.get("hard_negative_variants") or []
        
        # Ensure neg_variants is a list of strings
        if not isinstance(neg_variants, list):
            continue
            
        neg_variants = [str(n).strip() for n in neg_variants if str(n).strip()]
        candidates = [pos_variant] + neg_variants
        
        # We expect 1 positive and typically 3 negatives
        if not query or not pos_variant or len(candidates) < 2:
            continue
            
        all_queries.append([instruction, query])
        all_candidate_sets.append(candidates)
        valid_indices.append(i)

    if not all_queries:
        print("No valid samples found.")
        return

    print(f"Encoding {len(all_queries)} queries...")
    # Batch encode queries
    # model.encode expects List[List[str]] for instruction pairs
    q_reps = model.encode(all_queries, batch_size=args.batch_size, convert_to_tensor=True, device=device)

    print(f"Encoding candidates for {len(all_candidate_sets)} sets...")
    # Flatten candidates for batch encoding
    # Store the number of candidates per set to reconstruction
    cand_counts = [len(c) for c in all_candidate_sets]
    flat_candidates = [c for set_c in all_candidate_sets for c in set_c]
    # Candidates have no instruction -> ["", cand]
    flat_cand_inputs = [["", c] for c in flat_candidates]
    
    flat_cand_reps = model.encode(flat_cand_inputs, batch_size=args.batch_size, convert_to_tensor=True, device=device)

    # Calculate accuracy
    total = 0
    correct = 0
    
    import torch.nn.functional as F
    
    start_idx = 0
    for i, count in enumerate(tqdm(cand_counts, desc="Calculating similarities")):
        # Get query rep
        q_rep = q_reps[i].unsqueeze(0)  # (1, hidden)
        
        # Get candidate reps for this set
        end_idx = start_idx + count
        c_reps_set = flat_cand_reps[start_idx:end_idx]  # (count, hidden)
        start_idx = end_idx
        
        # Calculate cosine similarity
        sims = F.cosine_similarity(q_rep, c_reps_set, dim=1)
        pred = int(torch.argmax(sims).item())
        
        total += 1
        # The correct answer is always at index 0 (positive_variant)
        if pred == 0:
            correct += 1

    accuracy = (correct / total) if total > 0 else 0.0

    results = {
        "input": args.input,
        "model_path": args.base_model_name_or_path,
        "pooling_mode": args.pooling_mode,
        "max_length": args.max_length,
        "enable_bidirectional": args.enable_bidirectional,
        "count": total,
        "accuracy": accuracy,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Accuracy: {accuracy:.4f} (count={total})")



if __name__ == "__main__":
    main()
