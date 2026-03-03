import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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

def encode_texts(tokenizer, model, texts, device, max_length):
    # GPT2没有padding token,需要手动设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        # GPT2是decoder-only模型,获取last_hidden_state
        # output_hidden_states=True是默认包含在model(**enc)输出中的(如果config支持)
        # 这里使用hidden_states[-1]或者直接取last_hidden_state
        out = model(**enc, output_hidden_states=True)
        
        # 兼容不同版本的transformers输出
        if hasattr(out, "last_hidden_state"):
            hs = out.last_hidden_state
        elif hasattr(out, "hidden_states"):
            hs = out.hidden_states[-1]
        else:
            # Fallback for some causal LM outputs
            hs = out[0]

        # 均值池化 (Mean Pooling)
        # 注意要mask掉padding部分
        attention_mask = enc["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hs.size()).float()
        sum_embeddings = torch.sum(hs * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
    return pooled


def main():
    parser = argparse.ArgumentParser(description="Task4: Accuracy evaluation using GPT2 model")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用AutoModelForCausalLM加载GPT2
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to(device).eval()

    dataset = load_jsonl(args.input)

    total = 0
    correct = 0

    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        if args.max_samples and total >= args.max_samples:
            break
        item = dataset[i]
        
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
            
        q_emb = encode_texts(tokenizer, model, [query], device, args.max_length)[0]
        c_embs = encode_texts(tokenizer, model, candidates, device, args.max_length)
        
        import torch.nn.functional as F
        sims = F.cosine_similarity(q_emb.unsqueeze(0), c_embs, dim=1)
        pred = int(torch.argmax(sims).item())
        
        total += 1
        # The correct answer is always at index 0 (positive_variant)
        if pred == 0:
            correct += 1

    accuracy = (correct / total) if total > 0 else 0.0
    results = {
        "input": args.input,
        "model_path": args.model_path,
        "max_length": args.max_length,
        "max_samples": args.max_samples,
        "output": args.output,
        "count": total,
        "accuracy": accuracy,
    }
    
    if args.output:
        output_path = os.path.join(args.output, f"task4_accuracy_gpt2_{os.path.basename(args.model_path.replace('/', '-'))}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
    print(f"Accuracy: {accuracy:.4f} (count={total})")


if __name__ == "__main__":
    main()
