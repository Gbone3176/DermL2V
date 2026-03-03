import argparse
import time
import random
import string
import torch

from llm2vec.llm2vec import LLM2Vec


def make_text(n_words: int = 32) -> str:
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(n_words)]
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM2Vec encoding throughput")
    parser.add_argument("--model", type=str, required=True, help="Base model name or path for LLM2Vec.from_pretrained")
    parser.add_argument("--pooling_mode", type=str, default="mean", help="Pooling mode (mean, weighted_mean, eos_token, last_token, bos_token, latent_pooling)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding")
    parser.add_argument("--seq_words", type=int, default=64, help="Approximate number of words per sequence")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches to run for timing")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (auto if None)")
    parser.add_argument("--use_separator", action="store_true", help="Benchmark encode_with_separator instead of encode_text")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model '{args.model}' with pooling_mode='{args.pooling_mode}' on device '{device}'...")
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.model,
        pooling_mode=args.pooling_mode,
    )
    model.to(device)
    model.eval()

    # Warmup texts
    def build_batch(batch_size: int):
        if args.use_separator:
            return [
                f"instruction: summarize!@#$%^&*(){make_text(args.seq_words)}"
                for _ in range(batch_size)
            ]
        else:
            return [make_text(args.seq_words) for _ in range(batch_size)]

    # Warm-up
    warm_batch = build_batch(args.batch_size)
    with torch.no_grad():
        if args.use_separator:
            _ = model.encode_with_separator(warm_batch, device=device)
        else:
            _ = model.encode_text(warm_batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timing
    start = time.time()
    total_seqs = 0
    for _ in range(args.num_batches):
        batch = build_batch(args.batch_size)
        with torch.no_grad():
            if args.use_separator:
                _ = model.encode_with_separator(batch, device=device)
            else:
                _ = model.encode_text(batch)
        total_seqs += len(batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - start

    seqs_per_sec = total_seqs / max(elapsed, 1e-9)
    print(f"Processed {total_seqs} sequences in {elapsed:.2f}s -> {seqs_per_sec:.2f} seq/s")


if __name__ == "__main__":
    main()