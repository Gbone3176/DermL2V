import argparse
import os

import torch

from llm2vec.pooling_latent import LatentAttentionPooling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="latentpooling_init.pt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"{args.path} does not exist")

    weights = torch.load(args.path, map_location="cpu")
    if not isinstance(weights, torch.Tensor):
        raise TypeError("Loaded object is not a torch.Tensor")
    if weights.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(weights.shape)}")

    num_latents, d_model = weights.shape

    module = LatentAttentionPooling(d_model=d_model, num_latents=num_latents)
    if module.latents.shape != weights.shape:
        raise ValueError(
            f"Shape mismatch between module.latents {tuple(module.latents.shape)} and weights {tuple(weights.shape)}"
        )
    module.latents.data.copy_(weights.to(module.latents.dtype))

    hidden_states = torch.randn(args.batch_size, args.seq_len, d_model)
    attention_mask = torch.ones(args.batch_size, args.seq_len, dtype=torch.long)

    with torch.no_grad():
        output = module(hidden_states, attention_mask=attention_mask)

    print(f"Loaded weights from {args.path}")
    print(f"Latent matrix shape: {weights.shape}")
    print(f"Module num_latents: {module.num_latents}, d_model: {module.d_model}")
    print(f"Forward output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()

