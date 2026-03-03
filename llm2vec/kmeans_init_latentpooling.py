import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from transformers import HfArgumentParser

from llm2vec import LLM2Vec
from llm2vec.dataset.utils import load_dataset
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    extra_model_name_or_path: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "Path to extra Lora models"}
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable bidirectional attention in the model. If set to False, the model will use unidirectional attention."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


def collect_texts_from_dataset(dataset) -> List[str]:
    texts: List[str] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if not hasattr(sample, "texts"):
            continue
        sample_texts = getattr(sample, "texts")
        for t in sample_texts[:2]:
            if isinstance(t, str):
                texts.append(t)
    return texts


def encode_corpus(
    model: LLM2Vec,
    texts: List[str],
    batch_size: int,
    separator: str = "!@#$%^&*()",
) -> np.ndarray:
    sentences: List[List[Any]] = []
    for t in texts:
        if separator in t:
            inst, content = t.split(separator, 1)
        else:
            inst, content = "", t
        sentences.append([inst.strip(), content.strip(), 0])
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        convert_to_tensor=False,
    )
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    return embeddings


def run_kmeans_init(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    output_dir: str,
    kmeans_k: int = 512,
    encode_batch_size: int = 16,
    use_separator: bool = True,
) -> Dict[str, Any]:
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        extra_model_name_or_path=model_args.extra_model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    train_dataset = load_dataset(
        data_args.dataset_name,
        split="train",
        file_path=data_args.dataset_file_path,
        effective_batch_size=encode_batch_size,
    )
    texts = collect_texts_from_dataset(train_dataset)

    embeddings = encode_corpus(
        model,
        texts,
        batch_size=encode_batch_size,
        separator="!@#$%^&*()",
    )

    clustering_model = MiniBatchKMeans(
        n_clusters=kmeans_k,
        batch_size=min(4096, max(kmeans_k * 4, 1024)),
        init="k-means++",
        n_init="auto",
    )
    clustering_model.fit(embeddings)
    centers = clustering_model.cluster_centers_

    os.makedirs(output_dir, exist_ok=True)
    centers_tensor = torch.from_numpy(centers).to(torch.float32)
    save_path = os.path.join(output_dir, "latentpooling_init.pt")
    torch.save(centers_tensor, save_path)

    return {
        "num_texts": len(texts),
        "embedding_dim": centers_tensor.shape[1],
        "num_clusters": centers_tensor.shape[0],
        "save_path": save_path,
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        output_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
        output_dir = os.getcwd()

    results = run_kmeans_init(
        model_args=model_args,
        data_args=data_args,
        output_dir=output_dir,
        kmeans_k=512,
        encode_batch_size=16,
        use_separator=True,
    )

    print(f"Encoded {results['num_texts']} texts")
    print(f"KMeans centers shape: {results['num_clusters']} x {results['embedding_dim']}")
    print(f"Saved latent pooling initialization to: {results['save_path']}")


if __name__ == "__main__":
    main()
