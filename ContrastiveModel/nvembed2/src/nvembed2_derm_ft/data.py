import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from .prompts import DERM_EMBEDDING_PROMPTS


@dataclass
class TripletSample:
    query: str
    positive: str
    negative: str
    task_name: str


class DermVariantsTripletDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        split: str,
        separator: str,
        dermqa_upsample_ratio: int = 1,
        seed: int = 42,
    ) -> None:
        self.file_path = file_path
        self.split = split
        self.separator = separator
        self.dermqa_upsample_ratio = dermqa_upsample_ratio
        self.rng = random.Random(seed)
        self.samples = self._load()

    def _load(self) -> List[TripletSample]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"DermVariants path does not exist: {self.file_path}")

        all_samples: List[TripletSample] = []
        for task_name, prompts in DERM_EMBEDDING_PROMPTS.items():
            task_file = os.path.join(self.file_path, f"{task_name}_{self.split}.jsonl")
            if not os.path.exists(task_file):
                continue
            with open(task_file, "r", encoding="utf-8") as handle:
                task_rows = [json.loads(line) for line in handle if line.strip()]

            task_samples: List[TripletSample] = []
            for row in task_rows:
                instruction = self.rng.choice(prompts)
                task_samples.append(
                    TripletSample(
                        query=f"{instruction}{self.separator}{row['original']}",
                        positive=f"{self.separator}{row['positive_variant']}",
                        negative=f"{self.separator}{row['hard_negative_variant']}",
                        task_name=task_name,
                    )
                )
            if task_name == "DermQA" and self.dermqa_upsample_ratio > 1:
                task_samples = task_samples * self.dermqa_upsample_ratio
            all_samples.extend(task_samples)

        if not all_samples:
            raise ValueError(f"No usable samples were found under {self.file_path} for split={self.split}")

        self.rng.shuffle(all_samples)
        return all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        sample = self.samples[index]
        return {
            "query": sample.query,
            "positive": sample.positive,
            "negative": sample.negative,
            "task_name": sample.task_name,
        }


def _tokenize_with_pool_mask(
    tokenizer,
    texts: Sequence[str],
    separator: str,
    max_length: int,
    add_eos: bool,
) -> Dict[str, torch.Tensor]:
    eos = tokenizer.eos_token or ""
    processed_texts = [text + eos if add_eos else text for text in texts]
    encoded = tokenizer(
        list(processed_texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    pool_mask = torch.zeros_like(encoded["attention_mask"])

    for i, raw_text in enumerate(texts):
        attention_len = int(encoded["attention_mask"][i].sum().item())
        if separator not in raw_text:
            pool_mask[i, :attention_len] = 1
            continue

        prefix, suffix = raw_text.split(separator, 1)
        prefix_with_sep = prefix + separator
        suffix_with_eos = suffix + (eos if add_eos else "")
        prefix_len = len(tokenizer(prefix_with_sep, add_special_tokens=False)["input_ids"])
        suffix_len = len(tokenizer(suffix_with_eos, add_special_tokens=False)["input_ids"])
        start = min(prefix_len, attention_len)
        end = min(prefix_len + suffix_len, attention_len)
        if end <= start:
            pool_mask[i, :attention_len] = 1
        else:
            pool_mask[i, start:end] = 1

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "pool_mask": pool_mask,
    }


class TripletCollator:
    def __init__(self, tokenizer, max_length: int, separator: str, add_eos: bool = True) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator = separator
        self.add_eos = add_eos

    def __call__(self, batch: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]
        negatives = [item["negative"] for item in batch]
        return {
            "query": _tokenize_with_pool_mask(self.tokenizer, queries, self.separator, self.max_length, self.add_eos),
            "positive": _tokenize_with_pool_mask(self.tokenizer, positives, self.separator, self.max_length, self.add_eos),
            "negative": _tokenize_with_pool_mask(self.tokenizer, negatives, self.separator, self.max_length, self.add_eos),
        }
