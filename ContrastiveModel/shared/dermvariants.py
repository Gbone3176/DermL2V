"""DermVariants triplet loading for isolated contrastive fine-tuning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from torch.utils.data import Dataset


DERM_EMBEDDING_PROMPTS: dict[str, list[str]] = {
    "SemVariants": [
        "Read the provided dermatological condition description and return the candidate description that matches its meaning most closely.",
        "Given a skin-condition description, select the candidate description with the highest meaning-level similarity.",
        "Match the input dermatology description to the closest candidate description by semantics rather than exact wording.",
        "From all candidate descriptions, choose the one that best corresponds to the same dermatological condition described in the input.",
        "Identify the candidate description that is most conceptually aligned with the input skin-condition description.",
        "Return the single candidate description that best preserves the clinical meaning of the input dermatological description.",
        "Compare the input skin-condition description against the candidates and output the most semantically relevant one.",
        "Find the candidate description that would be the best paraphrase of the input dermatology condition description.",
        "Retrieve the candidate description that is nearest in meaning to the input dermatology description, ignoring surface-level phrasing.",
        "Select the candidate description that most accurately reflects the same underlying dermatological condition as the input description.",
    ],
    "VisVariants": [
        "Given a diagnosis-style dermatology text, retrieve the visual-description text that best matches it in meaning.",
        "Match a dermatological diagnosis or summary text to the most semantically aligned visual description text, and return the top match.",
        "Using the provided dermatology diagnostic statement as input, select the visual-description passage that is most relevant.",
        "From a pool of visual-description texts, return the one that most closely corresponds to the given dermatological diagnosis or summary.",
        "Identify the single visual-description text that best reflects the condition described by the provided diagnosis-oriented dermatology text.",
    ],
    "DermQA": [
        "Given a dermatology-related question, select the answer that is most relevant to what the question is asking.",
        "For a question in the skin-disease domain, return the candidate answer with the highest semantic relevance.",
        "Match the provided skin-disease question to the best corresponding answer from the available answers.",
        "From the answer candidates, pick the one that best aligns with the content and intent of the dermatology question.",
        "Given a skin-condition question, identify the answer that is most suitable and most closely related in meaning.",
        "Given a skin-disease question, retrieve the answer that is most semantically aligned, regardless of whether it concerns symptoms, diagnosis, or treatment.",
        "Select the answer that best matches the provided question within the dermatology domain.",
        "Given a dermatologic question, return the answer that most closely corresponds to the information being asked for.",
        "For the provided question about skin disorders, find and return the most relevant answer among the candidates.",
    ],
    "SI1": [
        "Retrieve the most appropriate answer for this dermatology question.",
        "Retrieve the answer entry that best matches the clinical vignette and question.",
        "Retrieve the most relevant response to the patient scenario described.",
        "Retrieve the best-matching answer based on the key clinical clues in the case.",
        "Given the dermatology prompt, retrieve the answer that most directly resolves the question.",
        "Retrieve the answer that is most consistent with standard dermatology clinical reasoning for this case.",
        "Retrieve the answer that best explains the findings and fits the question intent.",
        "Retrieve the answer that best aligns with the case details and the provided answer choices, if any.",
        "Retrieve the closest matching answer from the dataset for this dermatology related query.",
    ],
    "SI2": [
        "Given a dermatology question, retrieve the single most relevant and most correct answer passage that directly answers it.",
        "Given a question about a skin condition, find the answer that most accurately and directly addresses the question.",
        "Find the dermatology answer passage that best matches this question and provides the highest-correctness response.",
        "Given a dermatologic presentation question, retrieve the answer that most directly answers it and is most medically accurate.",
        "Retrieve the most relevant dermatology answer that correctly resolves what the question is asking.",
        "Given a dermatology clinical question, retrieve the answer passage that is most relevant and has the highest factual correctness.",
        "Match this dermatology query to the answer that most precisely answers the question and is most correct.",
    ],
}


@dataclass(frozen=True)
class DermTriplet:
    id: int
    query: str
    positive: str
    negative: str
    task_name: str


class DermTripletDataset(Dataset):
    def __init__(self, samples: list[DermTriplet]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DermTriplet:
        return self.samples[index]


def load_dermvariants_triplets(
    data_dir: str | Path,
    split: str = "train",
    effective_batch_size: int = 32,
    shuffle_individual_datasets: bool = True,
    seed: int = 42,
    separator: str = "!@#$%^&*()",
    dermqa_upsample_ratio: int = 1,
    use_query_instruction: bool = True,
    max_samples: int | None = None,
) -> list[DermTriplet]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"DermVariants data_dir does not exist: {data_path}")

    rng = random.Random(seed)
    all_samples: list[DermTriplet] = []
    next_id = 0

    for task_name, prompts in DERM_EMBEDDING_PROMPTS.items():
        dataset_path = data_path / f"{task_name}_{split}.jsonl"
        if not dataset_path.exists():
            continue

        task_samples = []
        for row in _iter_jsonl(dataset_path):
            instruction = rng.choice(prompts)
            query_text = row["original"]
            pos_text = row["positive_variant"]
            neg_text = row["hard_negative_variant"]

            if use_query_instruction:
                query = f"{instruction}{separator}{query_text}"
                positive = f"{separator}{pos_text}"
                negative = f"{separator}{neg_text}"
            else:
                query = query_text
                positive = pos_text
                negative = neg_text

            task_samples.append(
                DermTriplet(
                    id=next_id,
                    query=query,
                    positive=positive,
                    negative=negative,
                    task_name=task_name,
                )
            )
            next_id += 1

        repeat = dermqa_upsample_ratio if task_name == "DermQA" else 1
        for _ in range(max(1, repeat)):
            all_samples.extend(task_samples)

    if not all_samples:
        raise RuntimeError(f"No DermVariants samples loaded from {data_path} split={split}")

    indices = list(range(len(all_samples)))
    if shuffle_individual_datasets:
        rng.shuffle(indices)

    batched_indices: list[list[int]] = []
    for start in range(0, len(indices), effective_batch_size):
        batch = indices[start : start + effective_batch_size]
        if len(batch) == effective_batch_size:
            batched_indices.append(batch)
    rng.shuffle(batched_indices)

    ordered_indices = [idx for batch in batched_indices for idx in batch]
    samples = [all_samples[idx] for idx in ordered_indices]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def collate_triplets(batch: list[DermTriplet]) -> dict[str, list[str]]:
    return {
        "query": [sample.query for sample in batch],
        "positive": [sample.positive for sample in batch],
        "negative": [sample.negative for sample in batch],
        "task_name": [sample.task_name for sample in batch],
    }


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
