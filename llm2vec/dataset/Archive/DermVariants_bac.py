import json
import random
import os

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

from accelerate import PartialState
PartialState()

logger = get_logger(__name__, log_level="INFO")

DERM_EMBEDDING_PROMPTS = {
    "SemVariants": [
    "Read the provided dermatological condition description and return the candidate description that matches its meaning most closely.",
    "Given a skin-condition description, select the candidate description with the highest meaning-level similarity.",
    "Match the input dermatology description to the closest candidate description by semantics rather than exact wording.",
    "From all candidate descriptions, choose the one that best corresponds to the same dermatological condition described in the input.",
    "Identify the candidate description that is most conceptually aligned with the input skin-condition description.",
    "Return the single candidate description that best preserves the clinical meaning of the input dermatological description.",
    "Compare the input skin-condition description against the candidates and output the most semantically relevant one.",
    "Find the candidate description that would be the best paraphrase of the input dermatological condition description.",
    "Retrieve the candidate description that is nearest in meaning to the input dermatology description, ignoring surface-level phrasing.",
    "Select the candidate description that most accurately reflects the same underlying dermatological condition as the input description.",
    ],
    "VisVariants":  [
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
}


class DermVariants(Dataset):
    def __init__(
        self,
        dataset_name: str = "DermVariants",
        split: str = "train",
        file_path: str | None = "/storage/dataset/dermatoscop/Derm1M/DermVariantsData",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        dermqa_upsample_ratio: int = 1,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.dermqa_upsample_ratio = dermqa_upsample_ratio

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str | None = None):
        logger.info(f"Loading DermVariants data from {file_path}...")

        all_samples = []
        id_ = 0

        if file_path is None:
            raise ValueError("file_path must be provided for DermVariants.")
        if not os.path.exists(file_path):
            raise ValueError(f"file_path {file_path} does not exist.")

        for dataset in DERM_EMBEDDING_PROMPTS:
            dataset_path = os.path.join(file_path, f"{dataset}_{self.split}.jsonl")
            if not os.path.exists(dataset_path):
                continue
            logger.info(f"Loading dataset {dataset} from {dataset_path}...")

            with open(dataset_path, "r") as f:
                dataset_samples = [json.loads(d) for d in f if d.strip()]

            subtask_samples = []
            for sample in dataset_samples:
                instruction = random.choice(DERM_EMBEDDING_PROMPTS[dataset])
                query_text = sample["original"]
                pos_text = sample["positive_variant"]
                neg_text = sample["hard_negative_variant"]

                if dataset in ["SemVariants", "VisVariants"]:
                    query = f"{instruction}" + self.separator + query_text
                    pos = f"{instruction}" + self.separator + pos_text
                    neg = f"{instruction}" + self.separator + neg_text
                else:
                    query = f"{instruction}" + self.separator + query_text
                    pos = self.separator + pos_text
                    neg = self.separator + neg_text

                subtask_samples.append(
                    DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name=dataset,
                    )
                )
                id_ += 1

            if not subtask_samples:
                continue

            if dataset == "DermQA" and self.dermqa_upsample_ratio > 1:
                for _ in range(self.dermqa_upsample_ratio):
                    for s in subtask_samples:
                        all_samples.append(
                            DataSample(
                                id_=id_,
                                query=s.query,
                                positive=s.positive,
                                negative=s.negative,
                                task_name=s.task_name,
                            )
                        )
                        id_ += 1
            else:
                all_samples.extend(subtask_samples)

        indices = list(range(len(all_samples)))
        if self.shuffle_individual_datasets:
            random.shuffle(indices)

        logger.info(
            f"Batching DermVariants data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for i in range(0, len(indices), self.effective_batch_size):
            batch = indices[i : i + self.effective_batch_size]
            if len(batch) == self.effective_batch_size:
                all_batches.append(batch)
            else:
                logger.info("Skip 1 batch for DermVariants.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split in ["train", "validation", "test"]:
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )


if __name__ == "__main__":
    split = "train"
    dataset = DermVariants(
        split=split,
        file_path="/storage/dataset/dermatoscop/Derm1M/DermVariantsData",
        effective_batch_size=32,
        dermqa_upsample_ratio=1,
    )

    print(f"Loaded {len(dataset)} samples for split {split}")
    sentences = []
    for sample in dataset:
        sentences.extend(sample.texts)
        if len(sentences) >= 12:
            break
    print(sentences)
