# dataset with negatives

import csv
import random
from typing import Optional

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

DermPrompt = "Given a question related to dermatology, retrieve the most relevant answers: "

SPLIT_TO_PATH = {
    "train": "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/text-img/dermatoscop/DermQA-datasets/combined_data_clean_with_neg_train.csv",
    "validation": "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/text-img/dermatoscop/DermQA-datasets/combined_data_clean_with_neg_valid.csv",
    "test": "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/text-img/dermatoscop/DermQA-datasets/combined_data_clean_with_neg_test.csv",
}

class DermQA(Dataset):
    def __init__(
        self,
        dataset_name: str = "DermQA",
        split: str = "train",
        file_path: Optional[str] = None,
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split.lower()
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        if self.split not in ("train", "validation", "test"):
            raise ValueError(f"Unsupported split: {split}")

        # Allow overriding the default path; otherwise use mapping
        self.file_path = file_path or SPLIT_TO_PATH[self.split]

        self.data = []
        self.load_data(self.file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str):
        logger.info(f"Loading DermQA CSV data for split='{self.split}' from {file_path}...")

        all_samples = []
        indices = []
        id_ = 0

        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = {"prompt", "response", "neg_responce"}
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing required columns in CSV: {missing}")

            for row in reader:
                prompt = (row.get("prompt") or "").strip()
                positive = (row.get("response") or "").strip()
                negative = (row.get("neg_responce") or "").strip()

                all_samples.append(
                    DataSample(
                        id_=id_,
                        query=f"{DermPrompt}{self.separator}{prompt}",
                        positive=f"{self.separator}{positive}",
                        negative=f"{self.separator}{negative}",
                        task_name=self.dataset_name,
                    )
                )
                indices.append(id_)
                id_ += 1

        # Shuffle batches only for train (or if explicitly requested)
        if self.split == "train" and self.shuffle_individual_datasets:
            random.shuffle(indices)

        logger.info(
            f"Batching DermQA CSV data for effective batch size {self.effective_batch_size}..."
        )
        all_batches = []
        for i in range(0, len(indices), self.effective_batch_size):
            batch = indices[i : i + self.effective_batch_size]
            if len(batch) == self.effective_batch_size:
                all_batches.append(batch)
            else:
                logger.info("Skip 1 batch due to insufficient size.")

        if self.split == "train" and self.shuffle_individual_datasets:
            random.shuffle(all_batches)

        final_idx_order = [idx for batch in all_batches for idx in batch]
        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples for split='{self.split}'.")

    def __getitem__(self, index):
        sample = self.data[index]
        return TrainSample(texts=[sample.query, sample.positive, sample.negative], label=1.0)