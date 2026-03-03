import json
import random
from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

DermPrompt = "Given a description of a dermatological condition, retrieve the description with the highest semantic relevance;" 

class Derm1M_Variants_Eval(Dataset):
    def __init__(
        self,
        dataset_name: str = "Derm1M-SimVariants",
        split: str = "validation",
        file_path: str | None = None,
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        if file_path is None:
            if self.split == "train":
                file_path = "/storage/dataset/dermatoscop/Derm1M/SimilarityVariationNeg_train.jsonl"
            elif self.split == "validation":
                file_path = "/storage/dataset/dermatoscop/Derm1M/SimilarityVariationNeg_eval.jsonl"
            elif self.split == "test":
                file_path = "/storage/dataset/dermatoscop/Derm1M/SimilarityVariationNeg_test.jsonl"

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading Derm1M-SimVariants data from {file_path}...")

        all_samples = []
        indices = []
        id_ = 0

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                all_samples.append(
                    DataSample(
                        id_=id_,
                        query=f"{DermPrompt}{self.separator}{obj['original']}",
                        positive=f"{self.separator}{obj['variants']}",
                        negative=f"{self.separator}{obj['negative']}",
                        task_name=self.dataset_name,
                    )
                )
                indices.append(id_)
                id_ += 1

        if self.shuffle_individual_datasets:
            random.shuffle(indices)

        logger.info(
            f"Batching Derm1M-SimVariants data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for i in range(0, len(indices), self.effective_batch_size):
            batch = indices[i : i + self.effective_batch_size]
            if len(batch) == self.effective_batch_size:
                all_batches.append(batch)
            else:
                logger.info("Skip 1 batch for dataset Derm1M-SimVariants.")
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

