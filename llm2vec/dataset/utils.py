from ..dataset import E5Data
from ..dataset import Wiki1M
from ..dataset import Derm1M
from ..dataset import Derm1M_SimVariants
from ..dataset import Derm1M_Variants_Eval
from ..dataset import DermQA
from ..dataset import DermVariants

def load_dataset(dataset_name, split="validation", file_path=None, **kwargs):
    """
    Loads a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Split of the dataset to load.
        file_path (str): Path to the dataset file.
    """
    dataset_mapping = {
        "E5": E5Data,
        "Wiki1M": Wiki1M,
        "Derm1M": Derm1M,
        "Derm1M_SimVariants": Derm1M_SimVariants,
        "Derm1M_Variants_Eval": Derm1M_Variants_Eval,
        "DermQA":DermQA,
        "DermVariants": DermVariants,
    }

    if dataset_name not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")

    if split not in ["train", "validation", "test"]:
        raise NotImplementedError(f"Split {split} not supported.")

    return dataset_mapping[dataset_name](split=split, file_path=file_path, **kwargs)
