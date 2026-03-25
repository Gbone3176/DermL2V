import json
import logging
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, seed_worker

from beir.retrieval.evaluation import EvaluateRetrieval

from peft import LoraConfig, get_peft_model

from llm2vec.llm2vecV1 import LLM2Vec
from llm2vec.dataset.utils import load_dataset
from llm2vec.loss.utils import list_available_losses, load_loss
from llm2vec.experiment_utils import generate_experiment_id

from tqdm import tqdm
import swanlab
from swanlab.integration.transformers import SwanLabCallback

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_STATE_NAME = "pytorch_model.bin"

def configure_torch_resume_loading() -> None:
    """
    Make checkpoint resume compatible with PyTorch>=2.6 defaults.
    transformers.Trainer loads RNG state with torch.load(...) and does not
    pass weights_only=False in some versions, which can fail on old checkpoints.
    """
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals(
        [
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
        ]
    )


def prepare_for_tokenization(model, text, pooling_mode="mean"):
    if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct" or isinstance(model.config, LlamaConfig):
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        )
        return text
    if model.config._name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if model.config._name_or_path in [
        "google/gemma-2-9b-it",
    ]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    if model.config._name_or_path in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen3-8B-Embedding",
    ]:
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    if pooling_mode == "eos_token":
        if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(
            model.config, MistralConfig
        ):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(model.config, Qwen2Config) or isinstance(model.config, Qwen3Config):
            text = text.strip() + "<|endoftext|>"
    return text


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_r <= 0:
        for _, p in model.named_parameters():
            p.requires_grad = False
        print("LoRA is disabled (lora_r <= 0). Encoder parameters are frozen; only outer modules such as latent_pooling remain trainable.")
        return model
    if lora_modules is None and (
        model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]
    ):  
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)

    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

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
    
    extra_model_name_or_path: Optional[List[str]] = field(default_factory=list, metadata={"help": "Path to extra Lora models"})

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
            "choices": ["mean", "weighted_mean", "eos_token", "latent_pooling", "structured_selfattn"],
        },
    )
    selfattn_attn_hidden_dim: int = field(
        default=512,
        metadata={"help": "Hidden size of the structured self-attention scoring MLP."},
    )
    selfattn_num_hops: int = field(
        default=8,
        metadata={"help": "Number of attention hops for structured self-attention pooling."},
    )
    selfattn_output_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout applied before projecting structured self-attention output back to hidden size."},
    )
    selfattn_output_layernorm: bool = field(
        default=True,
        metadata={"help": "Apply LayerNorm after structured self-attention output projection."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    # TODO: implement this
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    dermqa_upsample_ratio: int = field(
        default=10,
        metadata={
            "help": "Upsampling factor applied to DermQA samples inside DermVariants dataset during training."
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": f"The loss class to use for training. Options: {', '.join(list_available_losses())}"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )

    loss_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"help": "Extra keyword arguments passed to the selected loss class."},
    )


@dataclass
class EvalArguments:
    eval_batch_size: int = field(default=16, metadata={"help": "Eval batch size"})
    eval_top_k: int = field(default=10, metadata={"help": "Top-k for retrieval"})
    eval_separator: str = field(default="!@#$%^&*()", metadata={"help": "Separator used when joining variants"})


@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                text = prepare_for_tokenization(
                    self.model, text, pooling_mode=self.model.pooling_mode
                )
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


def _read_eval_rows(file_path: str, separator: str) -> List[Dict[str, str]]:
    rows = []
    if not os.path.exists(file_path):
        logger.warning(f"Eval data file not found: {file_path}")
        return rows
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "original" not in record or "variants" not in record:
                continue
            prompt = record.get("original")
            variants = record.get("variants")
            if prompt is None or variants is None:
                continue
            if isinstance(variants, list):
                response = separator.join([v for v in variants if isinstance(v, str)])
            else:
                response = str(variants)
            if response:
                rows.append({"prompt": str(prompt), "response": response})
    return rows


def _append_instruction(instruction: str, sentences: List[str]) -> List[List[Union[str, int]]]:
    return [[instruction, s, 0] for s in sentences]


def _cos_sim(a: torch.Tensor, b: torch.Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def _encode_texts(
    model: LLM2Vec,
    texts: List[str],
    batch_size: int,
    device: str,
    separator: str,
):
    # Require each sample to carry its own instruction/content split via separator to avoid accidental misuse.
    inputs: List[List[Union[str, int]]] = []
    for t in texts:
        if separator in t:
            inst, content = t.split(separator, 1)
            inst = inst.strip()
            content = content.strip()
        else:
            raise ValueError(
                "Expected text to contain separator for instruction/content; got: " + t
            )
        inputs.append([inst, content, 0])

    # Force single-device path to avoid multiprocessing in callbacks
    orig_device_count = torch.cuda.device_count
    torch.cuda.device_count = lambda: 1  # type: ignore
    try:
        return model.encode(
            inputs,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=device,
        )
    finally:
        torch.cuda.device_count = orig_device_count  # type: ignore


def run_retrieval_eval(
    model: LLM2Vec,
    eval_args: EvalArguments,
    device: str,
    eval_examples: Optional[List[Any]] = None,
    is_main_process: bool = True,
):
    if not is_main_process:
        return None

    if not eval_examples:
        logger.warning("Eval examples are empty; skip eval.")
        return None

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, Dict[str, int]] = {}

    for idx, sample in enumerate(eval_examples):
        if not hasattr(sample, "texts") or not sample.texts:
            continue
        pair_id = str(idx)
        try:
            query_text = sample.texts[0]
            pos_text = sample.texts[1] if len(sample.texts) > 1 else None
        except Exception:
            continue

        if pos_text is None:
            continue

        queries[pair_id] = query_text
        pos_id = f"{pair_id}_pos"
        corpus[pos_id] = {"text": pos_text}
        relevant_docs[pair_id] = {pos_id: 1}

    if not queries or not corpus:
        logger.warning("Eval corpus or queries empty after parsing; skip eval.")
        return None

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

    model.eval()
    with torch.no_grad():
        q_emb = _encode_texts(
            model,
            query_texts,
            eval_args.eval_batch_size,
            device,
            eval_args.eval_separator,
        )
        d_emb = _encode_texts(
            model,
            corpus_texts,
            eval_args.eval_batch_size,
            device,
            eval_args.eval_separator,
        )

    scores = _cos_sim(q_emb, d_emb)
    scores[torch.isnan(scores)] = -1
    top_k = min(eval_args.eval_top_k, len(corpus_ids))
    top_vals, top_idx = torch.topk(scores, top_k, dim=1, largest=True, sorted=True)
    top_vals = top_vals.cpu().tolist()
    top_idx = top_idx.cpu().tolist()

    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {}
        for rank, idx in enumerate(top_idx[i]):
            doc_id = corpus_ids[idx]
            score = top_vals[i][rank]
            results[qid][doc_id] = score

    retriever = EvaluateRetrieval(model, score_function="cos_sim")
    default_k_values = [1, 3, 5, 10, 100, 1000]
    k_values = [k for k in default_k_values if k <= top_k]
    if not k_values:
        k_values = [top_k]
    ndcg, _map, recall, precision = retriever.evaluate(
        relevant_docs, results, k_values, ignore_identical_ids=False
    )

    metrics = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }

    return metrics


class EvaluateAndLogCallback(TrainerCallback):
    def __init__(self, eval_args: EvalArguments, eval_examples: Optional[List[Any]] = None):
        self.eval_args = eval_args
        self.eval_examples = eval_examples

    def on_save(self, args, state, control, **kwargs):
        if not self.trainer.is_world_process_zero:
            return control

        model = self.trainer.model
        if hasattr(model, "module"):
            model = model.module

        device = str(next(model.parameters()).device)
        metrics = run_retrieval_eval(
            model,
            self.eval_args,
            device=device,
            eval_examples=self.eval_examples,
            is_main_process=self.trainer.is_world_process_zero(),
        )
        if metrics is None:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        metrics_path = os.path.join(ckpt_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        for k in [1, 3, 5, 10]:
            key = f"ndcg_at_{k}"
            if key in metrics:
                try:
                    swanlab.log({f"eval/ndcg@{k}": metrics[key]}, step=state.global_step)
                except Exception:
                    pass

        return control


class LLM2VecSupervisedTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_function=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = inputs
        if hasattr(self.model, "reset_pooling_aux_loss"):
            self.model.reset_pooling_aux_loss()
        q_reps = self.model(features[0])
        d_reps = self.model(features[1])

        d_reps_neg = None
        if len(features) > 2:
            d_reps_neg = self.model(features[2])

        loss_kwargs = {}
        if getattr(self.loss_function, "supports_aux_loss", False) and hasattr(
            self.model, "consume_pooling_aux_loss"
        ):
            loss_kwargs["aux_loss"] = self.model.consume_pooling_aux_loss(reset=True)
        elif hasattr(self.model, "consume_pooling_aux_loss"):
            self.model.consume_pooling_aux_loss(reset=True)

        loss = self.loss_function(q_reps, d_reps, d_reps_neg, **loss_kwargs)

        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output

        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)

        if prediction_loss_only:
            return loss.detach(), None, None

        features, labels = inputs
        return loss.detach(), None, labels

    def get_train_dataloader(self) -> DataLoader:
        # Copying most of the code from the parent class, changing the sampler to SequentialSampler
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # Changing from random sampler to sequential sampler
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)
        state_dict = state_dict if state_dict is not None else self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, MODEL_STATE_NAME))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments, EvalArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args, eval_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
            eval_args,
        ) = parser.parse_args_into_dataclasses()

    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    if accelerator.is_main_process:
        try:
            swanlab.init(
                project="LLM2Vec-supervised",
                name="_".join(training_args.output_dir.split("/")[-2:])
                if training_args.output_dir
                else None,
                config={
                    **vars(model_args),
                    **vars(data_args),
                    **training_args.to_dict(),
                    **vars(custom_args),
                },
            )
        except Exception:
            pass

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if hasattr(training_args, "save_only_model") and training_args.save_only_model:
        if accelerator.is_main_process:
            logger.info(
                "Detected save_only_model=True; overriding to False so checkpoints remain resumable."
            )
        training_args.save_only_model = False

    if custom_args.experiment_id is not None:
        experiment_id = custom_args.experiment_id
    else:
        experiment_id = generate_experiment_id(
            name=data_args.dataset_name,
            split="train",
            model_name=(
                model_args.model_name_or_path
                if "/" not in model_args.model_name_or_path
                else model_args.model_name_or_path.split("/")[-1]
            ),
            pooling_mode=model_args.pooling_mode,
            train_batch_size=training_args.per_device_train_batch_size
            * accelerator.num_processes
            * training_args.gradient_accumulation_steps,
            max_seq_length=model_args.max_seq_length,
            bidirectional=model_args.bidirectional,
            epochs=training_args.num_train_epochs,
            seed=training_args.seed,
            warmup_steps=training_args.warmup_steps,
            lr=training_args.learning_rate,
            lora_r=custom_args.lora_r,
        )

    training_args.output_dir = f"{training_args.output_dir}/{experiment_id}"

    last_checkpoint = None
    if training_args.do_train and os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (
            accelerator.is_main_process
            and last_checkpoint is not None
            and training_args.resume_from_checkpoint is None
        ):
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    train_dataset = load_dataset(
        data_args.dataset_name,
        split="train",
        file_path=data_args.dataset_file_path,
        effective_batch_size=training_args.per_device_train_batch_size
        * accelerator.num_processes,
        dermqa_upsample_ratio=data_args.dermqa_upsample_ratio,
    )

    train_examples = [
        train_dataset[i]
        for i in tqdm(
            range(len(train_dataset)),
            desc="Loading train examples...",
            disable=not accelerator.is_main_process,
        )
    ]

    eval_examples = None
    if training_args.do_eval:
        eval_dataset = load_dataset(
            data_args.dataset_name,
            split="validation",
            file_path=data_args.dataset_file_path,
            effective_batch_size=training_args.per_device_eval_batch_size
            * accelerator.num_processes,
            dermqa_upsample_ratio=1,
        )
        eval_examples = [
            eval_dataset[i]
            for i in tqdm(
                range(len(eval_dataset)),
                desc="Loading eval examples...",
                disable=not accelerator.is_main_process,
            )
        ]

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        extra_model_name_or_path=model_args.extra_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        selfattn_attn_hidden_dim=model_args.selfattn_attn_hidden_dim,
        selfattn_num_hops=model_args.selfattn_num_hops,
        selfattn_output_dropout=model_args.selfattn_output_dropout,
        selfattn_output_layernorm=model_args.selfattn_output_layernorm,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    # model organization is LLM2VecModel.model -> HF Model, we have to apply PEFT to the inner model
    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    tokenizer = model.tokenizer

    total_params = 0
    trainable_params = 0
    lora_trainable_params = 0
    latent_trainable_params = 0
    structured_selfattn_trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if not param.requires_grad:
            continue
        trainable_params += param.numel()
        if "lora_" in name:
            lora_trainable_params += param.numel()
        elif name.startswith("latent_attn."):
            latent_trainable_params += param.numel()
        elif name.startswith("structured_self_attn."):
            structured_selfattn_trainable_params += param.numel()
    print(
        f"Model trainable parameters: {trainable_params:,}, total parameters: {total_params:,}, trainable ratio: {100 * trainable_params / total_params:.4f}%"
    )
    print(
        "LoRA trainable parameters: "
        f"{lora_trainable_params:,}, latent_pooling trainable parameters: {latent_trainable_params:,}, "
        f"structured_selfattn trainable parameters: {structured_selfattn_trainable_params:,}"
    )


    train_loss = load_loss(
        custom_args.loss_class,
        scale=custom_args.loss_scale,
        **(custom_args.loss_kwargs or {}),
    )

    data_collator = DefaultCollator(model)

    swanlab_callback = SwanLabCallback()

    eval_callback = EvaluateAndLogCallback(eval_args, eval_examples)

    trainer = LLM2VecSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=eval_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=train_loss,
        callbacks=[swanlab_callback, eval_callback],
    )

    # Ensure callbacks needing trainer context can access it
    eval_callback.trainer = trainer

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    if checkpoint:
        configure_torch_resume_loading()
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()

    if training_args.do_predict:
        test_dataset = load_dataset(
            data_args.dataset_name,
            split="test",
            file_path=None,
            effective_batch_size=training_args.per_device_eval_batch_size
            * accelerator.num_processes,
            dermqa_upsample_ratio=1,
        )
        test_examples = [
            test_dataset[i]
            for i in tqdm(
                range(len(test_dataset)),
                desc="Loading test examples...",
                disable=not accelerator.is_main_process,
            )
        ]
        metrics = trainer.evaluate(eval_dataset=test_examples)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
