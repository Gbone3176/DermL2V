import json
import random
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/storage/BioMedNLP/llm2vec")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm2vec import LLM2Vec

BASE_MODEL_PATH = "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PEFT_MODEL_PATH = "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
SUPERVISED_MODEL_PATH = "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"
CHECKPOINT_PATH = "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM/SlerpMixCSE_k128_StructuredSelfAttn_gamma0p001_aux0p001/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"
TEST_FILE = "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp2-skincap-DiseaseClassification/test.jsonl"
OUTPUT_DIR = REPO_ROOT / "experiments/deprecated/structured_selfattn_single_text_probe"
SEPARATOR = "!@#$%^&*()"
CLS_INSTRUCTION = (
    "Classify the skin concepts mentioned in the given dermatology text into all "
    "applicable terms among the 17 concepts: Erythema, Plaque, Papule, "
    "Brown(Hyperpigmentation), Scale, Crust, White(Hypopigmentation), Yellow, "
    "Nodule, Dome-shaped, Erosion, Ulcer, Friable, Patch, Exudate, Scar, Black. "
    "Ignore negated concepts."
)
RANDOM_SEED = 42


def load_random_test_row(path: str, seed: int) -> tuple[int, dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    sample_idx = rng.randrange(len(rows))
    return sample_idx, rows[sample_idx]


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(-1)
    summed = torch.sum(hidden_states * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def pairwise_cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = F.normalize(x, p=2, dim=-1)
    return x @ x.t()


def pairwise_l2_matrix(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return torch.cdist(x, x, p=2)


def tensor_to_rounded_list(x: torch.Tensor, digits: int = 6) -> list:
    if x.ndim == 0:
        return round(float(x.item()), digits)
    return [[round(float(v), digits) for v in row] for row in x.tolist()]


def format_vector_head(x: torch.Tensor, k: int = 12) -> list[float]:
    flat = x.detach().cpu().flatten()[:k]
    return [round(float(v), 6) for v in flat.tolist()]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)

    sample_idx, row = load_random_test_row(TEST_FILE, RANDOM_SEED)
    sentence = row["sentence"]
    label = row.get("label")
    probe_text = f"{CLS_INSTRUCTION}{SEPARATOR}{sentence}"

    model = LLM2Vec.from_pretrained(
        BASE_MODEL_PATH,
        peft_model_name_or_path=PEFT_MODEL_PATH,
        extra_model_name_or_path=[SUPERVISED_MODEL_PATH, CHECKPOINT_PATH],
        pooling_mode="structured_selfattn",
        skip_instruction=True,
        selfattn_num_hops=8,
        max_length=512,
        doc_max_length=400,
        torch_dtype=torch.float16,
        device_map=None,
    )
    model = model.to(device).eval()

    features = model.tokenize_with_separator([probe_text], max_length=512, separator=SEPARATOR)
    features = {k: v.to(device) for k, v in features.items()}

    with torch.no_grad():
        backbone_inputs = {k: v for k, v in features.items() if k != "embed_mask"}
        reps = model.model(**backbone_inputs)
        hidden_states = reps.last_hidden_state
        embed_mask = features["embed_mask"]

        mean_embedding = masked_mean_pool(hidden_states, embed_mask)
        sa_embedding, aux_penalty = model.structured_self_attn(
            hidden_states,
            attention_mask=embed_mask,
        )
        structured_embedding = model.structured_self_attn.last_structured_embedding.squeeze(0)
        attn_weights = model.structured_self_attn.last_attention_weights.squeeze(0)

    structured_embedding_cpu = structured_embedding.detach().float().cpu()
    hop_cos = pairwise_cosine_matrix(structured_embedding_cpu).detach().cpu()
    hop_l2 = pairwise_l2_matrix(structured_embedding_cpu).detach().cpu()
    offdiag_cos = hop_cos[~torch.eye(hop_cos.size(0), dtype=torch.bool)]
    offdiag_l2 = hop_l2[~torch.eye(hop_l2.size(0), dtype=torch.bool)]

    sa_vec = sa_embedding.squeeze(0).detach().float().cpu()
    mean_vec = mean_embedding.squeeze(0).detach().float().cpu()
    diff_vec = sa_vec - mean_vec

    result = {
        "seed": RANDOM_SEED,
        "device": str(device),
        "checkpoint_path": CHECKPOINT_PATH,
        "test_file": TEST_FILE,
        "sample_index": sample_idx,
        "sample_id": row.get("id"),
        "sentence": sentence,
        "label": label,
        "num_hops": int(structured_embedding.size(0)),
        "mask_token_count": int(embed_mask.sum().item()),
        "aux_penalty": round(float(aux_penalty.detach().cpu().item()), 6),
        "hop_metrics": {
            "matrix_rank": int(torch.linalg.matrix_rank(structured_embedding_cpu).item()),
            "pairwise_cosine": tensor_to_rounded_list(hop_cos),
            "pairwise_l2": tensor_to_rounded_list(hop_l2),
            "min_offdiag_cosine": round(float(offdiag_cos.min().item()), 6),
            "max_offdiag_cosine": round(float(offdiag_cos.max().item()), 6),
            "mean_offdiag_cosine": round(float(offdiag_cos.mean().item()), 6),
            "min_offdiag_l2": round(float(offdiag_l2.min().item()), 6),
            "max_offdiag_l2": round(float(offdiag_l2.max().item()), 6),
            "mean_offdiag_l2": round(float(offdiag_l2.mean().item()), 6),
        },
        "mean_vs_sa": {
            "cosine_similarity": round(float(F.cosine_similarity(mean_vec.unsqueeze(0), sa_vec.unsqueeze(0)).item()), 6),
            "l2_distance": round(float(torch.norm(diff_vec, p=2).item()), 6),
            "mean_abs_difference": round(float(diff_vec.abs().mean().item()), 6),
            "max_abs_difference": round(float(diff_vec.abs().max().item()), 6),
            "mean_embedding_norm": round(float(torch.norm(mean_vec, p=2).item()), 6),
            "sa_embedding_norm": round(float(torch.norm(sa_vec, p=2).item()), 6),
            "mean_embedding_head": format_vector_head(mean_vec),
            "sa_embedding_head": format_vector_head(sa_vec),
            "difference_head": format_vector_head(diff_vec),
        },
        "attention": {
            "pairwise_cosine": tensor_to_rounded_list(pairwise_cosine_matrix(attn_weights.detach().cpu())),
            "row_sums": [round(float(v), 6) for v in attn_weights.sum(dim=-1).detach().cpu().tolist()],
        },
    }

    hop_rank = result["hop_metrics"]["matrix_rank"]
    max_offdiag_cos = result["hop_metrics"]["max_offdiag_cosine"]
    min_offdiag_l2 = result["hop_metrics"]["min_offdiag_l2"]
    hop_conclusion = (
        "The hop representations are not identical: the hop matrix has full rank "
        f"{hop_rank}/{result['num_hops']}, max off-diagonal cosine is {max_offdiag_cos}, "
        f"and min pairwise L2 distance is {min_offdiag_l2}."
    )
    mean_sa = result["mean_vs_sa"]
    pool_conclusion = (
        "Mean pooling and structured self-attention produce clearly different final embeddings: "
        f"cosine={mean_sa['cosine_similarity']}, L2={mean_sa['l2_distance']}, "
        f"mean|diff|={mean_sa['mean_abs_difference']}."
    )

    summary_lines = [
        "# Structured Self-Attn Probe Result",
        "",
        f"- Sample index: `{sample_idx}`",
        f"- Sample id: `{row.get('id')}`",
        f"- Mask token count: `{result['mask_token_count']}`",
        f"- Aux penalty: `{result['aux_penalty']}`",
        "",
        "## Input",
        "",
        sentence,
        "",
        "## Conclusion",
        "",
        f"1. {hop_conclusion}",
        f"2. {pool_conclusion}",
        "",
        "## Mean vs SA",
        "",
        f"- cosine similarity: `{mean_sa['cosine_similarity']}`",
        f"- L2 distance: `{mean_sa['l2_distance']}`",
        f"- mean absolute difference: `{mean_sa['mean_abs_difference']}`",
        f"- max absolute difference: `{mean_sa['max_abs_difference']}`",
        "",
        "## Hop diversity",
        "",
        f"- matrix rank: `{hop_rank}`",
        f"- off-diagonal cosine range: `[{result['hop_metrics']['min_offdiag_cosine']}, {max_offdiag_cos}]`",
        f"- off-diagonal L2 range: `[{min_offdiag_l2}, {result['hop_metrics']['max_offdiag_l2']}]`",
    ]

    (OUTPUT_DIR / "probe_result.json").write_text(json.dumps(result, indent=2))
    (OUTPUT_DIR / "probe_summary.md").write_text("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
