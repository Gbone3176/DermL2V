import argparse
import torch
from pathlib import Path

def compare_pt_files(path_a: Path, path_b: Path):
    obj_a = torch.load(path_a, map_location="cpu")
    obj_b = torch.load(path_b, map_location="cpu")

    diffs: list[str] = []

    def collect_keys(obj, prefix=""):
        keys = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                keys.append(p)
                keys.extend(collect_keys(v, p))
        elif isinstance(obj, (list, tuple)):
            for idx, v in enumerate(obj):
                p = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                keys.append(p)
                keys.extend(collect_keys(v, p))
        else:
            if prefix:
                keys.append(prefix)
        return keys

    def _compare(x, y, prefix: str = "") -> bool:
        # Tensor: exact match or numerically close
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if torch.equal(x, y) or torch.allclose(x, y):
                return True
            diffs.append(prefix or "<tensor>")
            return False

        # Dict: same keys and all values equal
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x.keys()) != set(y.keys()):
                diffs.append(f"{prefix or '<root>'}: key mismatch -> {set(x.keys()) ^ set(y.keys())}")
                return False
            ok = True
            for k in x.keys():
                if not _compare(x[k], y[k], f"{prefix}.{k}" if prefix else str(k)):
                    ok = False
            return ok

        # List/tuple: same length and pairwise equal
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                diffs.append(f"{prefix or '<list>'}: length {len(x)} vs {len(y)}")
                return False
            ok = True
            for idx, (a, b) in enumerate(zip(x, y)):
                if not _compare(a, b, f"{prefix}[{idx}]"):
                    ok = False
            return ok

        # Fallback: type must match and Python equality must hold
        if type(x) is not type(y):
            diffs.append(f"{prefix or '<root>'}: type {type(x)} vs {type(y)}")
            return False
        try:
            if x == y:
                return True
            diffs.append(prefix or '<root>')
            return False
        except Exception:
            diffs.append(prefix or '<root>')
            return False

    equal = _compare(obj_a, obj_b)
    all_keys = sorted(set(collect_keys(obj_a) + collect_keys(obj_b)))
    return equal, diffs, all_keys, obj_a


def count_params(obj) -> int:
    """Recursively count total number of scalar parameters in the object."""
    if isinstance(obj, torch.Tensor):
        return obj.numel()
    if isinstance(obj, dict):
        return sum(count_params(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(count_params(v) for v in obj)
    # Non-tensor leaves count as 0
    return 0


def main():
    file_a = "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec/output/mntp-simcse/LLM2Vec-Llama-3.1-8B-Instruct-debug/Derm1M/based-on-originalLLM-mntp-1950/checkpoint-50/latent_attn.pt"
    file_b = "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec/output/mntp-simcse/LLM2Vec-Llama-3.1-8B-Instruct-debug/Derm1M/based-on-originalLLM-mntp-1950/checkpoint-100/latent_attn.pt"
    identical, diffs, all_keys, obj_a = compare_pt_files(file_a, file_b)

    total_params = count_params(obj_a)
    print(f"参数总量: {total_params}")

    print("全部键/路径 (并集):")
    for k in all_keys:
        print(" -", k)

    if identical:
        print("内容相同")
    else:
        print("内容不同，差异位置:")
        for d in diffs:
            print(" -", d)


if __name__ == "__main__":
    main()