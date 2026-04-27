import importlib.util
from pathlib import Path

from llm2vec.llm2vecV5 import LLM2Vec as LLM2VecV5


def _load_base_module():
    script_path = Path(__file__).with_name("nonhomo_RT_l2v.py")
    spec = importlib.util.spec_from_file_location(
        "nonhomo_RT_l2v_base",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load base script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = _load_base_module()
    module.LLM2Vec = LLM2VecV5
    module.main()


if __name__ == "__main__":
    main()
