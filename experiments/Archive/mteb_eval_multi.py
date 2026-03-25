import argparse
import json
import torch
import numpy as np
import mteb

from llm2vec import LLM2Vec

def llm2vec_instruction(instruction):  # 末尾补一个冒号
    if len(instruction) > 0 and instruction[-1] != ":":
        instruction = instruction.strip(".") + ":"
    return instruction

class LLM2VecWrapper:
    def __init__(self, model=None, task_to_instructions=None):
        self.task_to_instructions = task_to_instructions or {}
        self.model = model

    def encode(self, sentences: list[str], *, prompt_name: str = None, **kwargs):
        if prompt_name is not None:
            instruction = self.task_to_instructions.get(prompt_name, "")
            instruction = llm2vec_instruction(instruction) if instruction else ""
        else:
            instruction = ""
        sentences = [[instruction, s] for s in sentences]
        return self.model.encode(sentences, **kwargs)

    def encode_corpus(self, corpus, prompt_name: str = None, **kwargs):
        # 兼容常见 corpus 格式：list[str] 或 list[dict]
        def to_texts(c):
            if isinstance(c, list):
                if len(c) > 0 and isinstance(c[0], dict):
                    return [" ".join(v for v in item.values() if isinstance(v, str)) for item in c]
                elif len(c) > 0 and isinstance(c[0], str):
                    return c
            elif isinstance(c, dict):
                # 合并所有字段
                vals = []
                for v in c.values():
                    if isinstance(v, list):
                        vals.extend([x for x in v if isinstance(x, str)])
                    elif isinstance(v, str):
                        vals.append(v)
                return [" ".join(vals)]
            return c

        sentences = to_texts(corpus)
        sentences = [["", s] for s in sentences]
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        return self.model.encode(sentences, **kwargs)

    def encode_queries(self, queries: list[str], **kwargs):
        return self.encode(queries, **kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_model_name_or_path", type=str, default=None)
    parser.add_argument("--task_names", type=str, nargs="+", required=True)
    parser.add_argument("--task_to_instructions_fp", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    task_to_instructions = None
    if args.task_to_instructions_fp:
        with open(args.task_to_instructions_fp, "r") as f:
            task_to_instructions = json.load(f)

    l2v_model = LLM2Vec.from_pretrained(
        args.base_model_name_or_path,
        peft_model_name_or_path=args.peft_model_name_or_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    model = LLM2VecWrapper(model=l2v_model, task_to_instructions=task_to_instructions)

    tasks = mteb.get_tasks(tasks=args.task_names)
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, output_folder=args.output_dir)

if __name__ == "__main__":
    main()