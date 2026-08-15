# DermL2V Inference Data Manifest

Prepared on 2026-07-18 for the standalone `DermL2V` inference repository.

The RT-full test files were copied from the local benchmark root and renamed
using the abbreviations recorded in `local_info/local_path.md`.

| Local file | Records | Source |
|---|---:|---|
| `data/DermaSynth-E3.jsonl` | 1998 | Local release copy |
| `data/MedMCQA.jsonl` | 1788 | Local release copy |
| `data/MedQuAD.jsonl` | 5092 | Local release copy |
| `data/SCE-Derma-SQ.jsonl` | 100 | Local release copy |

`infer_derml2v.py --eval_rt_full` uses these local files by default.
