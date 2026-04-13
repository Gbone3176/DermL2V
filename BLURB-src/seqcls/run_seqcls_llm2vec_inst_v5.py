#!/usr/bin/env python
# coding=utf-8

import runpy

import llm2vec
from llm2vec.llm2vecV5 import LLM2Vec as LLM2VecV5


def main():
    llm2vec.LLM2Vec = LLM2VecV5
    runpy.run_module("BLURB-src.seqcls.run_seqcls_llm2vec_inst", run_name="__main__")


if __name__ == "__main__":
    main()
