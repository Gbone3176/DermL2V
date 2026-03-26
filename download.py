from huggingface_hub import snapshot_download

if __name__ == "__main__":
    snapshot_download(
        repo_id="McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse",
        allow_patterns=["*"],
        local_dir_use_symlinks=False,
        max_workers=8
    )