from transformers import CLIPTokenizer, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
text = "a long text input..." * 100  # 超长文本
tokens = tokenizer(text, truncation=True, max_length=77)
print(len(tokens.input_ids))  # 应输出77