from datasets import load_dataset
import re
from transformers import AutoTokenizer

def get_tokenized_datasets(model_name_or_path, max_input_length, max_target_length, train_batch_size, eval_batch_size):
    ds = load_dataset("cnn_dailymail", "3.0.0")

    def clean_text(example):
        article = example["article"]
        article = re.sub(r"<.*?>", "", article).strip()
        example["article"] = article
        
        highlights = example["highlights"]
        highlights = re.sub(r"<.*?>", "", highlights).strip()
        example["highlights"] = highlights
        return example

    ds = ds.map(clean_text, remove_columns=["id"])

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize_fn(ex):
        inputs = tokenizer(ex["article"], truncation=True,
                        max_length=max_input_length, padding="max_length")
        targets = tokenizer(ex["highlights"], truncation=True,
                            max_length=max_target_length, padding="max_length")
        ex["input_ids"]   = inputs.input_ids
        ex["attention_mask"] = inputs.attention_mask
        ex["labels"]      = targets.input_ids
        return ex

    ds = ds.map(tokenize_fn, remove_columns=ds["train"].column_names, batched=True)
    ds.set_format(type="torch")
    
    return ds, tokenizer

