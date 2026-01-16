from datasets import load_dataset
from transformers import AutoTokenizer

PROMPT_TEMPLATE = """### Instruction:
You are an internal IT support assistant.

### Issue:
{issue}

### Response:
{resolution}
"""

def load_and_prepare_data(path: str, model_name: str, max_length: int = 512):
    dataset = load_dataset("json", data_files=path)["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        text = PROMPT_TEMPLATE.format(
            issue=example["issue"], resolution=example["resolution"]
        )
        return tokenizer(
            text, truncation=True, padding="max_length", max_length=max_length
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    return tokenized
