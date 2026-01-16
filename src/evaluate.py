import torch
from transformers import AutoTokenizer
from src.model import load_qlora_model
from src.config import load_config

def evaluate_issue(issue: str):
    cfg = load_config()
    model = load_qlora_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model.eval()

    prompt = f"""
### Instruction:
You are an internal IT support assistant.

### Issue:
{issue}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    evaluate_issue("VPN disconnects every few minutes")
