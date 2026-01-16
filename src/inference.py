import torch
from transformers import AutoTokenizer
from src.model import load_qlora_model
from src.config import load_config

def run_inference(issue: str):
    cfg = load_config()
    model = load_qlora_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    prompt = f"""
### Instruction:
You are an internal IT support assistant.

### Issue:
{issue}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=120)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    run_inference("Kubernetes pod stuck in CrashLoopBackOff")
