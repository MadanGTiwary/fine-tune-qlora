import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def load_qlora_model(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        device_map="auto",
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=cfg["model"]["lora_r"],
        lora_alpha=cfg["model"]["lora_alpha"],
        target_modules=cfg["model"]["target_modules"],
        lora_dropout=cfg["model"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, lora_config)
