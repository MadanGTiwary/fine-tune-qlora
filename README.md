# QLoRA Enterprise IT Support Assistant 🚀

This project fine-tunes a large language model using **QLoRA (4-bit quantization + LoRA adapters)** to function as an **enterprise IT support assistant**.

### 💡 Why QLoRA?
- Enables fine-tuning 7B+ models on a single GPU  
- Dramatically reduces VRAM usage  
- Only LoRA adapters are updated (efficient training)

### 🧠 Use Case
Trains on IT ticket data — the model learns to generate accurate, context-aware resolutions for issues like:
- VPN / Access problems  
- CI/CD failures  
- Cloud infra troubleshooting  

---

### 🧰 Tech Stack
- PyTorch ⚙️  
- Hugging Face Transformers 🤗  
- PEFT (LoRA / QLoRA)  
- bitsandbytes (quantization)  
- Weights & Biases (tracking)

---

### ⚙️ Training
```bash
bash scripts/run_train.sh
