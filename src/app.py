import streamlit as st
import torch
from transformers import AutoTokenizer
from src.config import load_config
from src.model import load_qlora_model

@st.cache_resource
def load_model_and_tokenizer():
    cfg = load_config()
    model = load_qlora_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    return model, tokenizer

def generate_response(model, tokenizer, issue: str):
    prompt = f"""
### Instruction:
You are an internal IT support assistant.

### Issue:
{issue}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Enterprise IT Support Assistant", page_icon="💻")

st.title("💻 Enterprise IT Support Assistant (QLoRA)")
st.markdown(
    "This assistant has been fine-tuned with **QLoRA** to handle common enterprise IT support issues. "
    "It runs fully locally using your GPU (if available)."
)

with st.sidebar:
    st.header("Settings ⚙️")
    st.write("You can configure model or response parameters here.")
    temperature = st.slider("Response Temperature", 0.1, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max Response Tokens", 50, 512, 200, 50)

st.divider()
issue_input = st.text_area("📝 Describe your IT issue:", height=120, placeholder="e.g., VPN disconnects every few minutes...")

if st.button("Get Solution 🚀"):
    if not issue_input.strip():
        st.warning("Please describe an IT issue before submitting.")
    else:
        with st.spinner("Generating response... please wait ⏳"):
            model, tokenizer = load_model_and_tokenizer()
            response = generate_response(model, tokenizer, issue_input)
            st.success("✅ Solution Generated!")
            st.write(response)
