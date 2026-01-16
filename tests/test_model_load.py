from src.model import load_qlora_model
from src.config import load_config

def test_model_load():
    cfg = load_config()
    model = load_qlora_model(cfg)
    assert model is not None
