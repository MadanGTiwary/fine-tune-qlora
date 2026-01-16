from src.data import load_and_prepare_data
from src.config import load_config

def test_data_load():
    cfg = load_config()
    dataset = load_and_prepare_data(
        cfg["data"]["train_path"], cfg["model"]["name"], cfg["data"]["max_length"]
    )
    assert len(dataset) > 0
