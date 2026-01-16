from src.inference import run_inference

def test_inference_runs():
    run_inference("Test network issue")
    assert True
