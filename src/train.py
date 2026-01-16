from transformers import Trainer, TrainingArguments
from src.data import load_and_prepare_data
from src.model import load_qlora_model
from src.config import load_config
from src.utils import seed_everything, ensure_dir

def main():
    cfg = load_config()
    seed_everything(42)
    ensure_dir("outputs/checkpoints")

    model = load_qlora_model(cfg)
    dataset = load_and_prepare_data(
        cfg["data"]["train_path"], cfg["model"]["name"], cfg["data"]["max_length"]
    )

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        fp16=cfg["training"]["fp16"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        report_to=cfg["training"]["report_to"],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

if __name__ == "__main__":
    main()
