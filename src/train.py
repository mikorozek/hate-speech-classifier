from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import wandb
from src.config import Config
from src.features import load_or_create_datasets, compute_metrics


def main():
    wandb.init(
        project=Config.WANDB_PROJECT,
        config={
            "model_name": Config.MODEL_NAME,
            "max_length": Config.MAX_LENGTH,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "num_epochs": Config.NUM_EPOCHS,
            "warmup_steps": Config.WARMUP_STEPS,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    )

    print(f"Using model: {Config.MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    train_dataset, val_dataset = load_or_create_datasets(tokenizer)
    num_labels = len(set(train_dataset["label"]))

    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=num_labels
    )

    print(f"Final train dataset size: {len(train_dataset)}")
    print(f"Steps per epoch should be: {len(train_dataset) // Config.BATCH_SIZE}")

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_steps=Config.SAVE_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        report_to="wandb",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(Config.BEST_MODEL_PATH)

    wandb.finish()


if __name__ == "__main__":
    main()
