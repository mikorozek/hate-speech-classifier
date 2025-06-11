from pathlib import Path
from src.config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_from_disk
import torch
import numpy as np
import os


def tokenize_and_save_prediction_data(tokenizer):
    file_path = Path(Config.PREDICTION_DATA)
    if not file_path.exists():
        raise FileNotFoundError(f"Prediction file data does not exist: {file_path}")

    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=Config.MAX_LENGTH,
    )

    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataset.save_to_disk(Config.DATASET_PATHS["prediction"])
    return dataset


def load_or_create_prediction_data(tokenizer):
    if os.path.exists(Config.DATASET_PATHS["train"]) and os.path.exists(
        Config.DATASET_PATHS["validation"]
    ):
        try:
            prediction_data = load_from_disk(Config.DATASET_PATHS["prediction"])
        except Exception as e:
            print(f"Error while reading prediction data: {e}")
            print(f"Beggining tokenization of prediction data")
            prediction_data = tokenize_and_save_prediction_data(tokenizer)
    else:
        print(f"Tokenized data does not exist. Beggining tokenization")
        prediction_data = tokenize_and_save_prediction_data(tokenizer)
    return prediction_data


def load_model_from_checkpoint(checkpoint_path, num_labels):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    if not (checkpoint_path / "model.safetensors").exists():
        raise FileNotFoundError(f"No model safetensors in: {checkpoint_path}")

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path, num_labels=num_labels
    )

    return model


def predict_batch(model, dataset, device, batch_size=32):
    model.eval()
    logits = []
    model = model.to(device)

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            logits.extend(outputs.logits.cpu().tolist())

    return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    prediction_dataset = load_or_create_prediction_data(tokenizer)
    model = load_model_from_checkpoint(Config.MODEL_CHECKPOINT_PATH, 2)
    logits = predict_batch(model, prediction_dataset, device)
    predictions = np.argmax(logits, axis=1)
    np.savetxt(Config.OUTPUT_PREDICTION_PATH, predictions, fmt="%d", delimiter=",")
    print(f"Predictions saved to {Config.OUTPUT_PREDICTION_PATH}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
