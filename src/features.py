from datasets import load_dataset
import os
import numpy as np
from datasets import load_from_disk
from transformers.tokenization_utils import PreTrainedTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.config import Config


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def tokenize_and_save_datasets(tokenize_function):
    dataset = load_dataset("csv", data_files=Config.TRAIN_DATASET_PATH)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset["train"]
    dataset = dataset.train_test_split(
        test_size=Config.VAL_SIZE,
        seed=42,
        stratify_by_column="label" if "label" in dataset.column_names else None,
    )

    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    val_dataset = dataset["test"].map(tokenize_function, batched=True)
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    train_dataset.save_to_disk(Config.DATASET_PATHS["train"])
    val_dataset.save_to_disk(Config.DATASET_PATHS["validation"])
    return train_dataset, val_dataset


def load_or_create_datasets(tokenizer: PreTrainedTokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
        )

    if os.path.exists(Config.DATASET_PATHS["train"]) and os.path.exists(
        Config.DATASET_PATHS["validation"]
    ):
        try:
            train_dataset = load_from_disk(Config.DATASET_PATHS["train"])
            val_dataset = load_from_disk(Config.DATASET_PATHS["validation"])
        except Exception as e:
            print(f"Error while reading datasets: {e}")
            print(f"Beggining tokenization of datasets")
            train_dataset, val_dataset = tokenize_and_save_datasets(tokenize_function)
    else:
        print(f"One of the tokenized datasets does not exist. Beggining tokenization")
        train_dataset, val_dataset = tokenize_and_save_datasets(tokenize_function)

    return train_dataset, val_dataset
