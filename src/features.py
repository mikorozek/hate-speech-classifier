from datasets import load_dataset
import os
import numpy as np
from datasets import load_from_disk, ClassLabel
from transformers.tokenization_utils import PreTrainedTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.config import Config


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = np.sum(predictions == labels) / len(labels)

    classes = np.unique(labels)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for c in classes:
        tp = np.sum((predictions == c) & (labels == c))
        tn = np.sum((predictions != c) & (labels != c))
        fp = np.sum((predictions == c) & (labels != c))
        fn = np.sum((predictions != c) & (labels == c))

        accuracy = tp + tn / (tp + fn + fp + fn) if (tp + fn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "f1": np.mean(f1s),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
    }


def tokenize_and_save_datasets(tokenize_function):
    dataset = load_dataset("csv", data_files=Config.TRAIN_DATASET_PATH)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset["train"]
    dataset = dataset.cast_column("label", ClassLabel(names=[0, 1]))
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
