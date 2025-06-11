class Config:
    WANDB_PROJECT = "hate-speech-classification"
    TOKENIZER_NAME = "allegro/herbert-klej-cased-tokenizer-v1"
    MODEL_NAME = "allegro/herbert-klej-cased-v1"
    MAX_LENGTH = 512
    VAL_SIZE = 0.25

    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 15
    SAVE_STEPS = 500
    WEIGHT_DECAY = 0.01

    OUTPUT_DIR = "results"
    TRAIN_DATASET_PATH = "data/hate_train.csv"
    BEST_MODEL_PATH = "/models"
    PREDICTION_DATA = "data/hate_test_data.txt"
    DATASET_PATHS = {
        "train": "data/train_tokenized",
        "validation": "data/val_tokenized",
        "prediciton": "data/pred_tokenized",
    }
    REQUIRED_COLUMNS = ["sentence", "label"]
