import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# directories
DIR_RAW = os.path.join(BASE_DIR, "raw_data") # from bot-or-not challenge
DIR_TRAINING = os.path.join(BASE_DIR, "training_data")
DIR_INFERENCE = os.path.join(BASE_DIR, "inference_data")
DIR_MODELS = os.path.join(BASE_DIR, "models")

# feature building
# note I was in a bit of a rush, so it's entirely possible using USE_EMBEDDINGS=False right now will result
# in all kinds of errors. (just use it, it's the whole interesting part of this model!)
USE_EMBEDDINGS = True

# training data file lists (paths under DIR_RAW)
TRAINING_POST_FILES = [
    os.path.join(DIR_RAW, "dataset.posts&users.1.json"),
    os.path.join(DIR_RAW, "dataset.posts&users.2.json"),
    os.path.join(DIR_RAW, "dataset.posts&users.3.json"),
    os.path.join(DIR_RAW, "dataset.posts&users.4.json"),
    os.path.join(DIR_RAW, "dataset.posts&users.5.json"),
    os.path.join(DIR_RAW, "dataset.posts&users.6.json"),
]
TRAINING_BOT_FILES = [
    os.path.join(DIR_RAW, "dataset.bots.1.txt"),
    os.path.join(DIR_RAW, "dataset.bots.2.txt"),
    os.path.join(DIR_RAW, "dataset.bots.3.txt"),
    os.path.join(DIR_RAW, "dataset.bots.4.txt"),
    os.path.join(DIR_RAW, "dataset.bots.5.txt"),
    os.path.join(DIR_RAW, "dataset.bots.6.txt"),
]

# inference data file list
INFERENCE_POST_FILES = [
    os.path.join(DIR_INFERENCE, "dataset.posts&users.7.json"), # english
    os.path.join(DIR_INFERENCE, "dataset.posts&users.8.json"), # french
]

# derived paths: training outputs
if USE_EMBEDDINGS:
    TRAINING_PARQUET_PATH = os.path.join(DIR_TRAINING, "user_features.parquet")
else:
    TRAINING_PARQUET_PATH = os.path.join(DIR_TRAINING, "user_features_no_emb.parquet")

def inference_paths(json_file):
    """Return (parquet_path, predictions_path) for a given inference JSON file."""
    num = os.path.basename(json_file).split('.')[-2]
    suffix = "" if USE_EMBEDDINGS else "_no_emb"
    parquet = os.path.join(DIR_INFERENCE, f"user_features_{num}{suffix}.parquet")
    preds   = os.path.join(DIR_INFERENCE, f"predicted_bot_ids_{num}.txt")
    return parquet, preds

# model artifact
ARTIFACT_PATH = os.path.join(DIR_MODELS, "bot_detector.pkl")
