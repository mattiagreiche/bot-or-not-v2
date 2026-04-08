# Bot-or-Not Challenge

Bot detection pipeline for the April McGill Bot-or-Not Challenge. Classifies Twitter/X users as bots or humans using behavioral features, semantic embeddings, and a stacking ensemble. Works for both English and French (and presumably other languages).

## Features

- **Semantic embeddings:** Tweets and bios embedded with `paraphrase-multilingual-MiniLM-L12-v2`. For tweets, both mean and std of embeddings across a user's posts are used; the std captures semantic consistency, which differs significantly between bots and humans.
- **Temporal rhythm:** Entropy of posting hours and day-of-week, plus inter-post gap variation (`delta_s_cv`). Humans and bots have detectably different posting patterns.
- **Vocabulary:** Type-token ratio, average word length, unique hashtag ratio. Bots reuse the same words and hashtags far more than humans do.
- **PCA:** Tweet and description embeddings compressed to 25 and 30 components to keep features topic-agnostic and avoid overfitting to specific domains in the training data.

## Model

Stacking ensemble: XGBoost + LightGBM + CatBoost base learners, with a Logistic Regression meta-learner trained on out-of-fold predictions. XGBoost is run on 3 seeds and averaged to reduce variance. Hyperparameters tuned with Optuna directly on the challenge profit metric.

Scoring is TP=+2, FP=−6, FN=−2, so `scale_pos_weight` is calibrated to ~1.8 (rather than the naive class imbalance ratio) to reflect that false positives are much more costly.

## Setup

```
pip install -r requirements.txt
```

## How to Run

1. **Build training features**
   `python build_training_features.py`

2. **Evaluate / tune threshold**
   `python cv_eval.py`
   5-fold CV with per-fold scores and SHAP importance across all three models. Saves beeswarm plots to `assets/`. Prints the recommended threshold at the end.

3. **Tune hyperparameters** (optional)
   `python tune.py --model all --trials 50`
   Prints best params to paste into `train_final.py`.

4. **Train final model**
   `python train_final.py`

5. **Build inference features**
   `python build_inference_features.py`
   Saves separate parquets per inference file → `user_features_7.parquet`, `user_features_8.parquet`.

6. **Predict**
   `python predict.py`
   Outputs `predicted_bot_ids_7.txt` and `predicted_bot_ids_8.txt`, plus a borderline cases CSV.

## Config

`botornot/config.py`:
- `USE_EMBEDDINGS` — set to `False` to skip embeddings (faster, worse)
- `TRAINING_POST_FILES` / `INFERENCE_POST_FILES` — input dataset file lists
