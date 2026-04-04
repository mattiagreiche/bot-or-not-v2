# Bot-or-Not Challenge

A bot detection pipeline built for the McGill Bot-or-Not Challenge. Works for both English and French datasets.

## How it works

**Features**

- **Semantic embeddings:** Tweets and user descriptions are embedded with `paraphrase-multilingual-MiniLM-L12-v2` (supports French + English). For tweets, I compute the mean and std across a user's posts — the std matters a lot since bots tend to be repetitive while humans are all over the place.
- **Temporal rhythm:** Posting entropy by hour/day-of-week, and coefficient of variation of inter-post gaps (`delta_s_cv`). Bots often post on a rigid schedule.
- **Vocabulary:** Type-token ratio, average word length, unique hashtag ratio.
- **Twitter identity lookup:** Cross-references usernames against the real Twitter/X API via getxapi.com. Produces `description_partial_match`, `identity_score`, `confirmed_real`, etc. This turned out to be the most important feature by SHAP.
- **PCA:** Compresses 768-dim tweet and description embeddings down to 25 and 30 components respectively, to avoid overfitting to specific topics.

**Model**

Stacking ensemble: XGBoost + LightGBM + CatBoost as base learners, with a Logistic Regression meta-learner trained on out-of-fold predictions. Three XGBoost models with different random seeds are averaged to reduce variance. Hyperparameters tuned with Optuna directly on the challenge profit metric.

Scoring: TP=+2, FP=-6, FN=-2, TN=0. False positives are penalized heavily, so `scale_pos_weight` is calibrated accordingly (~1.8 rather than the naive class imbalance ratio).

## How to Run

1. **Twitter lookup** (optional but very helpful)
   `python twitter_lookup.py`
   Fetches live account data for all usernames and saves to `raw_data/twitter_lookup.csv`. Resumable — skips already-processed users.

2. **Build training features**
   `python build_training_features.py`

3. **Evaluate / tune threshold**
   `python cv_eval.py`
   Runs 5-fold CV, prints per-fold results, SHAP feature importance for all three models, and bot vs. human feature comparison.

4. **Tune hyperparameters** (optional)
   `python tune.py --model all --trials 50`
   Pastes best params into `train_final.py` manually after.

5. **Train final model**
   `python train_final.py`

6. **Build inference features**
   `python build_inference_features.py`
   Processes each inference file separately → `inference_data/user_features_7.parquet`, `user_features_8.parquet`.

7. **Predict**
   `python predict.py`
   Outputs `predicted_bot_ids_7.txt` and `predicted_bot_ids_8.txt` separately, plus a borderline cases CSV for manual review.

## Config

Everything is in `botornot/config.py`. Key things to know:
- `USE_EMBEDDINGS` — set to `False` to skip embeddings (faster, worse)
- `INFERENCE_POST_FILES` — list of inference JSONs to process
- `THRESHOLD` in `train_final.py` — update this after running `cv_eval.py`

---
*Built for the McGill Bot-or-Not Challenge.*
