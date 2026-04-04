"""
tune.py

Optuna hyperparameter tuning for XGBoost, LightGBM, and CatBoost.
Optimizes directly on the custom profit metric (not logloss).

Usage:
    python tune.py              # tune all three models, 50 trials each
    python tune.py --model xgb  # tune one model only
    python tune.py --trials 100 # more trials
"""

import argparse
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

optuna.logging.set_verbosity(optuna.logging.WARNING)

from botornot.config import TRAINING_PARQUET_PATH

PCA_N_TWEET = 20
PCA_N_DESC  = 20
N_FOLDS     = 5

REWARD_TP = 2
PENALTY_FP = -2
PENALTY_FN = -6
REWARD_TN  = 0


def calculate_profit(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp * REWARD_TP) + (fp * PENALTY_FP) + (fn * PENALTY_FN) + (tn * REWARD_TN)


def find_optimal_threshold(y_true, y_probs):
    best_thresh, best_score = 0.5, -float("inf")
    for thresh in np.linspace(0.01, 0.99, 100):
        score = calculate_profit(y_true, (y_probs >= thresh).astype(int))
        if score > best_score:
            best_score, best_thresh = score, thresh
    return best_thresh, best_score


def cv_profit(X, y, make_model_fn):
    """5-fold CV profit score for a given model factory."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=2)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = make_model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[val_idx])[:, 1]
        _, score = find_optimal_threshold(y.iloc[val_idx], probs)
        scores.append(score)
    return np.mean(scores)


def make_xgb_objective(X, y):
    ratio = (abs(PENALTY_FN) / abs(PENALTY_FP)) * ((len(y) - sum(y)) / sum(y))
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 50, 500),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample       = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
            gamma           = trial.suggest_float("gamma", 0, 5),
            reg_alpha       = trial.suggest_float("reg_alpha", 0, 10),
            reg_lambda      = trial.suggest_float("reg_lambda", 0, 10),
        )
        def make(): return xgb.XGBClassifier(**params, scale_pos_weight=ratio, eval_metric="logloss", random_state=2)
        return cv_profit(X, y, make)
    return objective


def make_lgbm_objective(X, y):
    ratio = (abs(PENALTY_FN) / abs(PENALTY_FP)) * ((len(y) - sum(y)) / sum(y))
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 50, 500),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample       = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
            reg_alpha       = trial.suggest_float("reg_alpha", 0, 10),
            reg_lambda      = trial.suggest_float("reg_lambda", 0, 10),
            num_leaves      = trial.suggest_int("num_leaves", 15, 127),
        )
        def make(): return lgb.LGBMClassifier(**params, scale_pos_weight=ratio, random_state=2, verbose=-1)
        return cv_profit(X, y, make)
    return objective


def make_catboost_objective(X, y):
    ratio = (abs(PENALTY_FN) / abs(PENALTY_FP)) * ((len(y) - sum(y)) / sum(y))
    def objective(trial):
        params = dict(
            iterations   = trial.suggest_int("iterations", 50, 500),
            depth        = trial.suggest_int("depth", 3, 8),
            learning_rate= trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg  = trial.suggest_float("l2_leaf_reg", 1, 10),
            bagging_temperature= trial.suggest_float("bagging_temperature", 0, 1),
        )
        def make(): return CatBoostClassifier(**params, scale_pos_weight=ratio, random_seed=2, verbose=0)
        return cv_profit(X, y, make)
    return objective


def tune(X, y, objective_fn, model_name, n_trials):
    print(f"\nTuning {model_name} ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn(X, y), n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{model_name} best CV profit: {best.value:.2f}")
    print(f"{model_name} best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    return best.params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="all", choices=["all", "xgb", "lgbm", "cat"])
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    print("Loading training data...")
    df = pd.read_parquet(TRAINING_PARQUET_PATH)
    drop_cols = ["author_id", "is_bot"]
    features = [c for c in df.columns if c not in drop_cols]
    X_raw = df[features]
    y = df["is_bot"]

    # apply PCA once upfront (fitted on full dataset — slight optimism but acceptable for tuning)
    tweet_emb_cols = [c for c in X_raw.columns if c.startswith('emb_t')]
    desc_emb_cols  = [c for c in X_raw.columns if c.startswith('emb_d')]
    meta_cols = [c for c in X_raw.columns if c not in tweet_emb_cols and c not in desc_emb_cols]

    pca_t = PCA(n_components=PCA_N_TWEET, random_state=2).fit(X_raw[tweet_emb_cols])
    pca_d = PCA(n_components=PCA_N_DESC,  random_state=2).fit(X_raw[desc_emb_cols])

    df_tw = pd.DataFrame(pca_t.transform(X_raw[tweet_emb_cols]),
                         columns=[f"PCA_Tweet_{i}" for i in range(PCA_N_TWEET)], index=X_raw.index)
    df_de = pd.DataFrame(pca_d.transform(X_raw[desc_emb_cols]),
                         columns=[f"PCA_Desc_{i}" for i in range(PCA_N_DESC)], index=X_raw.index)
    X = pd.concat([X_raw[meta_cols], df_tw, df_de], axis=1)

    results = {}
    if args.model in ("all", "xgb"):
        results["xgb"] = tune(X, y, make_xgb_objective, "XGBoost", args.trials)
    if args.model in ("all", "lgbm"):
        results["lgbm"] = tune(X, y, make_lgbm_objective, "LightGBM", args.trials)
    if args.model in ("all", "cat"):
        results["cat"] = tune(X, y, make_catboost_objective, "CatBoost", args.trials)

    print("\n=== Summary — paste these into train_final.py ===")
    for model_name, params in results.items():
        print(f"\n# {model_name}")
        for k, v in params.items():
            print(f"  {k}={repr(v)},")


if __name__ == "__main__":
    main()
