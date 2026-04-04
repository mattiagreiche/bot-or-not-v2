import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import joblib
import os

from botornot.config import TRAINING_PARQUET_PATH, ARTIFACT_PATH

# keep in sync with cv_eval.py
PCA_N_TWEET = 25
PCA_N_DESC  = 30
THRESHOLD   = 0.45  # update after running cv_eval.py
OOF_FOLDS   = 5     # folds used to generate out-of-fold predictions for meta-learner


def make_xgb(ratio, seed=2):
    return xgb.XGBClassifier(
        n_estimators=439, max_depth=5, learning_rate=0.09448775990761163,
        subsample=0.500314394880139, colsample_bytree=0.8941804451471327,
        min_child_weight=3, gamma=1.0228339936036628, reg_alpha=0.658171190074973,
        reg_lambda=7.6588096892893285,
        scale_pos_weight=ratio, eval_metric="logloss", random_state=seed
    )

def make_lgbm(ratio, seed=2):
    return lgb.LGBMClassifier(
        n_estimators=458,
        max_depth=3,
        learning_rate=0.1553307591653191,
        subsample=0.7112481052266945,
        colsample_bytree=0.5421699186850049,
        min_child_samples=32,
        reg_alpha=0.821864634454065,
        reg_lambda=9.022241470538907,
        num_leaves=43,
        scale_pos_weight=ratio, random_state=seed, verbose=-1
    )

def make_catboost(ratio, seed=2):
    return CatBoostClassifier(
        iterations=377,
        depth=5,
        learning_rate=0.062049443814791376,
        l2_leaf_reg=5.615578738372279,
        bagging_temperature=0.7805848367506769,
        scale_pos_weight=ratio, random_seed=seed, verbose=0
    )


def generate_oof(X, y, make_model_fn, n_splits=OOF_FOLDS):
    """Generate out-of-fold probability predictions for stacking."""
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    OOF fold {i+1}/{n_splits}...", end=" ", flush=True)
        model = make_model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        print("done")
    return oof


def train_final_ensemble(X, y):
    # --- PCA ---
    tweet_emb_cols = [c for c in X.columns if c.startswith('emb_t')]
    desc_emb_cols  = [c for c in X.columns if c.startswith('emb_d')]
    meta_cols = [c for c in X.columns if c not in tweet_emb_cols and c not in desc_emb_cols]

    pca_tweets = PCA(n_components=PCA_N_TWEET, random_state=2)
    pca_desc   = PCA(n_components=PCA_N_DESC,  random_state=2)

    df_tweets = pd.DataFrame(
        pca_tweets.fit_transform(X[tweet_emb_cols]),
        columns=[f"PCA_Tweet_{i}" for i in range(PCA_N_TWEET)], index=X.index)
    df_desc = pd.DataFrame(
        pca_desc.fit_transform(X[desc_emb_cols]),
        columns=[f"PCA_Desc_{i}" for i in range(PCA_N_DESC)], index=X.index)

    X_final = pd.concat([X[meta_cols], df_tweets, df_desc], axis=1)

    # cost-sensitive weight: FP=-6, FN=-2 → bias toward precision, not recall
    class_ratio = (len(y) - sum(y)) / sum(y)
    ratio = (2 / 6) * class_ratio  # (FN_cost / FP_cost) × class_imbalance ≈ 1.8

    # --- generate OOF predictions from each base learner ---
    print("Generating XGBoost OOF predictions...")
    xgb_oof = generate_oof(X_final, y, lambda: make_xgb(ratio))

    print("Generating LightGBM OOF predictions...")
    lgbm_oof = generate_oof(X_final, y, lambda: make_lgbm(ratio))

    print("Generating CatBoost OOF predictions...")
    cat_oof = generate_oof(X_final, y, lambda: make_catboost(ratio))

    # --- train meta-learner on stacked OOF ---
    meta_X = np.column_stack([xgb_oof, lgbm_oof, cat_oof])
    meta_learner = LogisticRegression(C=1.0, max_iter=1000)
    meta_learner.fit(meta_X, y)
    print(f"Meta-learner coefficients: XGB={meta_learner.coef_[0][0]:.3f}  LGBM={meta_learner.coef_[0][1]:.3f}  CAT={meta_learner.coef_[0][2]:.3f}")

    # --- train final base learners on full training data ---
    # XGBoost: multiple seeds to reduce variance
    print("Training final XGBoost models...")
    xgb_models = []
    for i, seed in enumerate([12, 17, 39]):
        print(f"  XGBoost seed {seed}...", end=" ", flush=True)
        m = make_xgb(ratio, seed=seed)
        m.fit(X_final, y)
        xgb_models.append(m)
        print("done")

    print("Training final LightGBM model...")
    lgbm_model = make_lgbm(ratio)
    lgbm_model.fit(X_final, y)

    print("Training final CatBoost model...")
    cat_model = make_catboost(ratio)
    cat_model.fit(X_final, y)

    artifact = {
        "xgb_models":    xgb_models,
        "lgbm_model":    lgbm_model,
        "cat_model":     cat_model,
        "meta_learner":  meta_learner,
        "pca_tweets":    pca_tweets,
        "pca_desc":      pca_desc,
        "tweet_emb_cols": tweet_emb_cols,
        "desc_emb_cols":  desc_emb_cols,
        "meta_cols":      meta_cols,
        "threshold":      THRESHOLD,
    }

    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved stacking ensemble artifact to {ARTIFACT_PATH}")


if __name__ == "__main__":
    df = pd.read_parquet(TRAINING_PARQUET_PATH)
    drop_cols = ["author_id", "is_bot"]
    features = [c for c in df.columns if c not in drop_cols]
    X = df[features]
    y = df["is_bot"]
    train_final_ensemble(X, y)
