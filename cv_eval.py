import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import shap

from botornot.config import USE_EMBEDDINGS, TRAINING_PARQUET_PATH

INPUT_PATH = TRAINING_PARQUET_PATH
NUM_FOLDS = 5
PCA_N_TWEET = 25   # 20 captured 58.2% variance; 25 should get ~65%
PCA_N_DESC  = 30   # 20 captured only 47.4% variance; descriptions need more components
SHOW_SHAP = True

# penalties/rewards from the bot-or-not challenge
REWARD_TP = 2    # catch a bot
PENALTY_FP = -6  # accuse a human (very costly under new scoring)
PENALTY_FN = -2  # miss a bot
REWARD_TN = 0

def calculate_profit(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp * REWARD_TP) + (fp * PENALTY_FP) + (fn * PENALTY_FN) + (tn * REWARD_TN)

def find_optimal_threshold(y_true, y_probs):
    best_thresh, best_score = 0.5, -float("inf")
    for thresh in np.linspace(0.01, 0.99, 200):
        y_pred = (y_probs >= thresh).astype(int)
        score = calculate_profit(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score

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


def apply_pca(X_train, X_test):
    """Fit PCA on train split, transform both. Returns (X_train_final, X_test_final, pca_t, pca_d)."""
    tweet_emb_cols = [c for c in X_train.columns if c.startswith('emb_t')]
    desc_emb_cols  = [c for c in X_train.columns if c.startswith('emb_d')]
    meta_cols = [c for c in X_train.columns if c not in tweet_emb_cols and c not in desc_emb_cols]

    pca_t = PCA(n_components=PCA_N_TWEET, random_state=2)
    pca_d = PCA(n_components=PCA_N_DESC,  random_state=2)

    tw_tr = pd.DataFrame(pca_t.fit_transform(X_train[tweet_emb_cols]),
                         columns=[f"PCA_Tweet_{i}" for i in range(PCA_N_TWEET)], index=X_train.index)
    tw_te = pd.DataFrame(pca_t.transform(X_test[tweet_emb_cols]),
                         columns=[f"PCA_Tweet_{i}" for i in range(PCA_N_TWEET)], index=X_test.index)

    de_tr = pd.DataFrame(pca_d.fit_transform(X_train[desc_emb_cols]),
                         columns=[f"PCA_Desc_{i}" for i in range(PCA_N_DESC)], index=X_train.index)
    de_te = pd.DataFrame(pca_d.transform(X_test[desc_emb_cols]),
                         columns=[f"PCA_Desc_{i}" for i in range(PCA_N_DESC)], index=X_test.index)

    X_tr = pd.concat([X_train[meta_cols], tw_tr, de_tr], axis=1)
    X_te = pd.concat([X_test[meta_cols],  tw_te, de_te], axis=1)
    return X_tr, X_te, pca_t, pca_d


def train_and_eval(X, y):
    output_file = "results.txt"
    meta_cols = [c for c in X.columns if not c.startswith("emb_")]

    with open(output_file, "w") as f:
        f.write("Bot-or-Not Model CV Results\n")
        f.write("===========================\n\n")

        f.write(f"Scoring: TP={REWARD_TP}  FP={PENALTY_FP}  FN={PENALTY_FN}  TN={REWARD_TN}\n\n")

        # dataset summary
        n_total = len(y)
        n_bots = int(y.sum())
        n_humans = n_total - n_bots
        f.write("--- Dataset Summary ---\n")
        f.write(f"Total users : {n_total}\n")
        f.write(f"Bots        : {n_bots} ({100*n_bots/n_total:.1f}%)\n")
        f.write(f"Humans      : {n_humans} ({100*n_humans/n_total:.1f}%)\n")
        f.write(f"Class ratio (human/bot): {n_humans/n_bots:.2f}\n\n")

        # feature missing rates
        missing = X.isnull().mean()
        missing_nz = missing[missing > 0].sort_values(ascending=False)
        if len(missing_nz):
            f.write("--- Feature Missing Rates ---\n")
            for feat, rate in missing_nz.items():
                f.write(f"  {feat:<45} {100*rate:.1f}%\n")
            f.write("\n")
        else:
            f.write("No missing values in feature matrix.\n\n")

        # bot vs human feature comparison (metadata only, sorted by divergence)
        f.write("--- Bot vs Human Feature Means (sorted by divergence) ---\n")
        f.write(f"  {'Feature':<40} {'Bot mean':>12} {'Human mean':>12} {'B/H ratio':>10}\n")
        f.write("  " + "-" * 76 + "\n")
        bot_mask = y == 1
        rows = []
        for col in meta_cols:
            bm = X.loc[bot_mask, col].mean()
            hm = X.loc[~bot_mask, col].mean()
            rows.append((col, bm, hm, bm / (abs(hm) + 1e-9)))
        rows.sort(key=lambda r: abs(r[1] - r[2]) / (abs(r[2]) + 1e-9), reverse=True)
        for col, bm, hm, ratio in rows:
            f.write(f"  {col:<40} {bm:>12.4f} {hm:>12.4f} {ratio:>10.3f}\n")
        f.write("\n")

        # PCA explained variance on full dataset (informational, not per-fold)
        if USE_EMBEDDINGS:
            tweet_emb_cols = [c for c in X.columns if c.startswith('emb_t')]
            desc_emb_cols  = [c for c in X.columns if c.startswith('emb_d')]
            ev_t = PCA(n_components=PCA_N_TWEET, random_state=2).fit(X[tweet_emb_cols]).explained_variance_ratio_.sum()
            ev_d = PCA(n_components=PCA_N_DESC,  random_state=2).fit(X[desc_emb_cols]).explained_variance_ratio_.sum()
            f.write(f"--- PCA Explained Variance (n={PCA_N_TWEET} tweet, {PCA_N_DESC} desc) ---\n")
            f.write(f"  Tweet embeddings: {ev_t*100:.1f}% captured\n")
            f.write(f"  Desc  embeddings: {ev_d*100:.1f}% captured\n\n")

        f.write("--- Cross-Validation Results (XGB + LGBM + CatBoost ensemble) ---\n")
        f.write(f"  {'Fold':<6} {'Thresh':>7} {'Score':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}\n")
        f.write("  " + "-" * 62 + "\n")

    fold_scores_xgb  = []
    fold_scores_lgbm = []
    fold_scores_cat  = []
    fold_scores_ens  = []
    fold_thresholds  = []
    all_shap_by_model = {"xgb": [], "lgbm": [], "cat": []}
    all_cv_probs  = []
    all_cv_labels = []
    last_models = {}
    last_X_test_final = None
    last_shap_by_model = {}

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if USE_EMBEDDINGS:
            X_train_final, X_test_final, _, _ = apply_pca(X_train, X_test)
        else:
            X_train_final, X_test_final = X_train.copy(), X_test.copy()

        # cost-sensitive weight: accounts for class imbalance AND cost asymmetry.
        # FP=-6, FN=-2 → false positives are 3x more costly than false negatives,
        # so we weight bots less aggressively than pure class balance would suggest.
        class_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
        ratio = (abs(PENALTY_FN) / abs(PENALTY_FP)) * class_ratio  # ≈ (2/6) * 5.5 ≈ 1.8

        xgb_model  = make_xgb(ratio).fit(X_train_final, y_train)
        lgbm_model = make_lgbm(ratio).fit(X_train_final, y_train)
        cat_model  = make_catboost(ratio).fit(X_train_final, y_train)

        xgb_probs  = xgb_model.predict_proba(X_test_final)[:, 1]
        lgbm_probs = lgbm_model.predict_proba(X_test_final)[:, 1]
        cat_probs  = cat_model.predict_proba(X_test_final)[:, 1]
        ens_probs  = (xgb_probs + lgbm_probs + cat_probs) / 3.0

        # per-model scores (each at its own optimal threshold)
        xgb_t,  xgb_s  = find_optimal_threshold(y_test, xgb_probs)
        lgbm_t, lgbm_s = find_optimal_threshold(y_test, lgbm_probs)
        cat_t,  cat_s  = find_optimal_threshold(y_test, cat_probs)
        fold_scores_xgb.append(xgb_s)
        fold_scores_lgbm.append(lgbm_s)
        fold_scores_cat.append(cat_s)

        # ensemble optimal threshold + metrics
        best_thresh, best_score = find_optimal_threshold(y_test, ens_probs)
        y_pred = (ens_probs >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        fold_scores_ens.append(best_score)
        fold_thresholds.append(best_thresh)
        all_cv_probs.extend(ens_probs.tolist())
        all_cv_labels.extend(y_test.tolist())
        last_X_test_final = X_test_final
        last_models = {"xgb": xgb_model, "lgbm": lgbm_model, "cat": cat_model}

        print(f"Fold {fold}: thresh={best_thresh:.2f} | Score={best_score:4d} | TP={tp} FP={fp} FN={fn} | P={prec:.2f} R={rec:.2f} F1={f1:.2f}  [xgb={xgb_s} lgbm={lgbm_s} cat={cat_s}]")

        with open(output_file, "a") as f:
            f.write(f"  {fold:<6} {best_thresh:>7.2f} {best_score:>7d} {tp:>5d} {fp:>5d} {fn:>5d} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}   [xgb={xgb_s} lgbm={lgbm_s} cat={cat_s}]\n")

        if SHOW_SHAP:
            for mname, model in [("xgb", xgb_model), ("lgbm", lgbm_model), ("cat", cat_model)]:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_test_final)
                # LightGBM binary returns a list [neg_class, pos_class]
                if isinstance(sv, list):
                    sv = sv[1]
                if fold == NUM_FOLDS:
                    last_shap_by_model[mname] = sv
                all_shap_by_model[mname].append(
                    pd.Series(np.abs(sv).mean(axis=0), index=X_test_final.columns)
                )

    avg_thresh = np.mean(fold_thresholds)

    print(f"\nEnsemble avg profit: {np.mean(fold_scores_ens):.2f}  (std={np.std(fold_scores_ens):.2f})")
    print(f"  XGBoost alone  : {np.mean(fold_scores_xgb):.2f}")
    print(f"  LightGBM alone : {np.mean(fold_scores_lgbm):.2f}")
    print(f"  CatBoost alone : {np.mean(fold_scores_cat):.2f}")
    print(f"Optimal threshold (mean): {avg_thresh:.3f}")

    with open(output_file, "a") as f:
        f.write("\n")
        f.write(f"Ensemble avg profit : {np.mean(fold_scores_ens):.2f}  (std={np.std(fold_scores_ens):.2f})\n")
        f.write(f"  XGBoost alone     : {np.mean(fold_scores_xgb):.2f}\n")
        f.write(f"  LightGBM alone    : {np.mean(fold_scores_lgbm):.2f}\n")
        f.write(f"  CatBoost alone    : {np.mean(fold_scores_cat):.2f}\n")
        f.write(f"Optimal threshold (mean): {avg_thresh:.3f}\n")
        f.write(f"→ Use threshold {avg_thresh:.3f} for final predictions\n\n")

    if SHOW_SHAP and all_shap_by_model["xgb"]:
        # average SHAP importances per model across all folds
        avg_shap = {
            mname: pd.concat(folds, axis=1).mean(axis=1)
            for mname, folds in all_shap_by_model.items()
        }

        # comparison table: all three models side by side, sorted by mean importance
        all_feats = avg_shap["xgb"].index
        mean_imp = (avg_shap["xgb"] + avg_shap["lgbm"] + avg_shap["cat"]) / 3
        sorted_feats = mean_imp.sort_values(ascending=False).index

        with open(output_file, "a") as f:
            f.write("--- Feature Importance (mean |SHAP| across all folds) ---\n")
            f.write(f"  {'Feature':<42} {'XGBoost':>10} {'LightGBM':>10} {'CatBoost':>10} {'Mean':>10}\n")
            f.write("  " + "-" * 84 + "\n")
            for feat in sorted_feats:
                xv = avg_shap["xgb"].get(feat, 0)
                lv = avg_shap["lgbm"].get(feat, 0)
                cv = avg_shap["cat"].get(feat, 0)
                mv = (xv + lv + cv) / 3
                f.write(f"  {feat:<42} {xv:>10.4f} {lv:>10.4f} {cv:>10.4f} {mv:>10.4f}\n")
            f.write("\n")

            # XGBoost native gain (complementary view)
            gain = last_models["xgb"].get_booster().get_score(importance_type="gain")
            gain_series = pd.Series(gain).sort_values(ascending=False)
            f.write("--- XGBoost Native Importance (gain, last fold) ---\n")
            for feat, imp in gain_series.items():
                f.write(f"  {feat:<45} {imp:.2f}\n")
            f.write("\n")

        # SHAP beeswarm plots for all three models
        plot_configs = [
            ("xgb",  "XGBoost",   "assets/shap_xgb.png"),
            ("lgbm", "LightGBM",  "assets/shap_lgbm.png"),
            ("cat",  "CatBoost",  "assets/shap_cat.png"),
        ]
        for mname, title, path in plot_configs:
            shap.summary_plot(last_shap_by_model[mname], last_X_test_final,
                              show=False, plot_size=(12, 8))
            plt.title(title)
            plt.tight_layout()
            plt.savefig(path)
            plt.clf()

        # probability distribution
        all_cv_probs  = np.array(all_cv_probs)
        all_cv_labels = np.array(all_cv_labels)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(all_cv_probs[all_cv_labels == 0], bins=50, alpha=0.6, label="Human", color="steelblue")
        ax.hist(all_cv_probs[all_cv_labels == 1], bins=50, alpha=0.6, label="Bot",   color="tomato")
        ax.axvline(avg_thresh, color="black", linestyle="--", label=f"Threshold ({avg_thresh:.2f})")
        ax.set_xlabel("Predicted bot probability (ensemble)")
        ax.set_ylabel("Count")
        ax.set_title("Predicted probability distribution — all CV folds")
        ax.legend()
        plt.tight_layout()
        plt.savefig("assets/prob_dist.png")
        plt.clf()
        print("Plots saved: assets/shap_xgb.png, assets/shap_lgbm.png, assets/shap_cat.png, assets/prob_dist.png")

    print(f"Full results written to {output_file}")


def main():
    df = pd.read_parquet(INPUT_PATH)
    drop_cols = ["author_id", "is_bot"]
    features = [c for c in df.columns if c not in drop_cols]

    X = df[features]
    y = df["is_bot"]

    train_and_eval(X, y)

if __name__ == "__main__":
    main()
