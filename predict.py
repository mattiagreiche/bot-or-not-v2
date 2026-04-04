import joblib
import numpy as np
import pandas as pd
from botornot.config import ARTIFACT_PATH, INFERENCE_POST_FILES, inference_paths, DIR_INFERENCE
import os

ARTIFACT = joblib.load(ARTIFACT_PATH)

# features to include in the borderline report for manual inspection
REPORT_FEATURES = [
    "description_partial_match", "identity_score", "confirmed_real",
    "account_exists", "name_match", "location_match",
    "num_hashtags_mean", "hour_entropy", "delta_s_cv",
    "name_nonalpha", "avg_word_length", "type_token_ratio", "unique_hashtag_ratio",
]

def predict_bot(new_data_df, predicted_bot_ids_path=None, borderline_path=None):
    # unpack artifact
    xgb_models   = ARTIFACT["xgb_models"]
    lgbm_model   = ARTIFACT["lgbm_model"]
    cat_model    = ARTIFACT["cat_model"]
    meta_learner = ARTIFACT["meta_learner"]
    pca_tweets   = ARTIFACT["pca_tweets"]
    pca_desc     = ARTIFACT["pca_desc"]
    tweet_cols   = ARTIFACT["tweet_emb_cols"]
    desc_cols    = ARTIFACT["desc_emb_cols"]
    meta_cols    = ARTIFACT["meta_cols"]
    threshold    = ARTIFACT["threshold"]

    n_tweet = pca_tweets.n_components_
    n_desc  = pca_desc.n_components_

    df_tweets = pd.DataFrame(
        pca_tweets.transform(new_data_df[tweet_cols]),
        columns=[f"PCA_Tweet_{i}" for i in range(n_tweet)], index=new_data_df.index)
    df_desc = pd.DataFrame(
        pca_desc.transform(new_data_df[desc_cols]),
        columns=[f"PCA_Desc_{i}" for i in range(n_desc)], index=new_data_df.index)

    X_final = pd.concat([new_data_df[meta_cols], df_tweets, df_desc], axis=1)

    # base learner predictions (average multiple XGBoost seeds)
    xgb_probs  = np.mean([m.predict_proba(X_final)[:, 1] for m in xgb_models], axis=0)
    lgbm_probs = lgbm_model.predict_proba(X_final)[:, 1]
    cat_probs  = cat_model.predict_proba(X_final)[:, 1]

    # meta-learner combines the three
    meta_X = np.column_stack([xgb_probs, lgbm_probs, cat_probs])
    final_probs = meta_learner.predict_proba(meta_X)[:, 1]

    predictions = (final_probs >= threshold).astype(int)

    bot_ids = new_data_df.loc[predictions == 1, "author_id"]
    with open(predicted_bot_ids_path, "w") as f:
        for uid in bot_ids:
            f.write(f"{uid}\n")

    print(f"Wrote {len(bot_ids)} bot user ids to {predicted_bot_ids_path}.")
    print(f"Prediction complete. Found {sum(predictions)} bots out of {len(predictions)} users.")

    # --- borderline report ---
    report = pd.DataFrame({
        "author_id":    new_data_df["author_id"].values,
        "prob":         final_probs,
        "xgb_prob":     xgb_probs,
        "lgbm_prob":    lgbm_probs,
        "cat_prob":     cat_probs,
        "prediction":   predictions,
        "dist_to_thresh": np.abs(final_probs - threshold),
    })

    # attach interpretable features for manual review
    for feat in REPORT_FEATURES:
        if feat in new_data_df.columns:
            report[feat] = new_data_df[feat].values

    # sort by closeness to threshold; show top 40
    borderline = report.sort_values("dist_to_thresh").head(40)
    borderline.to_csv(borderline_path, index=False, float_format="%.4f")

    print(f"\n--- Most uncertain predictions (closest to threshold={threshold:.3f}) ---")
    print(f"{'author_id':<22} {'prob':>6} {'xgb':>6} {'lgbm':>6} {'cat':>6}  {'call':<6}  {'desc_sim':>8}  {'id_score':>8}  {'hashtags':>8}")
    print("-" * 95)
    for _, row in borderline.iterrows():
        call = "BOT" if row["prediction"] == 1 else "human"
        desc  = f"{row['description_partial_match']:.2f}" if pd.notna(row.get("description_partial_match")) else "  NaN"
        ids   = f"{row['identity_score']:.2f}"            if pd.notna(row.get("identity_score"))            else "  NaN"
        hts   = f"{row['num_hashtags_mean']:.2f}"         if pd.notna(row.get("num_hashtags_mean"))          else "  NaN"
        print(f"{str(row['author_id']):<22} {row['prob']:>6.3f} {row['xgb_prob']:>6.3f} {row['lgbm_prob']:>6.3f} {row['cat_prob']:>6.3f}  {call:<6}  {desc:>8}  {ids:>8}  {hts:>8}")

    print(f"\nFull borderline report saved to {borderline_path}")
    return predictions, final_probs


if __name__ == "__main__":
    for json_file in INFERENCE_POST_FILES:
        parquet_path, preds_path, _ = inference_paths(json_file)
        num = os.path.basename(json_file).split('.')[-2]
        borderline_path = os.path.join(DIR_INFERENCE, f"borderline_cases_{num}.csv")

        print(f"\n=== Predicting for dataset {num} ===")
        df = pd.read_parquet(parquet_path)
        predict_bot(df, predicted_bot_ids_path=preds_path, borderline_path=borderline_path)
