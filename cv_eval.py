import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import shap

# this is a simple XGBoost model, but it takes into account the bot-or-not challenge's reward system to calculate
# the optimal "threshold" for guessing whether a user is a bot (or not).
# here I trained with k-fold CV to get an idea of which features are best (out of curiosity) and to find this best threshold.
# then, we actually train on the full train set in a different file

# config
from botornot.config import USE_EMBEDDINGS, TRAINING_PARQUET_PATH
INPUT_PATH = TRAINING_PARQUET_PATH
NUM_FOLDS = 5
SHOW_SHAP = True # set to true to see SHAP graph

# penalties/rewards from the bot-or-not challenge
REWARD_TP = 2   # catch a bot
PENALTY_FP = -6 # accuse a human
PENALTY_FN = -2 # miss a bot
REWARD_TN = 0 # ignore human

# to help find the best "cutoff" for identifying a human in the challenge
def calculate_profit(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    score = (tp * REWARD_TP) + (fp * PENALTY_FP) + (fn * PENALTY_FN) + (tn * REWARD_TN)
    return score

def find_optimal_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_score = -float("inf")
    
    # check 200 possible thresholds
    thresholds = np.linspace(0.01, 0.99, 200)
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        score = calculate_profit(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
            
    return best_thresh, best_score

# first, train with k-fold CV to test architecture and get best threshold to use later
def train_and_eval(X, y):
    fold_scores = []
    fold_thresholds = []
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2)
    
    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if USE_EMBEDDINGS:
            # pca to reduce dimensions of all those embeddings
            tweet_emb_cols = [c for c in X.columns if c.startswith('emb_t')] # for text/tweets (has mean and stdev)
            desc_emb_cols = [c for c in X.columns if c.startswith('emb_d')] # for description (only has mean)
            meta_cols = [c for c in X_train.columns if c not in tweet_emb_cols and c not in desc_emb_cols]
            
            # pca for Ttweets
            pca_tweets = PCA(n_components=10, random_state=2)
            X_train_tweets = pca_tweets.fit_transform(X_train[tweet_emb_cols])
            X_test_tweets = pca_tweets.transform(X_test[tweet_emb_cols])

            # make df
            tweet_cols = [f"PCA_Tweet_{i}" for i in range(10)]
            df_train_tweets = pd.DataFrame(X_train_tweets, columns=tweet_cols, index=X_train.index)
            df_test_tweets = pd.DataFrame(X_test_tweets, columns=tweet_cols, index=X_test.index)

            # pca for descriptions
            pca_desc = PCA(n_components=10, random_state=2)
            X_train_desc = pca_desc.fit_transform(X_train[desc_emb_cols])
            X_test_desc = pca_desc.transform(X_test[desc_emb_cols])

            # make df
            desc_cols = [f"PCA_Desc_{i}" for i in range(10)]
            df_train_desc = pd.DataFrame(X_train_desc, columns=desc_cols, index=X_train.index)
            df_test_desc = pd.DataFrame(X_test_desc, columns=desc_cols, index=X_test.index)

            # Combine Everything
            X_train_final = pd.concat([X_train[meta_cols], df_train_tweets, df_train_desc], axis=1)
            X_test_final = pd.concat([X_test[meta_cols], df_test_tweets, df_test_desc], axis=1)
        else:
            X_train_final = X_train
            X_test_final = X_test
        
        
        # model
        
        ratio = (len(y_train) - sum(y_train)) / sum(y_train)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            gamma=0,
            min_child_weight=2,
            learning_rate=0.05, 
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            random_state=2
        )
        
        model.fit(X_train_final, y_train)
        
        # pred probs
        y_probs = model.predict_proba(X_test_final)[:, 1]
        
        # instead of default 0.5, we find the best cutoff for this fold
        best_thresh, best_score = find_optimal_threshold(y_test, y_probs)
        
        y_pred_final = (y_probs >= best_thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
        
        fold_scores.append(best_score)
        fold_thresholds.append(best_thresh)
        
        print(f"Fold {fold}: Best thresh={best_thresh:.2f} | Score={best_score} | (TP={tp}, FP={fp}, FN={fn})")
        fold += 1
    
    print(f"Average profit per fold: {np.mean(fold_scores):.2f}")
    print(f"Optimal threshold (mean): {np.mean(fold_thresholds):.2f}")
    
    avg_thresh = np.mean(fold_thresholds)
    print(f"When running on the test set, use threshold: {avg_thresh:.3f}")
    print(f"i.e., if model.predict_proba() > {avg_thresh:.3f}, call it a BOT.")
    
    if SHOW_SHAP:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_final)
        
        shap.summary_plot(shap_values, X_test_final, show=False, plot_size=(12,8))
        plt.tight_layout
        plt.savefig("assets/shap.png")
    
def main():
    df = pd.read_parquet(INPUT_PATH)
    drop_cols = ["author_id", "is_bot"]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df["is_bot"]
    
    train_and_eval(X, y)

if __name__ == "__main__":
    main()