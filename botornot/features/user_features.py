from collections import Counter
import re
import string
import json
import math
import os
import pandas as pd
from botornot.features.embeddings import get_embeddings
from botornot.config import DIR_RAW

TWITTER_LOOKUP_PATH = os.path.join(DIR_RAW, "twitter_lookup.csv")

# hi! this function extracts the features for use in the XGBoost model.
# some features are easily extractable, and the others are embeddings from a sentence transformer model

# text helpers
URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r'#\w+')
MENTION_RE = re.compile(r"@\w+")
NONALPHA_RE = re.compile(r'[^A-Za-z0-9_]')

def count_caps(s):
    return sum(1 for c in s if 'A' <= c <= 'Z')

def count_punct(s):
    return sum(1 for c in s if c in string.punctuation)

def count_exclam(s):
    return sum(1 for c in s if c == '!')

def count_nonalpha(s):
    return len(NONALPHA_RE.findall(s or "")) # avoids empty string errors

def count_hashtags(s):
    return len(HASHTAG_RE.findall(s or ""))

def char_entropy(s):
    # shannon entropy
    if not s:
        return 0.0
    cnt = Counter(s)
    L = len(s)
    ent = 0.0
    for c in cnt.values():
        p = c / L
        ent -= p * math.log2(p)
    return ent


# core feature extraction

def extract_features_from_posts(posts_list, use_embeddings=True):
    # users_list is the 'users' JSON
    posts = pd.DataFrame(posts_list)
    
    # pre-processing
    posts["created_at"] = pd.to_datetime(posts["created_at"], errors="coerce")
    posts["text"] = posts["text"].fillna("")
    
    # sort by user then time for time-diff calculations
    posts = posts.sort_values(["author_id", "created_at"]).reset_index(drop=True)
    
    # per-post features
    posts["len"] = posts["text"].str.len()
    posts["num_caps"] = posts["text"].map(count_caps)
    posts["num_punct"] = posts["text"].map(count_punct)
    posts["num_hashtags"] = posts["text"].map(count_hashtags)
    posts["num_exclams"] = posts["text"].map(count_exclam)
    posts["num_nonalpha"] = posts["text"].map(count_nonalpha)
    
    # time between posts
    posts["delta_s"] = posts.groupby("author_id")["created_at"].diff().dt.total_seconds()
    
    # aggregate per user and get a bunch of metrics
    posts_agg = posts.groupby("author_id").agg({
        "delta_s": ["mean", "min", "std", "median"],
        "len": ["mean", "min", "max", "std"],
        "num_caps": ["mean", "min", "max", "std"],
        "num_punct": ["mean", "min", "max", "std"],
        "num_hashtags": ["mean", "min", "max", "std"],
        "num_exclams": ["mean", "min", "max", "std"],
        "text": "count",
        "created_at": ["min", "max"],
    })
    
    # flatten columns
    posts_agg.columns = ["_".join(col).strip() for col in posts_agg.columns]
    posts_agg = posts_agg.reset_index()
    posts_agg = posts_agg.rename(columns={"text_count": "post_count"})
    
    # activity
    posts_agg["active_span_s"] = (posts_agg["created_at_max"] - posts_agg["created_at_min"]).dt.total_seconds()
    
    posts_agg["posts_per_day"] = posts_agg["post_count"] / (posts_agg["active_span_s"] / 86400 + 1e-9) # avoid division by zero if one post
    
    # fix delta_s to 0 for single posters
    delta_cols = [c for c in posts_agg.columns if c.startswith("delta_s_")]
    for col in delta_cols:
        posts_agg[col] = posts_agg[col].fillna(0)

    # for single posters, posts_per_day = 1/2
    posts_agg["single_post"] = (posts_agg["post_count"] == 1).astype(int)
    posts_agg.loc[posts_agg["single_post"] == 1, "posts_per_day"] = 0.5 # time period is 2 days in the datasets    
    
    posts_agg = posts_agg.drop(columns=["created_at_min", "created_at_max"])
    posts_agg = posts_agg.fillna(0) # fill NaNs from std dev calculations on single posts
    
    if not use_embeddings:
        return posts_agg
    
    # get LM embeddings
    post_embeddings = get_embeddings(posts, "text")
    
    # merge
    post_feats = pd.merge(posts_agg, post_embeddings, on="author_id", how="inner")
    
    return post_feats

def extract_features_from_users(users_list, use_embeddings=True):
    users = pd.DataFrame(users_list)
    
    # id match
    if "id" in users.columns:
        users = users.rename(columns={"id": "author_id"})
        
    # username
    u = users.get("username", "").fillna("")
    users["username_len"] = u.str.len()
    users["username_digits"] = u.str.count(r"\d")
    users["username_entropy"] = u.apply(lambda s: char_entropy(str(s)) if pd.notna(s) else 0.0)

    # name features
    n = users.get("name", "").fillna("")
    users["name_len"] = n.str.len()
    users["name_nonalpha"] = n.str.count(r"[^A-Za-z0-9_]")
    users["name_word_count"] = n.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)  
    
    # description features
    d = users.get("description", "").fillna("")
    users["description_len"] = d.str.len()
    users["description_has_url"] = d.str.contains(URL_RE).fillna(False).astype(int)
    users["description_has_hashtag"] = d.str.contains(HASHTAG_RE).fillna(False).astype(int)
    users["description_has_mention"] = d.str.contains(MENTION_RE).fillna(False).astype(int)
    users["description_nonalpha"] = d.str.count(r"[^A-Za-z0-9_]")
    users["missing_description"] = (d == "").astype(int)    
    
    # location features
    loc = users.get("location", "").fillna("")
    users["location_present"] = (loc != "").astype(int)
    users["location_len"] = loc.str.len()
    
    if not use_embeddings:
        # keep only useful features (no embeddings) + author_id
        keep_cols = [
            "author_id", "username",
            "username_len", "username_digits", "username_entropy",
            "name_len", "name_nonalpha", "name_word_count", 
            "description_len", "description_has_url", "description_has_hashtag", 
            "description_has_mention", "missing_description",
            "description_nonalpha",
            "location_present", "location_len",
        ]
        return users[keep_cols]
        
    # get LM embeddings
    user_embeddings = get_embeddings(users, "description")
    
    # merge
    user_feats = pd.merge(users, user_embeddings, on="author_id", how="inner")
    
    # keep only useful/numerical features (INCL. embeddings) + author_id
    keep_cols = [
        "author_id", "username", "z_score",
        "username_len", "username_digits", "username_entropy",
        "name_len", "name_nonalpha", "name_word_count", 
        "description_len", "description_has_url", "description_has_hashtag", 
        "description_has_mention", "missing_description",
        "location_present", "location_len",
    ] + [c for c in user_feats.columns if c.startswith('emb')]
    
    user_feats = user_feats[keep_cols]
    
    return user_feats

def process_single_file(filepath, use_embeddings=True):
    print(f"Processing {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # extract posts features
    posts_data = data.get('posts', []) 
    posts_df = extract_features_from_posts(posts_data, use_embeddings=use_embeddings)
    
    # extract users features
    users_data = data.get('users', [])
    users_df = extract_features_from_users(users_data, use_embeddings=use_embeddings)
    
    # merge (inner join to ensure only keep users we have full data for)
    merged = pd.merge(users_df, posts_df, on="author_id", how="inner")
    return merged
    


# load bot labels
def load_bot_labels(bot_files):
    bot_ids = set()
    for path in bot_files:
        with open(path) as f:
            for line in f:
                bot_ids.add(line.strip())
    return bot_ids


def build_features_df(post_files, bot_files=None, use_embeddings=True):
    all_features = []
    for post_file in post_files:
        df = process_single_file(post_file, use_embeddings=use_embeddings)
        all_features.append(df)
    final_df = pd.concat(all_features, ignore_index=True)

    if os.path.exists(TWITTER_LOOKUP_PATH):
        print(f"Merging Twitter lookup features from {TWITTER_LOOKUP_PATH}...")
        lookup_df = pd.read_csv(TWITTER_LOOKUP_PATH)
        lookup_df = lookup_df.replace(-1, float("nan"))
        final_df = pd.merge(final_df, lookup_df, on="username", how="left")

    final_df = final_df.drop(columns=["username"], errors="ignore")

    if bot_files is not None:
        bot_ids = load_bot_labels(bot_files)
        final_df["is_bot"] = final_df["author_id"].isin(bot_ids).astype(int)
    return final_df
