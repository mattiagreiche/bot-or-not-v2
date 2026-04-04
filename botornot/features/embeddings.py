from sentence_transformers import SentenceTransformer
import pandas as pd

# config
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 128

def get_embeddings(df, column):
    # in practice, (df, column) is either (posts_df, "text") or (users_df, "description")
    
    model = SentenceTransformer(MODEL_NAME)
    
    sentences = df[column]
    
    embeddings = model.encode(sentences, batch_size=BATCH_SIZE, show_progress_bar=True)
    
    emb_cols = [f"e_{i}" for i in range(embeddings.shape[1])] # name e1, e2, ...
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    
    # group
    emb_df["author_id"] = df["author_id"].values
    # aggregate by user
    agg_df = emb_df.groupby("author_id")[emb_cols].agg(['mean', 'std'] if column=="text" else ['mean']) # for description, only one value (just using mean here)
    
    # flatten and name
    agg_df.columns = [f"emb_{'t' if column=='text' else 'd'}_{stat}_{col.split('_')[1]}" for col, stat in agg_df.columns]
    
    # fix single user stdevs
    agg_df = agg_df.fillna(0)
    
    print(f"Generated {agg_df.shape[1]} embedding features for {len(agg_df)} users")
    return agg_df
