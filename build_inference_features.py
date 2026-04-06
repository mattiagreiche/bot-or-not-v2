import os
from botornot.config import USE_EMBEDDINGS, INFERENCE_POST_FILES, inference_paths
from botornot.features import build_features_df

if __name__ == "__main__":
    for json_file in INFERENCE_POST_FILES:
        parquet_path, _, _ = inference_paths(json_file)

        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

        print(f"\nProcessing {os.path.basename(json_file)}...")
        final_df = build_features_df([json_file], bot_files=None, use_embeddings=USE_EMBEDDINGS)

        print(f"Saving {len(final_df)} rows to {parquet_path}")
        final_df.to_parquet(parquet_path, index=False)

    print("\nDone.")
