from datasets import load_dataset
import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs("../data", exist_ok=True)

for split in ["continuations", "disinformation", "doubling_down_known_facts", "known_facts", "provided_facts", "statistics"]:
    ds = load_dataset("cais/mask", split)

    df = pd.DataFrame(ds['test'])
    df.to_csv(f"../data/{split}.csv", index=False)
    print(f"Downloaded {len(df)} samples for {split}")

# Download sycophancy dataset
print("Downloading sycophancy dataset...")
sycophancy_ds = load_dataset("EleutherAI/sycophancy")
df_sycophancy = pd.DataFrame(sycophancy_ds['test'])
df_sycophancy.to_csv(f"../data/sycophancy.csv", index=False)
print(f"Downloaded {len(df_sycophancy)} samples for sycophancy")

    # df.sample(5).to_csv(f"../test_data/{split}.csv", index=False)