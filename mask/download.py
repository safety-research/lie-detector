from datasets import load_dataset
import pandas as pd

for split in ["continuations", "disinformation", "doubling_down_known_facts", "known_facts", "provided_facts", "statistics"]:
    ds = load_dataset("cais/mask", split)

    df = pd.DataFrame(ds['test'])
    df.to_csv(f"./csv_data/{split}.csv", index=False)

    df.sample(5).to_csv(f"./test_csv_data/{split}.csv", index=False)