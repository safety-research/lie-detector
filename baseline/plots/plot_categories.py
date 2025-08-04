import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

def plot_split_categories(csv_path, output_path="split_categories_histogram.png"):
    # Load CSV
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Dictionary to hold counts
    category_counts = defaultdict(int)

    # Iterate over each row and column (excluding first column if it's 'Task')
    for _, row in df.iterrows():
        for col in df.columns[1:]:
            cell = row[col]
            if pd.notnull(cell):
                # Split on '/' and strip whitespace, then normalize
                categories = [c.strip().lower() for c in str(cell).split('/')]
                for category in categories:
                    category_counts[category] += 1

    print(f"Found {len(category_counts)} unique categories")
    print("Category counts:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    # Convert to pandas Series for sorting and plotting
    counts_series = pd.Series(category_counts).sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    ax = counts_series.plot(kind="bar", color="#FFA500")
    plt.title("Number of Tasks per Category")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    
    # Add percentage labels above each bar
    total_unique_tasks = len(df)  # Number of rows = number of unique tasks
    for i, (category, count) in enumerate(counts_series.items()):
        percentage = (count / total_unique_tasks) * 100
        plt.text(i, count + 0.5, f'{percentage:.1f}%', 
                ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    plt.tight_layout()

    # Save and show
    print(f"Saving plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully!")
    plt.show()


if __name__ == "__main__":
    # Usage: python script.py <csv_path>
    if len(sys.argv) < 2:
        print("Usage: python plot_categories.py <csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    plot_split_categories(csv_path)
