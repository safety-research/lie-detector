import json
import numpy as np
from flask.cli import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import openai
from typing import List, Dict, Any
import os
from tqdm import tqdm
import time

load_dotenv()

class QuestionDeduplicator:
    def __init__(self, api_key: str = None):
        """Initialize the deduplicator with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")

        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)

    def load_data(self, filepath: str) -> List[Dict[str, Any]]:
        """Load the JSON data from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Handle both direct list and wrapped format
        if isinstance(data, dict) and 'generated_samples' in data:
            return data['generated_samples']
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Unexpected data format")

    def get_embeddings_batch(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """Get embeddings for a batch of texts using OpenAI API."""
        embeddings = []
        batch_size = 100  # OpenAI recommends batches of up to 2048 for embedding models

        for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
            batch = texts[i:i + batch_size]

            # Retry logic for API calls
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=model,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"Error getting embeddings (retry {retry + 1}): {e}")
                        time.sleep(2 ** retry)  # Exponential backoff
                    else:
                        raise e

            # Rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.1)  # Small delay between batches

        return embeddings

    def find_duplicate_clusters(self, embeddings: np.ndarray, eps: float = 0.1, min_samples: int = 2) -> Dict[
        int, List[int]]:
        """Find clusters of duplicate/similar questions using DBSCAN."""
        # Use DBSCAN for clustering similar embeddings
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        # Group indices by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # -1 indicates noise/no cluster
                clusters[label].append(idx)

        return dict(clusters)

    def select_representative_samples(self, samples: List[Dict[str, Any]],
                                      clusters: Dict[int, List[int]],
                                      embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Select representative samples from each cluster and all unique samples."""
        selected_indices = set()

        # For each cluster, select the sample closest to the cluster center
        for cluster_id, indices in clusters.items():
            cluster_embeddings = embeddings[indices]
            center = np.mean(cluster_embeddings, axis=0)

            # Find the sample closest to the center
            distances = [np.linalg.norm(embeddings[idx] - center) for idx in indices]
            representative_idx = indices[np.argmin(distances)]
            selected_indices.add(representative_idx)

        # Add all samples that are not in any cluster (unique samples)
        all_clustered = set()
        for indices in clusters.values():
            all_clustered.update(indices)

        for i in range(len(samples)):
            if i not in all_clustered:
                selected_indices.add(i)

        # Return selected samples
        return [samples[i] for i in sorted(selected_indices)]

    def print_cluster_examples(self, samples: List[Dict[str, Any]],
                               clusters: Dict[int, List[int]],
                               num_examples: int = 5) -> None:
        """Print example questions from each cluster to verify clustering quality."""
        print("\n" + "=" * 80)
        print("CLUSTER ANALYSIS")
        print("=" * 80)

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

        # Show top clusters
        num_clusters_to_show = min(10, len(sorted_clusters))
        print(f"\nShowing top {num_clusters_to_show} largest clusters (out of {len(clusters)} total):\n")

        for i, (cluster_id, indices) in enumerate(sorted_clusters[:num_clusters_to_show]):
            print(f"Cluster {i + 1} (ID: {cluster_id}) - {len(indices)} questions")
            print("-" * 60)

            # Show up to num_examples questions from this cluster
            examples_to_show = min(num_examples, len(indices))
            for j in range(examples_to_show):
                idx = indices[j]
                question = samples[idx]['question_text']
                domain = samples[idx].get('domain', 'N/A')
                print(f"  {j + 1}. [{domain}] {question}")

            if len(indices) > num_examples:
                print(f"  ... and {len(indices) - num_examples} more questions")
            print()

        # Summary statistics
        cluster_sizes = [len(indices) for indices in clusters.values()]
        print("\nCluster Size Statistics:")
        print(f"  - Total clusters: {len(clusters)}")
        print(f"  - Average cluster size: {np.mean(cluster_sizes):.2f}")
        print(f"  - Median cluster size: {np.median(cluster_sizes):.0f}")
        print(f"  - Largest cluster: {max(cluster_sizes)} questions")
        print(f"  - Smallest cluster: {min(cluster_sizes)} questions")
        print("=" * 80 + "\n")

    def downsample_uniform(self, samples: List[Dict[str, Any]],
                           embeddings: np.ndarray,
                           target_size: int = None) -> List[Dict[str, Any]]:
        """Downsample to create a pseudo-uniform distribution in embedding space."""
        if target_size is None or target_size >= len(samples):
            return samples

        # Use k-means++ initialization strategy for uniform coverage
        selected_indices = []
        remaining_indices = list(range(len(samples)))

        # Start with a random sample
        first_idx = np.random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select samples that are farthest from already selected ones
        while len(selected_indices) < target_size and remaining_indices:
            # Calculate minimum distances to selected samples
            min_distances = []
            for idx in remaining_indices:
                distances = [np.linalg.norm(embeddings[idx] - embeddings[sel_idx])
                             for sel_idx in selected_indices]
                min_distances.append(min(distances))

            # Select the sample with maximum minimum distance
            next_idx = remaining_indices[np.argmax(min_distances)]
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)

        return [samples[i] for i in selected_indices]

    def deduplicate_and_downsample(self, filepath: str,
                                   output_filepath: str,
                                   eps: float = 0.1,
                                   target_size: int = None,
                                   save_embeddings: bool = True,
                                   print_clusters:bool = True) -> Dict[str, Any]:
        """Main method to deduplicate and downsample the dataset."""
        print("Loading data...")
        samples = self.load_data(filepath)
        print(f"Loaded {len(samples)} samples")

        # Extract question texts
        questions = [sample['question_text'] for sample in samples]

        # Get embeddings
        print("Getting embeddings from OpenAI...")
        embeddings_list = self.get_embeddings_batch(questions)
        embeddings = np.array(embeddings_list)

        # Find duplicate clusters
        print("Finding duplicate clusters...")
        clusters = self.find_duplicate_clusters(embeddings, eps=eps)
        print(f"Found {len(clusters)} duplicate clusters")

        # Print cluster examples if requested
        if print_clusters and clusters:
            self.print_cluster_examples(samples, clusters)

        # Select representative samples
        print("Selecting representative samples...")
        deduplicated_samples = self.select_representative_samples(samples, clusters, embeddings)

        # Get embeddings for deduplicated samples
        deduplicated_indices = []
        for i, sample in enumerate(samples):
            if sample in deduplicated_samples:
                deduplicated_indices.append(i)
        deduplicated_embeddings = embeddings[deduplicated_indices]

        print(f"Reduced from {len(samples)} to {len(deduplicated_samples)} samples after deduplication")

        # Downsample if requested
        if target_size and target_size < len(deduplicated_samples):
            print(f"Downsampling to {target_size} samples...")
            final_samples = self.downsample_uniform(deduplicated_samples,
                                                    deduplicated_embeddings,
                                                    target_size)
        else:
            final_samples = deduplicated_samples

        print(f"Final dataset size: {len(final_samples)} samples")

        # Prepare output
        output_data = {
            "metadata": {
                "original_count": len(samples),
                "deduplicated_count": len(deduplicated_samples),
                "final_count": len(final_samples),
                "duplicate_clusters_found": len(clusters),
                "eps_threshold": eps
            },
            "generated_samples": final_samples
        }

        # Save results
        with open(output_filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved results to {output_filepath}")

        # Optionally save embeddings
        if save_embeddings:
            embeddings_file = output_filepath.replace('.json', '_embeddings.npy')
            np.save(embeddings_file, embeddings)
            print(f"Saved embeddings to {embeddings_file}")

        return output_data


# Example usage
if __name__ == "__main__":
    # Initialize the deduplicator
    deduplicator = QuestionDeduplicator()  # Will use OPENAI_API_KEY env var

    # Run deduplication and downsampling
    result = deduplicator.deduplicate_and_downsample(
        filepath="generated_sandbagging_samples_10000_gpt.json",
        output_filepath="deduplicated_samples.json",
        eps=0.1,  # Similarity threshold (lower = more similar required for grouping)
        target_size=5000,  # Target number of samples after downsampling (optional)
        save_embeddings=True,  # Save embeddings for future use
        print_clusters=True  # Print cluster examples
    )

    print("\nDeduplication Summary:")
    print(f"Original samples: {result['metadata']['original_count']}")
    print(f"After deduplication: {result['metadata']['deduplicated_count']}")
    print(f"Final samples: {result['metadata']['final_count']}")
    print(f"Duplicate clusters found: {result['metadata']['duplicate_clusters_found']}")