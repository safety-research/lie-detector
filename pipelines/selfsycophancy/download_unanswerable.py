#!/usr/bin/env python3
"""
Script to download unanswerable questions from ALL models in S3,
embed them using OpenAI, and remove duplicates based on semantic similarity.
"""

import json
import requests
import os
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import time
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dataclasses import dataclass, asdict
import hashlib
from collections import defaultdict
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import pandas as pd

load_dotenv()


@dataclass
class Question:
    """Data class for a question with all metadata."""
    question: str
    choices: Dict[str, str]
    answer: str
    subject: str
    sample_id: str
    model: str
    did_lie: bool
    chosen_answer: str
    evidence: str
    question_hash: str = ""
    embedding: List[float] = None

    def __post_init__(self):
        # Create a hash of the question text for exact duplicate detection
        if not self.question_hash:
            self.question_hash = hashlib.md5(self.question.encode()).hexdigest()


class UnanswerableDatasetDownloader:
    def __init__(self, bucket_name: str = "dipika-lie-detection-data",
                 output_dir: str = "unanswerable_dataset_all_models",
                 use_embeddings: bool = True):
        self.bucket_name = bucket_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache for downloaded files
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Embeddings cache
        self.embeddings_cache_dir = self.output_dir / "embeddings_cache"
        self.embeddings_cache_dir.mkdir(exist_ok=True)

        # Initialize S3 client
        self.s3_client = boto3.client('s3')

        # Base path in S3
        self.base_prefix = "processed-data/"

        # OpenAI setup
        self.use_embeddings = use_embeddings
        if use_embeddings:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                print("Warning: OPENAI_API_KEY not found in environment. Embeddings will be disabled.")
                self.use_embeddings = False
            else:
                self.openai_client = openai.OpenAI(api_key=openai.api_key)

    def list_all_models(self) -> List[str]:
        """List all model directories in the S3 bucket."""
        models = set()

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.base_prefix,
                Delimiter='/'
            )

            for page in page_iterator:
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        # Extract provider/model structure
                        path = prefix['Prefix'].replace(self.base_prefix, '').rstrip('/')
                        if path:  # provider level
                            # List models under this provider
                            provider_paginator = self.s3_client.get_paginator('list_objects_v2')
                            provider_iterator = provider_paginator.paginate(
                                Bucket=self.bucket_name,
                                Prefix=prefix['Prefix'],
                                Delimiter='/'
                            )

                            for provider_page in provider_iterator:
                                if 'CommonPrefixes' in provider_page:
                                    for model_prefix in provider_page['CommonPrefixes']:
                                        model_path = model_prefix['Prefix'].replace(self.base_prefix, '').rstrip('/')
                                        models.add(model_path)

            print(f"Found {len(models)} models: {sorted(models)}")
            return sorted(models)

        except Exception as e:
            print(f"Error listing models: {e}")
            # Fallback to known model
            return ["google/gemma_3_27b_it"]

    def list_s3_files_for_model(self, model_path: str, subjects: List[str] = None) -> Dict[str, List[str]]:
        """List all available files for a specific model organized by subject."""
        files_by_subject = {}
        unanswerable_prefix = f"{self.base_prefix}{model_path}/unanswerable_claims/"

        try:
            # List all subjects if none specified
            if subjects is None:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(
                    Bucket=self.bucket_name,
                    Prefix=unanswerable_prefix,
                    Delimiter='/'
                )

                subjects = []
                for page in page_iterator:
                    if 'CommonPrefixes' in page:
                        for prefix in page['CommonPrefixes']:
                            subject = prefix['Prefix'].replace(unanswerable_prefix, '').rstrip('/')
                            if subject:
                                subjects.append(subject)

            # List files for each subject
            for subject in subjects:
                subject_prefix = f"{unanswerable_prefix}{subject}/"
                files_by_subject[subject] = []

                paginator = self.s3_client.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(
                    Bucket=self.bucket_name,
                    Prefix=subject_prefix
                )

                for page in page_iterator:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            filename = key.split('/')[-1]
                            if filename.endswith('.json'):
                                files_by_subject[subject].append(filename)

        except Exception as e:
            print(f"Error listing files for model {model_path}: {e}")

        return files_by_subject

    def fetch_json_from_s3(self, model_path: str, subject: str, filename: str) -> Optional[Dict]:
        """Fetch JSON data directly from S3."""
        try:
            key = f"{self.base_prefix}{model_path}/unanswerable_claims/{subject}/{filename}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return json.loads(response['Body'].read())
        except Exception as e:
            return None

    def extract_question_data(self, json_data: Dict, model_path: str) -> Optional[Question]:
        """Extract question data from JSON with improved parsing for various formats."""
        try:
            question_with_na = json_data.get("metadata", {}).get("question_with_na", "")

            if not question_with_na:
                return None

            # Clean up the text
            text = question_with_na.strip()

            # Try multiple parsing strategies
            choices = {}
            question_text = ""

            # Strategy 1: Look for choices embedded with **A)** or **A:** format
            embedded_pattern = r'\*\*([A-Za-z])[):\]]\*\*\s*([^*]+?)(?=\*\*[A-Za-z][):\]]|$)'
            embedded_matches = re.findall(embedded_pattern, text)

            if embedded_matches:
                # Extract embedded choices
                for letter, choice_text in embedded_matches:
                    choices[letter.upper()] = choice_text.strip()

                # Remove the choices from the text to get the question
                question_text = re.sub(embedded_pattern, '', text).strip()
                # Clean up any remaining asterisks
                question_text = re.sub(r'\*\*', '', question_text).strip()

            # Strategy 2: Look for line-by-line choices
            if not choices:
                lines = text.split('\n')
                question_lines = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check for various choice formats
                    choice_patterns = [
                        r'^([A-Za-z])[.)\]]\s+(.+)',  # A. text or A) text
                        r'^([A-Za-z]):\s+(.+)',  # A: text
                        r'^([A-Za-z])\s+-\s+(.+)',  # A - text
                    ]

                    matched = False
                    for pattern in choice_patterns:
                        match = re.match(pattern, line)
                        if match:
                            letter = match.group(1).upper()
                            choice_text = match.group(2).strip()
                            choices[letter] = choice_text
                            matched = True
                            break

                    # If not a choice, it's part of the question
                    if not matched and not choices:
                        question_lines.append(line)

                if question_lines:
                    question_text = ' '.join(question_lines)

            # Strategy 3: Handle single N/A option cases
            if not choices and "N/A" in text and "cannot answer" in text.lower():
                # Look for just an N/A option
                na_match = re.search(r'([A-Za-z])[.)\]]\s*N/A\s*-?\s*I cannot answer this question', text,
                                     re.IGNORECASE)
                if na_match:
                    letter = na_match.group(1).upper()
                    choices[letter] = "N/A - I cannot answer this question"
                    # Everything before this is the question
                    question_text = text[:na_match.start()].strip()

            # Clean up question text
            question_text = question_text.strip()
            # Remove any trailing colons or question marks followed by options text
            question_text = re.sub(r'\s*\*\*Options:\*\*\s*$', '', question_text)
            question_text = re.sub(r'\s*Options:\s*$', '', question_text)

            # If we still don't have a proper question, try to extract it differently
            if not question_text and choices:
                # The entire text minus the choices is the question
                remaining_text = text
                for letter, choice in choices.items():
                    # Remove each choice from the text
                    pattern = rf'\*?\*?{re.escape(letter)}[):\]]\*?\*?\s*{re.escape(choice)}'
                    remaining_text = re.sub(pattern, '', remaining_text, flags=re.IGNORECASE)
                question_text = remaining_text.strip()

            # Validate we have at least a question and one choice
            if not question_text or not choices:
                return None

            # Ensure we have the N/A option with key E if not already present
            if "E" not in choices and len(choices) < 5:
                # Find the N/A option if it exists with a different letter
                for letter, choice in list(choices.items()):
                    if "N/A" in choice and "cannot answer" in choice.lower():
                        if letter != "E":
                            # Move it to E
                            choices["E"] = choice
                            if len(choices) > 5:  # Only delete if we have too many choices
                                del choices[letter]
                        break

            task = json_data.get("task", "unknown")
            subject = task.replace("unanswerable-claims_", "") if task.startswith("unanswerable-claims_") else task
            subject = subject.replace("unanswerable-claims", "other")
            return Question(
                question=question_text,
                choices=choices,
                answer="E",
                subject=subject,
                sample_id=json_data.get("sample_id", ""),
                model=model_path,
                did_lie=json_data.get("did_lie", True),
                chosen_answer=json_data.get("metadata", {}).get("chosen_answer", ""),
                evidence=json_data.get("evidence", "")
            )

        except Exception as e:
            print(f"Error extracting question data: {e}")
            return None

    def get_embeddings_batch(self, texts: List[str]) -> Dict[str, List[float]]:
        """Get OpenAI embeddings for a batch of texts."""
        if not self.use_embeddings:
            return {}

        # Check cache and separate cached vs uncached
        text_to_hash = {text: hashlib.md5(text.encode()).hexdigest() for text in texts}
        cached_embeddings = {}
        uncached_texts = []

        for text, text_hash in text_to_hash.items():
            cache_file = self.embeddings_cache_dir / f"{text_hash}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_embeddings[text] = json.load(f)
            else:
                uncached_texts.append(text)

        # Get embeddings for uncached texts in batches
        new_embeddings = {}
        if uncached_texts:
            # OpenAI allows up to 2048 embeddings per request
            batch_size = 100  # Conservative batch size

            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]

                try:
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch
                    )

                    for text, embedding_data in zip(batch, response.data):
                        embedding = embedding_data.embedding
                        new_embeddings[text] = embedding

                        # Cache the embedding
                        text_hash = text_to_hash[text]
                        cache_file = self.embeddings_cache_dir / f"{text_hash}.json"
                        with open(cache_file, 'w') as f:
                            json.dump(embedding, f)

                    # Rate limiting between batches
                    if i + batch_size < len(uncached_texts):
                        time.sleep(0.5)

                except Exception as e:
                    print(f"Error getting embeddings batch: {e}")
                    # Try individual requests as fallback
                    for text in batch:
                        try:
                            response = self.openai_client.embeddings.create(
                                model="text-embedding-3-small",
                                input=text
                            )
                            embedding = response.data[0].embedding
                            new_embeddings[text] = embedding

                            # Cache the embedding
                            text_hash = text_to_hash[text]
                            cache_file = self.embeddings_cache_dir / f"{text_hash}.json"
                            with open(cache_file, 'w') as f:
                                json.dump(embedding, f)

                            time.sleep(0.1)  # Rate limiting for fallback
                        except Exception as e2:
                            print(f"Error getting individual embedding: {e2}")

        # Combine cached and new embeddings
        all_embeddings = {**cached_embeddings, **new_embeddings}
        return all_embeddings

    def embed_questions(self, questions: List[Question]) -> List[Question]:
        """Add embeddings to all questions using batch processing."""
        if not self.use_embeddings:
            return questions

        print("Generating embeddings for questions...")

        # Prepare all texts that need embeddings
        texts_to_embed = []
        question_to_text = {}

        for question in questions:
            if question.embedding is None:
                # Create a text representation of the full question
                full_text = f"{question.question}\n" + "\n".join(
                    [f"{k}. {v}" for k, v in sorted(question.choices.items())]
                )
                texts_to_embed.append(full_text)
                question_to_text[id(question)] = full_text

        if not texts_to_embed:
            print("All questions already have embeddings (from cache)")
            return questions

        print(f"Getting embeddings for {len(texts_to_embed)} questions...")
        print(f"({len(questions) - len(texts_to_embed)} already cached)")

        # Get all embeddings in batches
        embeddings_map = self.get_embeddings_batch(texts_to_embed)

        # Assign embeddings to questions
        for question in questions:
            if question.embedding is None:
                text = question_to_text.get(id(question))
                if text and text in embeddings_map:
                    question.embedding = embeddings_map[text]

        # Report how many successfully embedded
        embedded_count = sum(1 for q in questions if q.embedding is not None)
        print(f"Successfully embedded {embedded_count}/{len(questions)} questions")

        return questions

    def deduplicate_questions(self, questions: List[Question],
                              similarity_threshold: float = 0.95) -> List[Question]:
        """Remove duplicate questions based on embeddings and exact matches."""
        print(f"Deduplicating {len(questions)} questions...")

        # First, remove exact duplicates by question hash
        unique_by_hash = {}
        for q in questions:
            if q.question_hash not in unique_by_hash:
                unique_by_hash[q.question_hash] = q
            else:
                # Keep track of which models had this question
                existing = unique_by_hash[q.question_hash]
                if existing.model != q.model:
                    # Aggregate model information
                    if not hasattr(existing, 'all_models'):
                        existing.all_models = [existing.model]
                    existing.all_models.append(q.model)

        questions = list(unique_by_hash.values())
        print(f"After exact deduplication: {len(questions)} questions")

        # Semantic deduplication using embeddings
        if self.use_embeddings and similarity_threshold < 1.0:
            # Collect embeddings
            embeddings = []
            questions_with_embeddings = []

            for q in questions:
                if q.embedding:
                    embeddings.append(q.embedding)
                    questions_with_embeddings.append(q)

            if embeddings:
                print(f"Performing semantic deduplication on {len(embeddings)} embedded questions...")
                embeddings_array = np.array(embeddings)

                # Compute similarity matrix
                similarity_matrix = cosine_similarity(embeddings_array)

                # Find groups of similar questions
                keep_indices = set()
                processed = set()

                for i in range(len(questions_with_embeddings)):
                    if i in processed:
                        continue

                    # Find all questions similar to this one
                    similar_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]

                    # Keep the first one and mark others as processed
                    keep_indices.add(i)
                    processed.update(similar_indices)

                    # Track which models had similar questions
                    if len(similar_indices) > 1:
                        kept_question = questions_with_embeddings[i]
                        if not hasattr(kept_question, 'similar_from_models'):
                            kept_question.similar_from_models = []

                        for j in similar_indices:
                            if j != i:
                                kept_question.similar_from_models.append(
                                    questions_with_embeddings[j].model
                                )

                # Keep only unique questions
                unique_questions = [questions_with_embeddings[i] for i in sorted(keep_indices)]

                # Add back questions without embeddings
                for q in questions:
                    if not q.embedding:
                        unique_questions.append(q)

                questions = unique_questions
                print(f"After semantic deduplication: {len(questions)} questions")

        return questions

    def download_all_models(self, subjects: List[str] = None,
                            max_workers: int = 10,
                            limit_per_model: int = None,
                            models_filter: List[str] = None) -> List[Question]:
        """Download questions from all models."""

        # Get all models
        all_models = self.list_all_models()

        if models_filter:
            all_models = [m for m in all_models if any(f in m for f in models_filter)]
            print(f"Filtered to {len(all_models)} models")

        all_questions = []

        for model_path in all_models:
            print(f"\nProcessing model: {model_path}")

            # List files for this model
            files_by_subject = self.list_s3_files_for_model(model_path, subjects)

            if not any(files_by_subject.values()):
                print(f"  No unanswerable_claims files found for {model_path}")
                continue

            # Create list of all file info tuples
            model_files = []
            for subject, files in files_by_subject.items():
                for filename in files:
                    model_files.append((model_path, subject, filename))

            # Apply limit if specified
            if limit_per_model:
                model_files = model_files[:limit_per_model]

            print(f"  Processing {len(model_files)} files...")

            # Download and process in parallel
            model_questions = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for file_info in model_files:
                    future = executor.submit(self.download_and_process_file, file_info)
                    futures.append(future)

                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"  {model_path}"):
                    result = future.result()
                    if result:
                        model_questions.append(result)

            print(f"  Got {len(model_questions)} questions from {model_path}")
            all_questions.extend(model_questions)

        return all_questions

    def download_and_process_file(self, file_info: Tuple[str, str, str]) -> Optional[Question]:
        """Download and process a single file."""
        model_path, subject, filename = file_info

        # Check cache first
        cache_file = self.cache_dir / f"{model_path.replace('/', '_')}_{subject}_{filename}"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                json_data = json.load(f)
        else:
            json_data = self.fetch_json_from_s3(model_path, subject, filename)
            if json_data:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
            else:
                return None

        return self.extract_question_data(json_data, model_path)

    def save_deduplicated_dataset(self, questions: List[Question],
                                  filename: str = "deduplicated_questions.jsonl"):
        """Save deduplicated dataset with metadata about duplicates."""
        output_file = self.output_dir / filename

        with open(output_file, 'w') as f:
            for q in questions:
                entry = asdict(q)

                # Add duplicate tracking metadata
                if hasattr(q, 'all_models'):
                    entry['duplicate_in_models'] = q.all_models
                if hasattr(q, 'similar_from_models'):
                    entry['similar_in_models'] = q.similar_from_models

                # Remove embedding from saved data (too large)
                entry.pop('embedding', None)

                f.write(json.dumps(entry) + '\n')

        print(f"Saved {len(questions)} deduplicated questions to {output_file}")
        return output_file

    def save_statistics(self, questions: List[Question]):
        """Save detailed statistics about the dataset."""
        stats = {
            'total_questions': len(questions),
            'models': {},
            'subjects': {},
            'lying_stats': {},
            'duplicates': {'exact': 0, 'semantic': 0}
        }

        # Count by model
        model_counts = defaultdict(int)
        model_lying = defaultdict(lambda: {'lied': 0, 'total': 0})
        subject_counts = defaultdict(int)

        for q in questions:
            model_counts[q.model] += 1
            subject_counts[q.subject] += 1

            model_lying[q.model]['total'] += 1
            if q.did_lie:
                model_lying[q.model]['lied'] += 1

            if hasattr(q, 'all_models'):
                stats['duplicates']['exact'] += len(q.all_models) - 1
            if hasattr(q, 'similar_from_models'):
                stats['duplicates']['semantic'] += len(q.similar_from_models)

        stats['models'] = dict(model_counts)
        stats['subjects'] = dict(subject_counts)

        for model, counts in model_lying.items():
            if counts['total'] > 0:
                stats['lying_stats'][model] = {
                    'lied': counts['lied'],
                    'total': counts['total'],
                    'percentage': 100 * counts['lied'] / counts['total']
                }

        # Save statistics
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nDataset Statistics saved to {stats_file}")
        print(f"Total unique questions: {stats['total_questions']}")
        print(f"Models represented: {len(stats['models'])}")
        print(f"Subjects covered: {len(stats['subjects'])}")
        print(f"Exact duplicates removed: {stats['duplicates']['exact']}")
        if self.use_embeddings:
            print(f"Semantic duplicates removed: {stats['duplicates']['semantic']}")

        return stats

    def prepare_huggingface_dataset(self, questions: List[Question]) -> DatasetDict:
        """Prepare dataset in Hugging Face format."""
        print("Preparing dataset for Hugging Face...")

        # Convert questions to dictionary format suitable for HF
        data_by_subject = defaultdict(list)

        for q in questions:
            # Format choices as a list in order (A, B, C, D, E)
            choices_list = []
            for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                if letter in q.choices:
                    choices_list.append(q.choices[letter])
                elif letter == 'E' and 'E' not in q.choices:
                    # Ensure E is always N/A option if not present
                    choices_list.append("N/A - I cannot answer this question")

            # Create entry in MMLU-like format
            entry = {
                'question': q.question,
                'choices': choices_list,
                'answer': ord(q.answer) - ord('A'),  # Convert letter to index (A=0, B=1, etc.)
                'subject': q.subject,
                'model': q.model,
                'did_lie': q.did_lie,
                'chosen_answer': q.chosen_answer,
                'sample_id': q.sample_id,
            }

            # Add duplicate tracking if present
            if hasattr(q, 'all_models'):
                entry['duplicate_in_models'] = ', '.join(q.all_models)
            if hasattr(q, 'similar_from_models'):
                entry['similar_in_models'] = ', '.join(q.similar_from_models)

            data_by_subject[q.subject].append(entry)

        # Create datasets for each subject
        datasets = {}
        for subject, entries in data_by_subject.items():
            df = pd.DataFrame(entries)
            datasets[subject] = Dataset.from_pandas(df)

        # Also create an 'all' split with all questions
        all_entries = []
        for entries in data_by_subject.values():
            all_entries.extend(entries)
        df_all = pd.DataFrame(all_entries)
        datasets['all'] = Dataset.from_pandas(df_all)

        # Create DatasetDict
        dataset_dict = DatasetDict(datasets)

        print(f"Created dataset with {len(datasets)} splits: {list(datasets.keys())}")
        return dataset_dict

    def create_dataset_card(self, stats: Dict) -> str:
        """Create a README dataset card for Hugging Face."""
        card = f"""---
language:
- en
pretty_name: Unanswerable Questions Dataset
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-classification
configs:
- config_name: default
  data_files:
  - split: all
    path: "all/train-*"
- config_name: by_subject
  data_files:
  - split: science
    path: "science/train-*"
  - split: mathematics
    path: "mathematics/train-*"
  - split: history
    path: "history/train-*"
  - split: philosophy
    path: "philosophy/train-*"
  - split: legal
    path: "legal/train-*"
  - split: art
    path: "art/train-*"
  - split: fiction
    path: "fiction/train-*"
  - split: political
    path: "political/train-*"
tags:
- unanswerable-questions
- llm-evaluation
- truthfulness
- uncertainty
---

# Unanswerable Questions Dataset

## Dataset Description

This dataset contains questions that are designed to be unanswerable by language models, used to test whether models will admit uncertainty or attempt to provide false information.

### Dataset Summary

- **Total Questions**: {stats['total_questions']}
- **Subjects**: {len(stats['subjects'])}
- **Models Tested**: {len(stats['models'])}
- **Format**: Multiple choice with N/A option

### Statistics

#### By Subject:
"""
        for subject, count in sorted(stats['subjects'].items()):
            card += f"- {subject}: {count} questions\n"

        card += f"""

#### Model Performance:
"""
        for model, lying_stats in sorted(stats['lying_stats'].items()):
            card += f"- {model}: {lying_stats['percentage']:.1f}% lying rate ({lying_stats['lied']}/{lying_stats['total']})\n"

        card += f"""

#### Deduplication:
- Exact duplicates removed: {stats['duplicates']['exact']}
- Semantic duplicates removed: {stats['duplicates']['semantic']}

## Dataset Structure

Each example contains:
- `question`: The unanswerable question text
- `choices`: List of answer choices (including N/A option)
- `answer`: Correct answer index (always points to N/A option)
- `subject`: Subject category
- `model`: Model that generated/was tested on this question
- `did_lie`: Whether the model lied when tested
- `chosen_answer`: What the model actually chose
- `sample_id`: Unique identifier
- `duplicate_in_models`: List of models that had exact duplicate (if any)
- `similar_in_models`: List of models with similar questions (if any)

## Usage

```python
from datasets import load_dataset

# Load all questions
dataset = load_dataset("Noddybear/unanswerable")

# Load specific subject
dataset = load_dataset("Noddybear/unanswerable", "science")

# Example usage
for example in dataset['all']:
    print(f"Question: {{example['question']}}")
    print(f"Choices: {{example['choices']}}")
    print(f"Correct Answer: {{example['choices'][example['answer']]}}")
    break
```

## Purpose

This dataset is designed to evaluate:
1. **Truthfulness**: Whether models admit when they don't know something
2. **Calibration**: How well models recognize unanswerable questions
3. **Hallucination**: Tendency to generate false information

## Citation

If you use this dataset, please cite:
```bibtex
@dataset{{unanswerable2024,
  title={{Unanswerable Questions Dataset}},
  author={{Various Contributors}},
  year={{2024}},
  publisher={{Hugging Face}}
}}
```

## License

This dataset is released under the MIT License.
"""
        return card

    def upload_to_huggingface(self, questions: List[Question], stats: Dict,
                              repo_id: str = "Noddybear/unanswerable",
                              private: bool = False):
        """Upload the dataset to Hugging Face Hub."""
        try:
            # Login to Hugging Face
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                print("Warning: HF_TOKEN not found in environment. Please set it to upload.")
                print("You can get your token from: https://huggingface.co/settings/tokens")
                return False

            login(token=hf_token)
            print(f"Logged in to Hugging Face")

            # Prepare dataset
            dataset_dict = self.prepare_huggingface_dataset(questions)

            # Create dataset card
            dataset_card = self.create_dataset_card(stats)

            # Save dataset card locally
            readme_file = self.output_dir / "README.md"
            with open(readme_file, 'w') as f:
                f.write(dataset_card)
            print(f"Dataset card saved to {readme_file}")

            # Upload to Hugging Face
            print(f"Uploading dataset to {repo_id}...")
            dataset_dict.push_to_hub(
                repo_id=repo_id,
                private=private,
                commit_message="Update unanswerable questions dataset with deduplication"
            )

            # Upload README separately to ensure it's the main card
            api = HfApi()
            api.upload_file(
                path_or_fileobj=readme_file,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Update dataset card"
            )

            print(f"✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")
            return True

        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")
            print("Make sure you have the correct permissions and the repo exists.")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Download unanswerable questions from all models and deduplicate"
    )
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to download")
    parser.add_argument("--models-filter", nargs="+",
                        help="Filter models by substring (e.g., 'google', 'anthropic')")
    parser.add_argument("--output-dir", default="unanswerable_dataset_all_models",
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers")
    parser.add_argument("--limit-per-model", type=int,
                        help="Limit files per model (for testing)")
    parser.add_argument("--similarity-threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for deduplication (0-1)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding generation and semantic deduplication")
    parser.add_argument("--upload-to-hf", action="store_true",
                        help="Upload dataset to Hugging Face Hub")
    parser.add_argument("--hf-repo", default="Noddybear/unanswerable",
                        help="Hugging Face repository ID")
    parser.add_argument("--hf-private", action="store_true",
                        help="Make the Hugging Face dataset private")

    args = parser.parse_args()

    # Initialize downloader
    downloader = UnanswerableDatasetDownloader(
        output_dir=args.output_dir,
        use_embeddings=not args.no_embeddings
    )

    # Download from all models
    print("Starting download from all models...")
    all_questions = downloader.download_all_models(
        subjects=args.subjects,
        max_workers=args.workers,
        limit_per_model=args.limit_per_model,
        models_filter=args.models_filter
    )
    upload_to_hf = True
    if all_questions:
        print(f"\nTotal questions downloaded: {len(all_questions)}")

        # Add embeddings if enabled
        if not args.no_embeddings:
            all_questions = downloader.embed_questions(all_questions)

        # Deduplicate
        unique_questions = downloader.deduplicate_questions(
            all_questions,
            similarity_threshold=args.similarity_threshold
        )

        # Save deduplicated dataset
        downloader.save_deduplicated_dataset(unique_questions)

        # Save statistics
        stats = downloader.save_statistics(unique_questions)

        # Save sample for verification
        sample_file = downloader.output_dir / "sample_deduplicated.json"
        sample_questions = unique_questions[:10] if len(unique_questions) > 10 else unique_questions
        with open(sample_file, 'w') as f:
            sample_data = []
            for q in sample_questions:
                entry = asdict(q)
                entry.pop('embedding', None)
                sample_data.append(entry)
            json.dump(sample_data, f, indent=2)
        print(f"\nSample saved to {sample_file}")

        print(f"\nDeduplication complete!")
        print(f"Original: {len(all_questions)} questions")
        print(f"After deduplication: {len(unique_questions)} questions")
        print(f"Removed: {len(all_questions) - len(unique_questions)} duplicates")

        # Upload to Hugging Face if requested
        if upload_to_hf:
            print(f"\n{'=' * 60}")
            print("Uploading to Hugging Face Hub...")
            print(f"{'=' * 60}")
            success = downloader.upload_to_huggingface(
                unique_questions,
                stats,
                repo_id=args.hf_repo,
                private=args.hf_private
            )
            if success:
                print(f"\n🎉 Dataset successfully uploaded to Hugging Face!")
                print(f"View at: https://huggingface.co/datasets/{args.hf_repo}")
            else:
                print("\n⚠️ Upload to Hugging Face failed. Please check the error messages above.")

    else:
        print("No data downloaded")


if __name__ == "__main__":
    main()