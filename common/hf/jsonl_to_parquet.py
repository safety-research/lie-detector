#!/usr/bin/env python3
"""
Convert JSONL files to Parquet format for Hugging Face data viewer.

Usage:
    python jsonl_to_parquet.py
    python jsonl_to_parquet.py --data-dir ./data --output-dir ./parquet_output
    python jsonl_to_parquet.py --data-dir ./data --single-file
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error in {file_path} line {line_num}: {e}")
    return records


def prepare_records_for_parquet(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare records for Parquet by converting complex types to strings.
    This ensures compatibility with Hugging Face data viewer.
    """
    processed_records = []

    for record in records:
        processed = {}

        for key, value in record.items():
            if key == 'metadata':
                continue
            if isinstance(value, (dict, list)):
                # Convert complex types to JSON strings
                processed[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                # Handle None values
                processed[key] = ""
            else:
                # Keep simple types as-is
                processed[key] = value

        processed_records.append(processed)

    return processed_records


def create_parquet_schema(df: pd.DataFrame) -> pa.Schema:
    """
    Create an appropriate PyArrow schema for the DataFrame.
    This helps ensure proper data types in the Parquet file.
    """
    fields = []

    for column in df.columns:
        # Check the dtype and map to PyArrow types
        dtype = df[column].dtype

        if dtype == 'object':
            # String type
            pa_type = pa.string()
        elif dtype == 'int64':
            pa_type = pa.int64()
        elif dtype == 'float64':
            pa_type = pa.float64()
        elif dtype == 'bool':
            pa_type = pa.bool_()
        else:
            # Default to string for safety
            pa_type = pa.string()

        fields.append(pa.field(column, pa_type))

    return pa.schema(fields)


def convert_to_parquet(records: List[Dict[str, Any]], output_path: Path, file_name: str = None) -> None:
    """Convert records to a Parquet file."""
    if not records:
        logger.warning(f"No records to write for {file_name or output_path}")
        return

    # Prepare records
    processed_records = prepare_records_for_parquet(records)

    # Create DataFrame
    df = pd.DataFrame(processed_records)

    # Fill NaN values with empty strings for string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')

    # Log info
    logger.info(f"Writing {len(df)} records with {len(df.columns)} columns to {output_path}")
    if file_name:
        logger.info(f"Source: {file_name}")

    # Create PyArrow table with schema
    schema = create_parquet_schema(df)
    table = pa.Table.from_pandas(df, schema=schema)

    # Write Parquet file with compression
    pq.write_table(
        table,
        output_path,
        compression='snappy',
        use_dictionary=True,
        compression_level=None
    )


def create_dataset_info(output_dir: Path, total_records: int, file_count: int) -> None:
    """Create a simple README for the dataset."""
    readme_content = f"""# Dataset Information

This dataset was converted from JSONL files to Parquet format for Hugging Face.

## Statistics
- Total records: {total_records:,}
- Source files: {file_count}
- Format: Parquet (snappy compression)

## Usage

You can load this dataset using the Hugging Face datasets library:

```python
from datasets import load_dataset

# If uploaded to Hugging Face Hub
dataset = load_dataset("Noddybear/lies")

# Or load locally
dataset = load_dataset("parquet", data_files="/data/*.parquet")
```
```
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    logger.info(f"Created README at {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL files to Parquet for Hugging Face")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"),
                        help="Directory containing JSONL files (default: ./data)")
    parser.add_argument("--output-dir", type=Path, default=Path("./parquet_output"),
                        help="Output directory for Parquet files (default: ./parquet_output)")
    parser.add_argument("--single-file", action="store_true",
                        help="Combine all JSONL files into a single Parquet file")
    parser.add_argument("--max-rows-per-file", type=int, default=None,
                        help="Maximum rows per Parquet file (for splitting large datasets)")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSONL files
    jsonl_files = list(args.data_dir.glob("*.jsonl"))

    if not jsonl_files:
        logger.error(f"No JSONL files found in {args.data_dir}")
        return

    logger.info(f"Found {len(jsonl_files)} JSONL files in {args.data_dir}")

    if args.single_file:
        # Combine all files into one Parquet file
        all_records = []

        for jsonl_file in jsonl_files:
            logger.info(f"Reading {jsonl_file.name}")
            records = read_jsonl_file(jsonl_file)
            all_records.extend(records)
            logger.info(f"  Added {len(records)} records")

        if args.max_rows_per_file and len(all_records) > args.max_rows_per_file:
            # Split into multiple files
            for i in range(0, len(all_records), args.max_rows_per_file):
                chunk = all_records[i:i + args.max_rows_per_file]
                output_path = args.output_dir / f"data_{i // args.max_rows_per_file:04d}.parquet"
                convert_to_parquet(chunk, output_path, "combined")
        else:
            # Single output file
            output_path = args.output_dir / "data.parquet"
            convert_to_parquet(all_records, output_path, "combined")

        total_records = len(all_records)

    else:
        # Convert each JSONL file to a separate Parquet file
        total_records = 0

        for jsonl_file in jsonl_files:
            logger.info(f"Processing {jsonl_file.name}")
            records = read_jsonl_file(jsonl_file)
            total_records += len(records)
            stem = "_".join(jsonl_file.stem.split("_")[:-1])
            name = stem + ".jsonl"
            if records:
                # Create output filename (replace .jsonl with .parquet)
                output_name = stem + ".parquet"
                output_path = args.output_dir / output_name

                if args.max_rows_per_file and len(records) > args.max_rows_per_file:
                    # Split large files
                    for i in range(0, len(records), args.max_rows_per_file):
                        chunk = records[i:i + args.max_rows_per_file]
                        chunk_path = args.output_dir / f"{jsonl_file.stem}_{i // args.max_rows_per_file:04d}.parquet"
                        convert_to_parquet(chunk, chunk_path, name)
                else:
                    convert_to_parquet(records, output_path, name)

    # Create dataset info
    create_dataset_info(args.output_dir, total_records, len(jsonl_files))

    logger.info(f"\nConversion complete!")
    logger.info(f"Total records processed: {total_records:,}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"\nTo upload to Hugging Face:")
    logger.info(f"1. Create a new dataset repository on https://huggingface.co/new-dataset")
    logger.info(f"2. Clone it: git clone https://huggingface.co/datasets/YOUR_USERNAME/YOUR_DATASET")
    logger.info(f"3. Copy the Parquet files: cp {args.output_dir}/*.parquet YOUR_DATASET/")
    logger.info(f"4. Copy the README: cp {args.output_dir}/README.md YOUR_DATASET/")
    logger.info(f"5. Commit and push: cd YOUR_DATASET && git add . && git commit -m 'Add data' && git push")


if __name__ == "__main__":
    main()