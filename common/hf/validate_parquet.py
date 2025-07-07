#!/usr/bin/env python3
"""
Validate and fix Parquet files for Hugging Face compatibility.

This script checks for common issues that cause UnexpectedApiError on HF:
- Column name issues (spaces, special characters)
- Data type inconsistencies
- Null/NaN values
- File size and structure issues

Usage:
    python validate_parquet.py --input-dir ./parquet_output
    python validate_parquet.py --input-dir ./parquet_output --fix
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import json
import argparse
from pathlib import Path
import logging
import re
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sanitize_column_name(name: str) -> str:
    """
    Sanitize column names for HF compatibility.
    - Replace spaces with underscores
    - Remove special characters
    - Ensure valid Python variable names
    """
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s\-]+', '_', name)
    # Remove any other special characters except underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = f"col_{name}"
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove trailing underscores
    name = name.strip('_')
    return name or "unnamed_column"


def validate_parquet_file(file_path: Path) -> Dict[str, Any]:
    """Validate a single Parquet file and return issues found."""
    issues = {
        'file': str(file_path),
        'errors': [],
        'warnings': [],
        'info': {}
    }

    try:
        # Read the Parquet file
        df = pd.read_parquet(file_path)
        table = pq.read_table(file_path)

        issues['info']['rows'] = len(df)
        issues['info']['columns'] = len(df.columns)
        issues['info']['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)

        # Check file size
        if issues['info']['file_size_mb'] > 5000:  # 5GB
            issues['warnings'].append(f"Large file size: {issues['info']['file_size_mb']:.2f} MB")

        # Check column names
        problematic_columns = []
        for col in df.columns:
            if col != sanitize_column_name(col):
                problematic_columns.append(col)

        if problematic_columns:
            issues['errors'].append(f"Problematic column names: {problematic_columns}")

        # Check for completely empty columns
        empty_columns = [col for col in df.columns if df[col].isna().all()]
        if empty_columns:
            issues['warnings'].append(f"Empty columns: {empty_columns}")

        # Check data types
        for col in df.columns:
            dtype = str(df[col].dtype)

            # Check for mixed types
            if dtype == 'object':
                # Sample some non-null values
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    types = set(type(x).__name__ for x in sample)
                    if len(types) > 1:
                        issues['warnings'].append(f"Column '{col}' has mixed types: {types}")

            # Check for Python objects that might not serialize well
            if dtype == 'object':
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample_val is not None and not isinstance(sample_val, (str, int, float, bool)):
                    issues['errors'].append(f"Column '{col}' contains complex Python objects")

        # Check schema
        schema = table.schema
        for i, field in enumerate(schema):
            if field.type == pa.null():
                issues['errors'].append(f"Column '{field.name}' has null type")

        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            issues['errors'].append("Duplicate column names found")

        # Check if any column names are reserved words
        reserved_words = {'id', 'index', '__index__', '__row__'}
        reserved_found = [col for col in df.columns if col.lower() in reserved_words]
        if reserved_found:
            issues['warnings'].append(f"Reserved column names found: {reserved_found}")

    except Exception as e:
        issues['errors'].append(f"Failed to read file: {str(e)}")

    return issues


def fix_parquet_file(file_path: Path, output_path: Path = None) -> bool:
    """Fix common issues in Parquet files."""
    if output_path is None:
        output_path = file_path.parent / f"{file_path.stem}_fixed.parquet"

    try:
        # Read the file
        df = pd.read_parquet(file_path)
        original_cols = df.columns.tolist()

        # Fix column names
        new_columns = {}
        for col in df.columns:
            new_name = sanitize_column_name(col)
            # Handle duplicates
            if new_name in new_columns.values():
                counter = 1
                while f"{new_name}_{counter}" in new_columns.values():
                    counter += 1
                new_name = f"{new_name}_{counter}"
            new_columns[col] = new_name

        df = df.rename(columns=new_columns)

        # Log column name changes
        for old, new in new_columns.items():
            if old != new:
                logger.info(f"  Renamed column: '{old}' -> '{new}'")

        # Fix data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to infer better types
                try:
                    # Check if all non-null values are strings
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        if all(isinstance(x, str) for x in non_null.head(100)):
                            # It's already strings, good
                            pass
                        else:
                            # Convert everything to strings
                            df[col] = df[col].apply(
                                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x) if pd.notna(
                                    x) else None)
                            logger.info(f"  Converted column '{col}' to strings")
                except Exception as e:
                    logger.warning(f"  Could not process column '{col}': {e}")

        # Remove completely empty columns
        empty_cols = [col for col in df.columns if df[col].isna().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            logger.info(f"  Removed empty columns: {empty_cols}")

        # Fill NaN values in string columns with empty strings
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')

        # Create a clean schema
        schema_fields = []
        for col in df.columns:
            if df[col].dtype == 'object':
                pa_type = pa.string()
            elif df[col].dtype == 'int64':
                pa_type = pa.int64()
            elif df[col].dtype == 'float64':
                pa_type = pa.float64()
            elif df[col].dtype == 'bool':
                pa_type = pa.bool_()
            else:
                pa_type = pa.string()  # Default to string

            schema_fields.append(pa.field(col, pa_type))

        schema = pa.schema(schema_fields)

        # Write the fixed file
        table = pa.Table.from_pandas(df, schema=schema)
        pq.write_table(
            table,
            output_path,
            compression='snappy',
            use_dictionary=True,
            version='2.6'  # Use a compatible version
        )

        logger.info(f"  Fixed file saved to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"  Failed to fix file: {e}")
        return False


def create_hf_compatible_parquet(input_dir: Path, output_dir: Path = None):
    """Create HF-compatible versions of all Parquet files."""
    if output_dir is None:
        output_dir = input_dir / "hf_compatible"

    output_dir.mkdir(exist_ok=True)

    parquet_files = list(input_dir.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} Parquet files")

    success_count = 0
    for file_path in parquet_files:
        if '_fixed' in file_path.stem:
            continue

        logger.info(f"\nProcessing {file_path.name}")
        output_path = output_dir / file_path.name

        if fix_parquet_file(file_path, output_path):
            success_count += 1

    logger.info(f"\nSuccessfully processed {success_count}/{len(parquet_files)} files")
    logger.info(f"HF-compatible files saved to: {output_dir}")

    # Create a simple dataset_info.json
    dataset_info = {
        "dataset_name": "converted_dataset",
        "dataset_size": sum(f.stat().st_size for f in output_dir.glob("*.parquet")),
        "download_size": sum(f.stat().st_size for f in output_dir.glob("*.parquet")),
        "features": {},
        "splits": {
            "train": {
                "num_examples": sum(len(pd.read_parquet(f)) for f in output_dir.glob("*.parquet")),
            }
        }
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Validate and fix Parquet files for HF")
    parser.add_argument("--input-dir", type=Path, default=Path("./parquet_output"),  help="Directory with Parquet files")
    parser.add_argument("--fix", action="store_true", help="Fix issues found")
    parser.add_argument("--output-dir", type=Path, default=Path("./parquet_output"), help="Output directory for fixed files")

    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Input directory {args.input_dir} does not exist")
        return

    parquet_files = list(args.input_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No Parquet files found in {args.input_dir}")
        return

    # Validate all files
    all_issues = []
    for file_path in parquet_files:
        logger.info(f"\nValidating {file_path.name}...")
        issues = validate_parquet_file(file_path)
        all_issues.append(issues)

        # Print issues
        if issues['errors']:
            logger.error(f"  Errors: {len(issues['errors'])}")
            for error in issues['errors']:
                logger.error(f"    - {error}")

        if issues['warnings']:
            logger.warning(f"  Warnings: {len(issues['warnings'])}")
            for warning in issues['warnings']:
                logger.warning(f"    - {warning}")

        if not issues['errors'] and not issues['warnings']:
            logger.info("  ✓ No issues found")

    # Summary
    total_errors = sum(len(i['errors']) for i in all_issues)
    total_warnings = sum(len(i['warnings']) for i in all_issues)

    logger.info(f"\nSummary: {total_errors} errors, {total_warnings} warnings")

    if args.fix and total_errors > 0:
        logger.info("\nFixing issues...")
        create_hf_compatible_parquet(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()