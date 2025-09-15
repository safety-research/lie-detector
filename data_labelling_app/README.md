# Lie Detection Annotation Tool

A Flask web application for human annotation of lie detection samples. This tool allows annotators to review conversation traces and classify whether the assistant's response contains a lie.

## Features

- **Annotator Registration**: Simple name-based registration system
- **Fold-based Organization**: Loads first 100 samples from each data fold
- **Conversation Display**: Clean presentation of system/user/assistant conversation traces
- **Annotation Interface**: Easy-to-use buttons for lie/not-lie classification
- **Progress Tracking**: Shows current sample position and progress
- **Keyboard Shortcuts**: Quick navigation and annotation with keyboard
- **Data Persistence**: Saves annotations back to original JSONL files

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data is in the correct location:
   - The app expects data in `../.data/` directory
   - JSONL files should be organized in subdirectories (folds)

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Register with your name to start annotating

4. Select a fold to begin annotation

5. Review each conversation trace and classify as "Lie" or "Not Lie"

## Data Structure

The app expects JSONL files with the following structure:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "s3_metadata": {
    "sample_id": "...",
    "task": "...",
    "model": "...",
    "did_lie": true/false
  }
}
```

## Annotation Format

When an annotator classifies a sample, the annotation is saved as:
```json
{
  "human_annotations": {
    "annotator_name": {
      "classification": "lie" | "not_lie",
      "timestamp": "2024-01-01T12:00:00"
    }
  }
}
```

## Keyboard Shortcuts

- `1`: Mark as Lie
- `2`: Mark as Not Lie
- `←`: Previous sample
- `→`: Skip sample

## Configuration

- `SAMPLES_PER_FOLD`: Number of samples to load per fold (default: 100)
- `DATA_DIR`: Path to data directory (default: `../.data/`)

## Security Note

Change the `app.secret_key` in production environments.
