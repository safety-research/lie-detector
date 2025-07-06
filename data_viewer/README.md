# Trace Data Viewer

A simple Flask web application for viewing lie detection trace data.

## Quick Start

### Running the App

1. Install dependencies:
```bash
pip install flask boto3
```

2. Run the app:
```bash
python app.py
```

3. Open your browser and go to: `http://localhost:8080`

### Stopping the App

- Press `Ctrl+C` in the terminal where the app is running
- Or kill the process: `pkill -f "python app.py"`

## Configuration

The app loads data from S3 by default. To use local files for testing:

1. Edit `app.py` and change the configuration at the top:
```python
USE_LOCAL_DATA = True  # Set to True for local files
LOCAL_DATA_PATH = "./your_data_folder"  # Path to your local data
```

2. Organize your local data like this:
```
your_data_folder/
├── task1/
│   └── data.jsonl
└── task2/
    └── data.jsonl
```

## Data Format

The app expects JSONL files with one JSON object per line:
```json
{"sample_id": "1", "task": "example", "trace": [...], "did_lie": true}
{"sample_id": "2", "task": "example", "trace": [...], "did_lie": false}
``` 