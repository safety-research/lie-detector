# Trace Data Viewer

A Flask web application for viewing and exploring trace data with lie detection annotations from S3 or local files.

## Features

- **S3 Integration**: Load data directly from S3 bucket `dipika-lie-detection-data`
- **Folder Tagging**: Automatically tag data based on S3 folder names (e.g., "n-truths-and-a-lie", "alibi")
- **Multi-source Support**: Load from S3 or local JSON files
- **Folder Filtering**: Filter samples by folder/task type
- **Statistics Dashboard**: View statistics by folder including average lies and turns
- **Structured Conversations**: View turn-by-turn breakdown with speaker identification
- **Lie Annotations**: See lie annotations highlighted in the text
- **Expandable Sample Cards**: Easy navigation through large datasets
- **Random Sampling**: Explore data with random sampling

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials for S3 access (see [AWS Setup Guide](aws_setup.md)):
```bash
aws configure
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:9009`

3. Choose your data source:
   - **S3 Data**: Select a specific folder or load all folders from S3
   - **Local File**: Enter the path to a local JSON file

4. Use the controls to:
   - Filter samples by folder/task type
   - Load random samples
   - View folder statistics
   - Explore individual samples

## S3 Data Structure

The application expects data in the following S3 structure:
```
s3://dipika-lie-detection-data/generated-data/
├── n-truths-and-a-lie/
│   ├── file1.json
│   └── file2.json
├── alibi/
│   ├── file3.json
│   └── file4.json
└── other-task/
    └── file5.json
```

Each folder name becomes a tag for the data, making it easy to filter and analyze by task type.

## Data Format

The application expects JSON files with the following structure:
```json
{
  "samples": [
    {
      "store": {
        "GameState:interrogation": [
          {
            "turn_number": 0,
            "speaker": "blue",
            "message": "..."
          }
        ]
      },
      "scores": {
        "lie_detection_scorer": {
          "metadata": {
            "all_lies": [
              {
                "turn_number": 1,
                "utterance": "..."
              }
            ]
          }
        }
      }
    }
  ]
}
```

## API Endpoints

- `GET /` - Main web interface
- `GET /list_folders` - List available S3 folders
- `POST /load_data_from_s3` - Load data from S3
- `POST /load_data` - Load data from local file
- `GET /get_samples` - Get random samples (with optional folder filtering)
- `GET /get_sample/<id>` - Get specific sample by ID
- `GET /get_folder_stats` - Get statistics by folder

## File Structure

```
data_viewer/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── aws_setup.md       # AWS credentials setup guide
├── SETUP.md           # Deployment setup
├── cloudflare_setup.md # Cloudflare configuration
├── ngrok_setup.md     # Ngrok tunnel setup
├── docker-compose.yml # Docker configuration
├── Dockerfile         # Docker image definition
├── gunicorn_config.py # Gunicorn server configuration
├── nginx_config.conf  # Nginx configuration
├── lie-detector-viewer.service # Systemd service
└── templates/
    └── index.html     # Web interface
```

## Folder Statistics

The application provides statistics for each folder including:
- Number of samples
- Average number of lies per sample
- Average number of turns per sample
- Total lies and turns across all samples

## Filtering and Navigation

- **Folder Filter**: Filter samples by specific task type (folder name)
- **Sample Count**: Control how many random samples to load
- **Expandable Cards**: Click on sample headers to expand/collapse details
- **Multiple Views**: 
  - Lies Only: Shows only turns containing lies
  - Full Trace: Complete conversation with lie highlighting
  - Lie Metadata: Detailed information about detected lies 