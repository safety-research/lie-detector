# Lie Detection Data Viewer

A Flask web application for viewing and filtering lie detection data from JSON/JSONL files.

## Architecture

This application uses a **local-first approach** with background S3 synchronization:

- **Web App** (`data_viewer/`): Fast Flask app that reads from local files
- **Sync Script** (`sync_s3_data.py`): Background process that syncs S3 data to local directory
- **Local Cache** (`../local_data/`): Local directory containing synced S3 data

## Setup

### 1. Install Dependencies

```bash
pip install flask boto3
```

### 2. Configure AWS Credentials

Make sure you have AWS credentials configured for S3 access:

```bash
aws configure
```

### 3. Start the S3 Sync Script

Run the sync script in a separate terminal to keep local data fresh:

```bash
cd /home/ec2-user/lie-detector
python sync_s3_data.py
```

This script will:
- Download all data from S3 bucket `dipika-lie-detection-data` under prefix `processed-data/`
- Store files in `../local_data/` (sibling to data_viewer directory)
- Sync every 10 seconds automatically
- Show sync progress and timestamps

### 4. Start the Web Application

In another terminal, start the Flask app:

```bash
cd /home/ec2-user/lie-detector/data_viewer
python app.py
```

The app will be available at `http://your-ec2-ip:8080`

## Features

- **Fast Performance**: Reads from local files instead of S3
- **Real-time Updates**: Background sync keeps data fresh every 10 seconds
- **Filtering**: Filter by task, model, and lie status
- **Caching**: 5-minute cache to avoid repeated file reads
- **Manual Refresh**: Button to force reload data
- **Auto-loading**: Data loads automatically on page load

## API Endpoints

- `GET /` - Main page
- `GET /get_samples?n=5&task=...&model=...&did_lie=...` - Get filtered samples
- `GET /get_unique_values` - Get available filter values
- `POST /refresh_data` - Manually reload data
- `GET /get_sample/<id>` - Get specific sample by ID

## Directory Structure

```
lie-detector/
├── data_viewer/
│   ├── app.py              # Flask web application
│   ├── templates/
│   │   └── index_new.html  # Web interface
│   └── README.md
├── sync_s3_data.py         # S3 sync script
└── local_data/             # Local cache (created by sync script)
    ├── task1/
    ├── task2/
    └── ...
```

## Configuration

You can modify these settings in the respective files:

- **Sync Interval**: Change `SYNC_INTERVAL` in `sync_s3_data.py`
- **Cache Duration**: Change `CACHE_DURATION` in `data_viewer/app.py`
- **S3 Bucket**: Change `S3_BUCKET` in `sync_s3_data.py`

## Troubleshooting

- **No data showing**: Make sure the sync script is running and has downloaded files
- **Slow performance**: Check that local_data directory exists and contains files
- **S3 errors**: Verify AWS credentials are configured correctly 