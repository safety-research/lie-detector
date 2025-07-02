# Trace Data Viewer

A Flask web application for viewing and exploring trace data with lie detection annotations.

## Features

- Load JSON files containing trace data
- View structured conversations with turn-by-turn breakdown
- See lie annotations highlighted in the text
- Expandable sample cards for easy navigation
- Random sampling of data for exploration

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter the path to your JSON file in the input field and click "Load Data"

4. Use the controls to load random samples and explore the data

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

## File Structure

```
data_viewer/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── templates/
    └── index.html     # Web interface
``` 