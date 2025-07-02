from flask import Flask, render_template, request, jsonify
import json
import os
import random
from datetime import datetime

app = Flask(__name__)

# Global variable to store processed data
processed_data = []

def clean_trace_full_turns(trace):
    """Define a clean_trace function that keeps full speaker/turn structure"""
    return [
        {"turn_number": msg["turn_number"], msg["speaker"]: msg["message"]}
        for msg in trace
    ]

# def tag_lies(trace, lies):
#     """Tag lies in the trace with <lie> tags"""
#     tagged_trace = trace.copy()
#     for lie in lies:
#         i = lie["turn_number"]
#         utterance = lie["utterance"]
#         if i >= len(tagged_trace):
#             continue  # skip if turn index is out of range
#         new_turn = tagged_trace[i].copy()
#         for speaker in ["blue", "red"]:
#             if speaker in new_turn:
#                 if utterance in new_turn[speaker]:
#                     new_turn[speaker] = new_turn[speaker].replace(
#                         utterance, f"<lie>{utterance}</lie>"
#                     )
#         tagged_trace[i] = new_turn
#     return tagged_trace


def tag_lies(trace, lies):
    tagged_trace = []
    for turn in trace:
        new_turn = turn.copy()
        for speaker in ["blue", "red"]:
            if speaker in new_turn:
                for lie in lies:
                    utterance = lie["utterance"]
                    if utterance in new_turn[speaker]:
                        new_turn[speaker] = new_turn[speaker].replace(
                            utterance, f"<lie>{utterance}</lie>"
                        )
        tagged_trace.append(new_turn)
    return tagged_trace

def filter_lie_turns(trace, lies):
    """Filter trace to only show turns that contain lies"""
    lie_turn_numbers = {lie["turn_number"] for lie in lies}
    return [turn for i, turn in enumerate(trace) if i in lie_turn_numbers]



def load_and_process_data(file_path):
    """Load and process the JSON file"""
    global processed_data
    
    try:
        # Load the JSON file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract all samples
        samples = data["samples"]
        
        # Process each sample
        processed_samples = []
        for i, sample in enumerate(samples):
            # Extract raw trace
            raw_trace = sample["store"]["GameState:interrogation"]
            
            # Clean trace to structured format
            structured_trace = clean_trace_full_turns(raw_trace)
            
            # Extract lie metadata
            lie_metadata = sample.get("scores", {}).get("lie_detection_scorer", {}).get("metadata", {}).get("all_lies", [])
            
            # Tag lies in the trace
            tagged_trace = tag_lies(structured_trace, lie_metadata)
            
            # Filter to only show turns with lies
            lie_only_trace = filter_lie_turns(tagged_trace, lie_metadata)
            
            # Create sample object
            processed_sample = {
                "id": i,
                "raw_trace": raw_trace,
                "structured_trace": structured_trace,
                "tagged_trace": tagged_trace,
                "lie_only_trace": lie_only_trace,
                "lie_metadata": lie_metadata,
                "num_lies": len(lie_metadata),
                "num_turns": len(structured_trace)
            }
            processed_samples.append(processed_sample)
        
        processed_data = processed_samples
        return len(processed_samples)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return 0

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    """Load data from a file"""
    file_path = request.json.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400
    
    num_samples = load_and_process_data(file_path)
    return jsonify({
        "success": True,
        "num_samples": num_samples,
        "message": f"Loaded {num_samples} samples"
    })

@app.route('/get_samples', methods=['GET'])
def get_samples():
    """Get random samples"""
    n = request.args.get('n', 5, type=int)
    if not processed_data:
        return jsonify({"error": "No data loaded"}), 400
    
    # Get random samples
    if n >= len(processed_data):
        samples = processed_data
    else:
        samples = random.sample(processed_data, n)
    
    return jsonify({
        "samples": samples,
        "total_samples": len(processed_data)
    })

@app.route('/get_sample/<int:sample_id>')
def get_sample(sample_id):
    """Get a specific sample by ID"""
    if not processed_data or sample_id >= len(processed_data):
        return jsonify({"error": "Sample not found"}), 404
    
    return jsonify(processed_data[sample_id])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9009) 