"""
Download functionality for Lie Detection Data Viewer

This module handles CSV download of filtered data.
"""

import csv
import io
import json
from datetime import datetime
from flask import request, jsonify, make_response


def download_filtered_data(get_cached_file_list, get_file_metadata_only, load_sample_from_file):
    """Download filtered data as CSV"""
    try:
        # Get filter parameters
        task = request.args.get('task')
        model = request.args.get('model')
        provider = request.args.get('provider')
        domain = request.args.get('domain')
        did_lie = request.args.get('did_lie')
        limit = request.args.get('limit')  # Optional limit for testing
        
        # Get cached file list
        all_files = get_cached_file_list()
        if not all_files:
            return jsonify({"error": "No files found"}), 400
        
        # Filter files based on parameters
        filtered_files = []
        for file_key in all_files:
            file_metadata = get_file_metadata_only(file_key)
            
            # Apply filters
            if task and file_metadata.get('task', '').replace('_', ' ') != task:
                continue
            if model and file_metadata.get('full_model', '') != model:
                continue
            if provider and file_metadata.get('provider', '') != provider:
                continue
            if domain and file_metadata.get('domain', '').replace('_', ' ') != domain:
                continue
            if did_lie is not None:
                did_lie_bool = did_lie.lower() == 'true'
                if file_metadata.get('did_lie') != did_lie_bool:
                    continue
            
            filtered_files.append(file_key)
            
            # Apply limit if specified (for testing)
            if limit and len(filtered_files) >= int(limit):
                break
        
        # Load all filtered samples
        samples = []
        for file_key in filtered_files:
            sample = load_sample_from_file(file_key)
            if sample:
                samples.append(sample)
        
        if not samples:
            return jsonify({"error": "No data found with the specified filters"}), 404
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        if samples:
            # Get all possible fields from the first sample
            all_fields = set()
            for sample in samples:
                all_fields.update(sample.keys())
            
            # Write header
            writer.writerow(list(all_fields))
            
            # Write data rows
            for sample in samples:
                row = []
                for field in all_fields:
                    value = sample.get(field, '')
                    # Convert complex objects to JSON strings
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    row.append(str(value))
                writer.writerow(row)
        
        # Create filename with filter info
        filter_parts = []
        if task:
            filter_parts.append(f"task_{task.replace(' ', '_')}")
        if model:
            filter_parts.append(f"model_{model.replace(' ', '_')}")
        if provider:
            filter_parts.append(f"provider_{provider}")
        if domain:
            filter_parts.append(f"domain_{domain.replace(' ', '_')}")
        if did_lie is not None:
            filter_parts.append(f"did_lie_{did_lie}")
        
        filename = f"lie_detection_data_{'_'.join(filter_parts)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Exception as e:
        print(f"Error in download_filtered_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error downloading data: {str(e)}"}), 500 