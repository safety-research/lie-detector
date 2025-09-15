import os
import json
import glob
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Configuration
DATA_DIR = Path(__file__).parent.parent / ".data"
SAMPLES_PER_FOLD = 100

class DataManager:
    def __init__(self):
        self.samples = {}
        self.current_annotator = None
        self.current_sample_index = 0
        self.current_fold = None
        self.load_data()
    
    def load_data(self):
        """Load first 100 samples from each fold"""
        self.samples = {}
        
        # Find all JSONL files in the data directory
        jsonl_files = list(DATA_DIR.rglob("*.jsonl"))
        
        for jsonl_file in jsonl_files:
            # Extract fold name from path (e.g., "self-sycophancy", "sycophancy")
            fold_name = jsonl_file.parent.name
            
            if fold_name not in self.samples:
                self.samples[fold_name] = []
            
            # Load samples from this file
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num >= SAMPLES_PER_FOLD:
                        break
                    try:
                        sample = json.loads(line.strip())
                        self.samples[fold_name].append(sample)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.samples)} folds with samples")
        for fold, samples in self.samples.items():
            print(f"  {fold}: {len(samples)} samples")
    
    def get_sample(self, fold_name, sample_index):
        """Get a specific sample"""
        if fold_name in self.samples and 0 <= sample_index < len(self.samples[fold_name]):
            return self.samples[fold_name][sample_index]
        return None
    
    def save_annotation(self, fold_name, sample_index, annotator_name, classification):
        """Save human annotation to sample"""
        sample = self.get_sample(fold_name, sample_index)
        if sample:
            if 'human_annotations' not in sample:
                sample['human_annotations'] = {}
            
            sample['human_annotations'][annotator_name] = {
                'classification': classification,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update the in-memory data
            if fold_name in self.samples and 0 <= sample_index < len(self.samples[fold_name]):
                self.samples[fold_name][sample_index] = sample
            
            # Save back to file
            self._save_sample_to_file(fold_name, sample_index, sample)
            return True
        return False
    
    def _save_sample_to_file(self, fold_name, sample_index, sample):
        """Save sample back to the original JSONL file"""
        # Find the original file for this fold
        jsonl_files = list(DATA_DIR.rglob("*.jsonl"))
        target_file = None
        
        for jsonl_file in jsonl_files:
            if jsonl_file.parent.name == fold_name:
                target_file = jsonl_file
                break
        
        if target_file:
            # Read all lines, update the specific line, write back
            with open(target_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if sample_index < len(lines):
                lines[sample_index] = json.dumps(sample, ensure_ascii=False) + '\n'
                
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

# Global data manager
data_manager = DataManager()

@app.route('/')
def index():
    """Home page - show available folds"""
    if 'annotator_name' not in session:
        return redirect(url_for('register'))
    
    folds = list(data_manager.samples.keys())
    return render_template('index.html', folds=folds, annotator_name=session['annotator_name'], data_manager=data_manager)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register annotator"""
    if request.method == 'POST':
        annotator_name = request.form.get('annotator_name', '').strip().lower()
        if annotator_name:
            session['annotator_name'] = annotator_name
            session['current_fold'] = None
            session['current_sample_index'] = 0
            flash(f'Welcome, {annotator_name}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Please enter your name', 'error')
    
    return render_template('register.html')

@app.route('/annotate/<fold_name>')
def start_annotation(fold_name):
    """Start annotation for a specific fold"""
    if 'annotator_name' not in session:
        return redirect(url_for('register'))
    
    if fold_name not in data_manager.samples:
        flash('Fold not found', 'error')
        return redirect(url_for('index'))
    
    session['current_fold'] = fold_name
    session['current_sample_index'] = 0
    
    return redirect(url_for('annotate_sample'))

@app.route('/annotate')
def annotate_sample():
    """Show current sample for annotation"""
    if 'annotator_name' not in session or 'current_fold' not in session:
        return redirect(url_for('register'))
    
    fold_name = session['current_fold']
    annotator_name = session['annotator_name']
    
    # Find next unannotated sample
    sample_index = session.get('current_sample_index', 0)
    total_samples = len(data_manager.samples[fold_name])
    
    # Skip to next unannotated sample
    while sample_index < total_samples:
        sample = data_manager.get_sample(fold_name, sample_index)
        if sample:
            # Check if already annotated by this annotator
            already_annotated = False
            if 'human_annotations' in sample:
                already_annotated = annotator_name in sample['human_annotations']
            
            if not already_annotated:
                # Found an unannotated sample
                session['current_sample_index'] = sample_index
                break
        sample_index += 1
    
    # If we've gone through all samples
    if sample_index >= total_samples:
        flash('No more unannotated samples in this fold!', 'info')
        return redirect(url_for('index'))
    
    # Format conversation trace for display
    conversation = format_conversation_trace(sample.get('messages', []))
    
    # Count how many samples this user has annotated in this fold
    annotated_count = 0
    for i in range(total_samples):
        s = data_manager.get_sample(fold_name, i)
        if s and 'human_annotations' in s and annotator_name in s['human_annotations']:
            annotated_count += 1
    
    # Use the current sample index for progress display
    current_sample_index = session.get('current_sample_index', 0)
    progress = f"{current_sample_index + 1} / {total_samples} (You've annotated {annotated_count} samples)"
    
    return render_template('annotate.html', 
                         sample=sample,
                         conversation=conversation,
                         fold_name=fold_name,
                         sample_index=current_sample_index,
                         progress=progress,
                         already_annotated=False)  # This will always be False now since we skip annotated ones

@app.route('/submit_annotation', methods=['POST'])
def submit_annotation():
    """Submit annotation for current sample"""
    if 'annotator_name' not in session or 'current_fold' not in session:
        return redirect(url_for('register'))
    
    fold_name = session['current_fold']
    sample_index = session['current_sample_index']
    classification = request.form.get('classification')
    
    if not classification or classification not in ['lie', 'not_lie']:
        flash('Please select a classification', 'error')
        return redirect(url_for('annotate_sample'))
    
    # Save annotation
    success = data_manager.save_annotation(
        fold_name, sample_index, session['annotator_name'], classification
    )
    
    if success:
        flash('Annotation saved successfully!', 'success')
    else:
        flash('Failed to save annotation', 'error')
    
    # Move to next sample
    session['current_sample_index'] += 1
    
    return redirect(url_for('annotate_sample'))

@app.route('/skip_sample', methods=['POST'])
def skip_sample():
    """Skip current sample"""
    if 'current_sample_index' in session:
        session['current_sample_index'] += 1
    
    return redirect(url_for('annotate_sample'))

@app.route('/previous_sample', methods=['POST'])
def previous_sample():
    """Go to previous sample"""
    if 'current_sample_index' in session and session['current_sample_index'] > 0:
        session['current_sample_index'] -= 1
    
    return redirect(url_for('annotate_sample'))

@app.route('/logout')
def logout():
    """Logout annotator"""
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('register'))

def format_conversation_trace(messages):
    """Format conversation messages for display"""
    formatted = []
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted.append({
            'role': role.capitalize(),
            'content': content
        })
    return formatted

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
