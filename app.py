import os
import torch
import torch.nn as nn
import numpy as np
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import base64

# Import model processors
from model1_processor import Model1Processor
from model2_processor import Model2Processor
from model3_processor import Model3Processor

app = Flask(__name__, static_folder='static')

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)
# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)
# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Define different model architectures for each model type
class Model1Net(nn.Module):
    def __init__(self):
        super(Model1Net, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(7, 128),  # Model 1 has 7 features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.age_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU()
        )
        self.gender_head = nn.Linear(32, 2)
        self.handedness_head = nn.Linear(32, 2)
        self.class_head = nn.Linear(32, 2)

    def forward(self, x):
        x = self.shared(x)
        age = self.age_head(x)
        gender = self.gender_head(x)
        handedness = self.handedness_head(x)
        class_output = self.class_head(x)
        return age, gender, handedness, class_output

class Model2Net(nn.Module):
    def __init__(self):
        super(Model2Net, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(20, 128),  # Model 2 has 20 histogram features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.age_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU()
        )
        self.gender_head = nn.Linear(32, 2)
        self.handedness_head = nn.Linear(32, 2)
        self.class_head = nn.Linear(32, 2)

    def forward(self, x):
        x = self.shared(x)
        age = self.age_head(x)
        gender = self.gender_head(x)
        handedness = self.handedness_head(x)
        class_output = self.class_head(x)
        return age, gender, handedness, class_output

class Model3Net(nn.Module):
    def __init__(self):
        super(Model3Net, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(20, 128),  # Model 3 has 20 combined features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.age_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU()
        )
        self.gender_head = nn.Linear(32, 2)
        self.handedness_head = nn.Linear(32, 2)
        self.class_head = nn.Linear(32, 2)

    def forward(self, x):
        x = self.shared(x)
        age = self.age_head(x)
        gender = self.gender_head(x)
        handedness = self.handedness_head(x)
        class_output = self.class_head(x)
        return age, gender, handedness, class_output

# Helper functions for feature extraction
def calculate_histogram_features(timing_data, num_bins=20):
    """
    Calculate histogram features from keystroke timing data
    """
    if len(timing_data) < 5:  # Need minimum keystrokes for meaningful analysis
        return None
    
    # Values are already in milliseconds, no need to convert
    timing_ms = timing_data
    
    # Create histogram with adaptive bin edges based on data range
    min_time = min(timing_ms)
    max_time = max(timing_ms)
    
    # Ensure we have a reasonable range, even with limited data
    if max_time - min_time < 20:
        max_time = min_time + 20  # Minimum range of 20ms
    
    # Create bins and calculate histogram
    bins = np.linspace(min_time, max_time, num_bins + 1)
    hist, _ = np.histogram(timing_ms, bins=bins, density=True)
    
    # Normalize the histogram to sum to 1
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def calculate_basic_features(timing_data):
    """
    Calculate basic statistical features from keystroke timing data
    """
    if len(timing_data) < 5:  # Need minimum keystrokes
        return None
    
    # Values are already in milliseconds, no need to convert
    timing_ms = timing_data
    
    # Calculate basic statistics
    mean = np.mean(timing_ms)
    median = np.median(timing_ms)
    std_dev = np.std(timing_ms)
    min_val = np.min(timing_ms)
    max_val = np.max(timing_ms)
    
    # Calculate quartiles
    q1 = np.percentile(timing_ms, 25)
    q3 = np.percentile(timing_ms, 75)
    iqr = q3 - q1
    
    # Advanced features
    coef_var = std_dev / mean if mean > 0 else 0
    
    # Combine all features
    features = [
        mean, median, std_dev, min_val, max_val,
        q1, q3, iqr, coef_var,
        # Padding to reach 20 features as expected by the model
        mean/2, std_dev/2, coef_var*2, 
        np.log(mean) if mean > 0 else 0,
        np.log(std_dev) if std_dev > 0 else 0,
        max_val/mean if mean > 0 else 0,
        min_val/mean if mean > 0 else 0,
        q1/q3 if q3 > 0 else 0,
        mean/max_val if max_val > 0 else 0,
        std_dev/max_val if max_val > 0 else 0,
        iqr/std_dev if std_dev > 0 else 0
    ]
    
    return features

def calculate_advanced_features(timing_data):
    """
    Calculate advanced features combining histogram and statistics
    """
    if len(timing_data) < 5:
        return None
    
    # Get histogram features (10 bins)
    hist_features = calculate_histogram_features(timing_data, num_bins=10)
    
    # Get a subset of basic features (10 features)
    basic_features = calculate_basic_features(timing_data)[:10]
    
    # Combine both feature sets
    if hist_features is not None and basic_features is not None:
        combined_features = np.concatenate([hist_features, basic_features])
        return combined_features
    
    return None

# Initialize model processors with correct model classes
model1 = Model1Processor('model_weights.pth')
model2 = Model2Processor('model1_weights.pth')
model3 = Model3Processor('model3_weights.pth')

# Load models at startup
def load_models():
    model1.load_model(Model1Net)
    model2.load_model(Model2Net)
    model3.load_model(Model3Net)
    print("Models loaded successfully")

# Load models when app starts
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contribute')
def contribute():
    return render_template('contribute.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    keystroke_timings = data.get('keystrokes', [])
    model_id = data.get('model', 'model2')  # Default to histogram model
    
    if len(keystroke_timings) < 10:
        return jsonify({'error': 'Not enough keystroke data. Please type more.'})
    
    try:
        # Select the appropriate model processor based on model_id
        processor = None
        if model_id == 'model1':
            processor = model1
        elif model_id == 'model2':
            processor = model2
        elif model_id == 'model3':
            processor = model3
        else:
            return jsonify({'error': 'Invalid model selection'})
        
        # Extract features
        features = processor.extract_features(keystroke_timings)
        if features is None:
            return jsonify({'error': 'Failed to extract features from keystroke data'})
        
        # Normalize features
        normalized_features = processor.normalize_features(features)
        
        # Make prediction
        result = processor.predict(normalized_features)
        if result is None:
            return jsonify({'error': 'Failed to make prediction'})
            
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/api/contribute', methods=['POST'])
def contribute_data():
    data = request.json
    
    if not data or not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'Invalid data format'})
    
    try:
        # Validate keystrokes data
        if 'keystrokes' not in data or not isinstance(data['keystrokes'], list) or len(data['keystrokes']) < 5:
            return jsonify({'success': False, 'error': 'Invalid keystroke data'})
            
        # Add timestamp and ID
        data['timestamp'] = datetime.now().isoformat()
        data['id'] = str(uuid.uuid4())
        
        # Ensure we capture the raw keystroke data
        # Add keystroke statistics for easier analysis
        keystroke_stats = {
            'count': len(data['keystrokes']),
            'avg_time': sum(data['keystrokes']) / len(data['keystrokes']),  # already in ms
            'min_time': min(data['keystrokes']),  # already in ms
            'max_time': max(data['keystrokes']),  # already in ms
        }
        data['keystroke_stats'] = keystroke_stats
        
        # Save data to JSON file
        file_path = os.path.join('data', f"{data['id']}.json")
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved contribution data to {file_path} with {len(data['keystrokes'])} keystrokes")
        return jsonify({'success': True, 'id': data['id']})
    
    except Exception as e:
        print(f"Error saving contribution: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to save data: {str(e)}'})

@app.route('/api/admin/data', methods=['GET'])
def get_admin_data():
    try:
        all_data = []
        data_dir = 'data'
        
        # Load all JSON files from the data directory
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                        all_data.append(file_data)
                except Exception as e:
                    print(f"Error reading file {filename}: {str(e)}")
        
        # Sort by timestamp (newest first)
        all_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify(all_data)
    
    except Exception as e:
        print(f"Error retrieving admin data: {str(e)}")
        return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 