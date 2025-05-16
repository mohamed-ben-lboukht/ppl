import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Define the neural network model architecture
class MultiOutputNet(nn.Module):
    def __init__(self):
        super(MultiOutputNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(20, 128),
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
    
    # Convert to milliseconds for better numerical stability
    timing_ms = [t / 1000 for t in timing_data]
    
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
    
    # Convert to milliseconds
    timing_ms = [t / 1000 for t in timing_data]
    
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

# Load pre-trained models
models = {}

def load_models():
    model_paths = {
        'model1': 'model_weights.pth',
        'model2': 'model1_weights.pth',
        'model3': 'model3_weights.pth'
    }
    
    for model_id, model_path in model_paths.items():
        # Create model instance
        model = MultiOutputNet()
        
        # Check if model file exists
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                models[model_id] = model
                print(f"Model {model_id} loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model {model_id}: {e}")
                # Use untrained model as fallback
                models[model_id] = model
        else:
            print(f"Model file {model_path} not found. Using untrained model.")
            models[model_id] = model

# Load models when app starts
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    keystroke_timings = data.get('keystrokes', [])
    model_id = data.get('model', 'model2')  # Default to histogram model
    
    if len(keystroke_timings) < 10:
        return jsonify({'error': 'Not enough keystroke data. Please type more.'})
    
    try:
        # Select the appropriate feature extraction method based on model
        if model_id == 'model1':
            features = calculate_basic_features(keystroke_timings)
        elif model_id == 'model2':
            features = calculate_histogram_features(keystroke_timings)
        elif model_id == 'model3':
            features = calculate_advanced_features(keystroke_timings)
        else:
            return jsonify({'error': 'Invalid model selection'})
        
        if features is None:
            return jsonify({'error': 'Failed to extract features from keystroke data'})
        
        # Use pre-trained model to make prediction
        model = models.get(model_id)
        if model is None:
            return jsonify({'error': 'Selected model not available'})
        
        # Convert features to tensor for model input
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            age_pred, gender_pred, handedness_pred, class_pred = model(input_tensor)
        
        # Process predictions
        # Scale age prediction to a reasonable range (20-70)
        age = max(20, min(70, int(age_pred.item() * 50 + 20)))
        
        gender = "Male" if torch.argmax(gender_pred, dim=1).item() == 1 else "Female"
        handedness = "Right-handed" if torch.argmax(handedness_pred, dim=1).item() == 1 else "Left-handed"
        user_class = "Class A" if torch.argmax(class_pred, dim=1).item() == 1 else "Class B"
        
        return jsonify({
            'age': age,
            'gender': gender,
            'handedness': handedness,
            'class': user_class
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 