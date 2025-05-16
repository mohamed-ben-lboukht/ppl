import numpy as np
import torch

class Model2Processor:
    """Process keystroke data using histogram-based features (Model 2)"""
    
    def __init__(self, model_path='model1_weights.pth'):
        self.model_path = model_path
        self.model = None
        self.scaler = RobustScaler()
        self.num_bins = 20
        self.min_range = 0
        self.max_range = 1000  # milliseconds
    
    def load_model(self, model_class):
        """Load the PyTorch model"""
        try:
            self.model = model_class()
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading Model 2: {e}")
            return False
    
    def extract_features(self, keystroke_timings):
        """Extract histogram features from keystroke timings"""
        if len(keystroke_timings) < 5:
            return None
        
        # Convert to milliseconds
        timing_ms = [t / 1000 for t in keystroke_timings]
        
        try:
            # Compute histogram features
            hist_features = self.compute_histogram_features(timing_ms)
            return hist_features
        except Exception as e:
            print(f"Error extracting histogram features: {e}")
            return None
    
    def compute_histogram_features(self, timings, bins=None, min_range=None, max_range=None):
        """Compute histogram features exactly as in the notebook"""
        if bins is None:
            bins = self.num_bins
        if min_range is None:
            min_range = self.min_range
        if max_range is None:
            max_range = self.max_range
            
        if np.any(np.isnan(timings)) or np.any(np.isinf(timings)) or np.max(timings) == np.min(timings):
            return np.zeros(bins)
            
        hist, _ = np.histogram(timings, bins=bins, range=(min_range, max_range))
        hist_sum = hist.sum()
        
        if hist_sum == 0:
            return np.zeros(bins)
            
        return hist / hist_sum
    
    def normalize_features(self, features):
        """Normalize features using RobustScaler"""
        features_array = np.array(features).reshape(1, -1)
        
        # If we haven't fit the scaler yet, initialize with some default values
        if self.scaler.means_ is None:
            # These would be derived from your training data
            self.scaler.means_ = np.mean(features_array, axis=0)
            self.scaler.stds_ = np.std(features_array, axis=0) 
            self.scaler.stds_[self.scaler.stds_ == 0] = 1.0
        
        return self.scaler.transform(features_array).flatten()
    
    def predict(self, features):
        """Make prediction using the loaded model"""
        if self.model is None:
            return None
            
        # Convert features to torch tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            age_pred, gender_pred, handedness_pred, class_pred = self.model(input_tensor)
        
        # Process predictions
        age = int(age_pred.item() * 50 + 25)  # Adjust scaling based on your model
        gender = "Male" if torch.argmax(gender_pred, dim=1).item() == 1 else "Female"
        handedness = "Right-handed" if torch.argmax(handedness_pred, dim=1).item() == 1 else "Left-handed"
        user_class = "Class A" if torch.argmax(class_pred, dim=1).item() == 1 else "Class B"
        
        return {
            'age': age,
            'gender': gender,
            'handedness': handedness,
            'class': user_class
        }


class RobustScaler:
    """Scaling class that matches the one used in training"""
    def __init__(self):
        self.means_ = None
        self.stds_ = None
    
    def fit(self, X):
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        self.stds_[self.stds_ == 0] = 1e-6
        return self
        
    def transform(self, X):
        if self.means_ is None or self.stds_ is None:
            return X
        return (X - self.means_) / self.stds_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        return X * self.stds_ + self.means_ 