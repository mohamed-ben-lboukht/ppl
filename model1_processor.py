import numpy as np
import torch
from scipy.stats import skew

class Model1Processor:
    """Process keystroke data using basic statistical features (Model 1)"""
    
    def __init__(self, model_path='model_weights.pth'):
        self.model_path = model_path
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = ['mean', 'std', 'min', 'max', 'total', 'median', 'skew']
    
    def load_model(self, model_class):
        """Load the PyTorch model"""
        try:
            self.model = model_class()
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading Model 1: {e}")
            return False
    
    def extract_features(self, keystroke_timings):
        """Extract basic statistical features from keystroke timings"""
        if len(keystroke_timings) < 5:
            return None
        
        # Convert to milliseconds
        timing_ms = [t / 1000 for t in keystroke_timings]
        
        # Calculate the statistical features
        try:
            features = [
                np.mean(timing_ms),
                np.std(timing_ms) if np.std(timing_ms) != 0 else 1e-6,
                np.min(timing_ms),
                np.max(timing_ms),
                np.sum(timing_ms),
                np.median(timing_ms),
                skew(timing_ms)
            ]
            
            # Make sure we have all 7 features
            if len(features) < 7:
                features.extend([0] * (7 - len(features)))
                
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def normalize_features(self, features):
        """Normalize features using RobustScaler"""
        # For initial prediction, use a simple standardization
        # In production, you would fit this on your training data
        features_array = np.array(features).reshape(1, -1)
        
        # If we haven't fit the scaler yet, initialize with some default values
        if self.scaler.means_ is None:
            # These would ideally be derived from your training data
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
        
        # Get predicted values
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