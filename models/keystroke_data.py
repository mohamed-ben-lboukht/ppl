"""
Professional Keystroke Data Models
Database models for storing keystroke timing data, predictions, and user contributions.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Index, text
from .database import db, BaseModel

logger = logging.getLogger(__name__)


class KeystrokeData(BaseModel):
    """
    Model for storing individual keystroke timing data
    """
    __tablename__ = 'keystroke_data'
    
    # Core data
    session_id = db.Column(db.String(36), nullable=False, index=True)
    user_agent = db.Column(db.Text)
    ip_address = db.Column(db.String(45))  # IPv6 compatible
    
    # Keystroke timing data (stored as JSON for flexibility)
    timing_data = db.Column(db.JSON, nullable=False)
    text_typed = db.Column(db.Text)
    keystroke_count = db.Column(db.Integer, nullable=False, default=0)
    
    # Metadata
    model_used = db.Column(db.String(20))
    processing_time_ms = db.Column(db.Float)
    data_quality_score = db.Column(db.Float)  # 0-1 quality rating
    prediction_id = db.Column(db.String(36), nullable=True)  # Link to prediction
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_keystroke_session_model', 'session_id', 'model_used'),
        Index('idx_keystroke_created_quality', 'created_at', 'data_quality_score'),
        Index('idx_keystroke_count_time', 'keystroke_count', 'created_at'),
    )
    
    def __init__(self, timing_data: List[float] = None, text_typed: str = None, session_id: str = None, 
                 model_used: str = None, user_agent: str = None, ip_address: str = None, prediction_id: str = None):
        """
        Initialize keystroke data
        
        Args:
            timing_data: List of keystroke timings in microseconds
            text_typed: The text that was typed
            session_id: Unique session identifier
            model_used: Which model was used for prediction
            user_agent: Browser user agent string
            ip_address: Client IP address
        """
        self.timing_data = timing_data or []
        self.text_typed = text_typed
        self.session_id = session_id
        self.model_used = model_used
        self.user_agent = user_agent
        self.ip_address = ip_address
        self.keystroke_count = len(timing_data) if timing_data else 0
        self.data_quality_score = self._calculate_quality_score()
        # Note: prediction_id will be set by foreign key relationship
    
    def _calculate_quality_score(self) -> float:
        """
        Calculate a quality score for the keystroke data
        
        Returns:
            Quality score between 0 and 1
        """
        if not self.timing_data or len(self.timing_data) < 5:
            return 0.0
        
        score = 1.0
        
        # Penalize very short sequences
        if self.keystroke_count < 10:
            score *= 0.7
        elif self.keystroke_count < 20:
            score *= 0.9
        
        # Check for reasonable timing values (10ms to 2000ms)
        reasonable_timings = [
            t for t in self.timing_data 
            if 10000 <= t <= 2000000  # 10ms to 2s in microseconds
        ]
        
        if len(reasonable_timings) < len(self.timing_data) * 0.8:
            score *= 0.5  # Many unreasonable timings
        
        # Check for variance (too uniform is suspicious)
        try:
            import numpy as np
            std_dev = np.std(self.timing_data)
            mean_val = np.mean(self.timing_data)
            cv = std_dev / mean_val if mean_val > 0 else 0
            
            if cv < 0.1:  # Too uniform
                score *= 0.6
            elif cv > 2.0:  # Too variable
                score *= 0.8
        except Exception:
            pass
        
        return max(0.0, min(1.0, score))
    
    @hybrid_property
    def timing_statistics(self) -> Dict[str, float]:
        """Get basic statistics about the timing data"""
        if not self.timing_data:
            return {}
        
        try:
            import numpy as np
            data = np.array(self.timing_data)
            
            return {
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75))
            }
        except Exception as e:
            logger.error(f"Error calculating timing statistics: {e}")
            return {}
    
    @classmethod
    def get_by_session(cls, session_id: str) -> List['KeystrokeData']:
        """Get all keystroke data for a session"""
        return cls.query.filter_by(session_id=session_id, is_active=True).all()
    
    @classmethod
    def get_recent_data(cls, hours: int = 24) -> List['KeystrokeData']:
        """Get recent keystroke data"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return cls.query.filter(
            cls.created_at >= cutoff,
            cls.is_active == True
        ).order_by(cls.created_at.desc()).all()
    
    @classmethod
    def get_quality_data(cls, min_quality: float = 0.7) -> List['KeystrokeData']:
        """Get high-quality keystroke data"""
        return cls.query.filter(
            cls.data_quality_score >= min_quality,
            cls.is_active == True
        ).all()
    
    def to_dict(self, include_timing_data: bool = False) -> Dict[str, Any]:
        """Convert to dictionary with optional timing data"""
        base_dict = super().to_dict(exclude_fields=['timing_data'] if not include_timing_data else [])
        
        if include_timing_data:
            base_dict['timing_statistics'] = self.timing_statistics
        
        return base_dict


class Prediction(BaseModel):
    """
    Model for storing ML model predictions
    """
    __tablename__ = 'predictions'
    
    # Override the id to be string
    id = db.Column(db.String(36), primary_key=True)
    
    # Foreign key and session info
    session_id = db.Column(db.String(36), nullable=False, index=True)
    model_name = db.Column(db.String(20), nullable=False)
    
    # Prediction results
    predicted_age = db.Column(db.Integer)
    predicted_gender = db.Column(db.String(10))
    predicted_handedness = db.Column(db.String(15))
    predicted_class = db.Column(db.String(10))
    
    # Store confidence scores as JSON
    confidence_scores = db.Column(db.JSON)
    data_quality_score = db.Column(db.Float)
    user_agent = db.Column(db.Text)
    
    # Confidence scores (0-1)
    age_confidence = db.Column(db.Float)
    gender_confidence = db.Column(db.Float)
    handedness_confidence = db.Column(db.Float)
    class_confidence = db.Column(db.Float)
    
    # Model information
    model_version = db.Column(db.String(20))
    prediction_time_ms = db.Column(db.Float)
    
    # Validation data (if available)
    actual_age = db.Column(db.Integer)
    actual_gender = db.Column(db.String(10))
    actual_handedness = db.Column(db.String(15))
    actual_class = db.Column(db.String(10))
    
    # Accuracy flags
    age_accurate = db.Column(db.Boolean)
    gender_accurate = db.Column(db.Boolean)
    handedness_accurate = db.Column(db.Boolean)
    class_accurate = db.Column(db.Boolean)
    
    def __init__(self, id: str = None, session_id: str = None, model_name: str = None, 
                 age: int = None, gender: str = None, handedness: str = None, 
                 user_class: str = None, confidence_scores: Dict = None, 
                 data_quality_score: float = None, user_agent: str = None, **kwargs):
        """
        Initialize prediction
        """
        self.id = id
        self.session_id = session_id
        self.model_name = model_name
        self.predicted_age = age
        self.predicted_gender = gender
        self.predicted_handedness = handedness
        self.predicted_class = user_class
        self.confidence_scores = confidence_scores or {}
        self.data_quality_score = data_quality_score
        self.user_agent = user_agent
    
    def validate_prediction(self, actual_data: Dict[str, Any]) -> None:
        """
        Validate predictions against actual data
        
        Args:
            actual_data: Dictionary with actual user data
        """
        self.actual_age = actual_data.get('age')
        self.actual_gender = actual_data.get('gender')
        self.actual_handedness = actual_data.get('handedness')
        self.actual_class = actual_data.get('class')
        
        # Calculate accuracy
        self.age_accurate = (
            abs(self.predicted_age - self.actual_age) <= 5 
            if self.predicted_age and self.actual_age else None
        )
        self.gender_accurate = (
            self.predicted_gender == self.actual_gender
            if self.predicted_gender and self.actual_gender else None
        )
        self.handedness_accurate = (
            self.predicted_handedness == self.actual_handedness
            if self.predicted_handedness and self.actual_handedness else None
        )
        self.class_accurate = (
            self.predicted_class == self.actual_class
            if self.predicted_class and self.actual_class else None
        )
    
    @classmethod
    def get_accuracy_stats(cls, model_version: str = None) -> Dict[str, float]:
        """Get accuracy statistics for predictions"""
        query = cls.query.filter(cls.is_active == True)
        
        if model_version:
            query = query.filter(cls.model_version == model_version)
        
        predictions = query.all()
        
        if not predictions:
            return {}
        
        stats = {}
        
        # Calculate accuracy for each attribute
        for attr in ['age', 'gender', 'handedness', 'class']:
            accurate_field = f"{attr}_accurate"
            accurate_predictions = [
                p for p in predictions 
                if getattr(p, accurate_field) is not None
            ]
            
            if accurate_predictions:
                accuracy = sum(
                    1 for p in accurate_predictions 
                    if getattr(p, accurate_field)
                ) / len(accurate_predictions)
                stats[f"{attr}_accuracy"] = accuracy
        
        return stats


class UserContribution(BaseModel):
    """
    Model for storing user contributions and feedback
    """
    __tablename__ = 'user_contributions'
    
    # Override the id to be string
    id = db.Column(db.String(36), primary_key=True)
    
    # Session and prediction info
    session_id = db.Column(db.String(36), nullable=False, index=True)
    prediction_id = db.Column(db.String(36), nullable=True)
    
    # Store the keystroke data directly
    timing_data = db.Column(db.JSON, nullable=False)
    text_typed = db.Column(db.Text)
    
    # User provided data
    user_age = db.Column(db.Integer)
    user_gender = db.Column(db.String(10))
    user_handedness = db.Column(db.String(15))
    user_class = db.Column(db.String(10))
    
    # Additional information
    user_profession = db.Column(db.String(50))
    typing_experience = db.Column(db.String(20))  # beginner, intermediate, expert
    keyboard_type = db.Column(db.String(30))
    
    # Feedback
    prediction_feedback = db.Column(db.Text)
    app_rating = db.Column(db.Integer)  # 1-5 stars
    
    # Privacy
    data_usage_consent = db.Column(db.Boolean, nullable=False, default=False)
    research_consent = db.Column(db.Boolean, nullable=False, default=False)
    
    def __init__(self, id: str = None, session_id: str = None, timing_data: List = None, 
                 text_typed: str = None, user_data: Dict[str, Any] = None, 
                 prediction_id: str = None, **kwargs):
        """
        Initialize user contribution
        """
        self.id = id
        self.session_id = session_id
        self.timing_data = timing_data or []
        self.text_typed = text_typed
        self.prediction_id = prediction_id
        
        if user_data:
            self.user_age = user_data.get('age')
            self.user_gender = user_data.get('gender')
            self.user_handedness = user_data.get('handedness')
            self.user_class = user_data.get('class')
            self.user_profession = user_data.get('profession')
            self.typing_experience = user_data.get('typing_experience')
            self.keyboard_type = user_data.get('keyboard_type')
            self.prediction_feedback = user_data.get('feedback')
            self.app_rating = user_data.get('rating')
            self.data_usage_consent = user_data.get('data_consent', False)
            self.research_consent = user_data.get('research_consent', False)
    
    @classmethod
    def get_demographics(cls) -> Dict[str, Any]:
        """Get demographic statistics from contributions"""
        contributions = cls.query.filter(cls.is_active == True).all()
        
        if not contributions:
            return {}
        
        demographics = {
            'total_contributions': len(contributions),
            'age_distribution': {},
            'gender_distribution': {},
            'handedness_distribution': {},
            'experience_distribution': {},
            'average_rating': 0
        }
        
        # Age distribution
        ages = [c.user_age for c in contributions if c.user_age]
        if ages:
            demographics['age_distribution'] = {
                '18-25': len([a for a in ages if 18 <= a <= 25]),
                '26-35': len([a for a in ages if 26 <= a <= 35]),
                '36-45': len([a for a in ages if 36 <= a <= 45]),
                '46-55': len([a for a in ages if 46 <= a <= 55]),
                '56+': len([a for a in ages if a > 55])
            }
        
        # Gender distribution
        genders = [c.user_gender for c in contributions if c.user_gender]
        for gender in set(genders):
            demographics['gender_distribution'][gender] = genders.count(gender)
        
        # Handedness distribution
        handedness = [c.user_handedness for c in contributions if c.user_handedness]
        for hand in set(handedness):
            demographics['handedness_distribution'][hand] = handedness.count(hand)
        
        # Experience distribution
        experiences = [c.typing_experience for c in contributions if c.typing_experience]
        for exp in set(experiences):
            demographics['experience_distribution'][exp] = experiences.count(exp)
        
        # Average rating
        ratings = [c.app_rating for c in contributions if c.app_rating]
        if ratings:
            demographics['average_rating'] = sum(ratings) / len(ratings)
        
        return demographics 