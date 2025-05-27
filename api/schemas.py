"""
Professional API Schemas
Comprehensive data validation and serialization schemas for the keystroke analytics API.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from marshmallow import Schema, fields, validate, validates, ValidationError, post_load
from pydantic import BaseModel, Field, validator, ConfigDict
import re


# Pydantic Models for Request/Response validation
class KeystrokeTimingData(BaseModel):
    """Model for keystroke timing data validation"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    timing_data: List[float] = Field(
        ...,
        min_items=5,
        max_items=1000,
        description="List of keystroke timings in microseconds"
    )
    text_typed: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="The text that was typed"
    )
    session_id: str = Field(
        ...,
        min_length=32,
        max_length=40,
        description="Unique session identifier"
    )
    
    @validator('timing_data')
    def validate_timing_values(cls, v):
        """Validate timing values are reasonable"""
        if not v:
            raise ValueError("Timing data cannot be empty")
        
        # Check for reasonable timing values (1ms to 5s in microseconds)
        invalid_timings = [t for t in v if not (1000 <= t <= 5000000)]
        
        if len(invalid_timings) > len(v) * 0.2:  # Allow up to 20% outliers
            raise ValueError("Too many unreasonable timing values detected")
        
        return v
    
    @validator('session_id')
    def validate_session_id(cls, v):
        """Validate session ID format"""
        if not re.match(r'^[a-zA-Z0-9-_]{32,40}$', v):
            raise ValueError("Invalid session ID format")
        return v


class PredictionRequest(BaseModel):
    """Model for prediction requests"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    keystroke_data: KeystrokeTimingData
    model_name: str = Field(
        ...,
        regex=r'^(model1|model2|model3)$',
        description="Name of the model to use for prediction"
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores in response"
    )
    user_agent: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Browser user agent string"
    )


class ConfidenceScores(BaseModel):
    """Model for prediction confidence scores"""
    gender: float = Field(..., ge=0.0, le=1.0)
    handedness: float = Field(..., ge=0.0, le=1.0)
    class_prediction: float = Field(..., ge=0.0, le=1.0, alias='class')


class PredictionResponse(BaseModel):
    """Model for prediction responses"""
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True
    )
    
    prediction_id: str = Field(..., description="Unique prediction identifier")
    age: int = Field(..., ge=18, le=100, description="Predicted age")
    gender: str = Field(..., regex=r'^(Male|Female)$', description="Predicted gender")
    handedness: str = Field(..., regex=r'^(Left-handed|Right-handed)$', description="Predicted handedness")
    user_class: str = Field(..., regex=r'^(Professional|Casual)$', description="Predicted user class", alias='class')
    confidence_scores: Optional[ConfidenceScores] = Field(default=None, description="Prediction confidence scores")
    prediction_time_ms: float = Field(..., ge=0, description="Time taken for prediction in milliseconds")
    model_used: str = Field(..., description="Model used for prediction")
    data_quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score of input data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


class UserContributionData(BaseModel):
    """Model for user contribution data"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    age: Optional[int] = Field(default=None, ge=18, le=100, description="User's actual age")
    gender: Optional[str] = Field(default=None, regex=r'^(Male|Female)$', description="User's actual gender")
    handedness: Optional[str] = Field(default=None, regex=r'^(Left-handed|Right-handed)$', description="User's actual handedness")
    user_class: Optional[str] = Field(default=None, regex=r'^(Professional|Casual)$', description="User's actual class")
    profession: Optional[str] = Field(default=None, max_length=100, description="User's profession")
    typing_experience: Optional[str] = Field(
        default=None, 
        regex=r'^(Beginner|Intermediate|Expert)$', 
        description="User's typing experience level"
    )
    keyboard_type: Optional[str] = Field(default=None, max_length=50, description="Type of keyboard used")
    feedback: Optional[str] = Field(default=None, max_length=1000, description="User feedback on predictions")
    app_rating: Optional[int] = Field(default=None, ge=1, le=5, description="App rating (1-5 stars)")
    data_consent: bool = Field(default=False, description="Consent for data usage")
    research_consent: bool = Field(default=False, description="Consent for research usage")


class ContributionRequest(BaseModel):
    """Model for contribution requests"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    keystroke_data: KeystrokeTimingData
    user_data: UserContributionData
    prediction_id: Optional[str] = Field(default=None, description="Associated prediction ID")


class ContributionResponse(BaseModel):
    """Model for contribution responses"""
    contribution_id: str = Field(..., description="Unique contribution identifier")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Contribution timestamp")


class ModelPerformanceStats(BaseModel):
    """Model for model performance statistics"""
    total_predictions: int = Field(..., ge=0)
    average_prediction_time_ms: float = Field(..., ge=0)
    cache_hit_ratio: float = Field(..., ge=0.0, le=1.0)
    last_prediction: Optional[str] = Field(default=None)
    cache_size: int = Field(..., ge=0)


class ModelInfo(BaseModel):
    """Model for model information"""
    name: str = Field(...)
    version: str = Field(...)
    total_parameters: int = Field(..., ge=0)
    trainable_parameters: int = Field(..., ge=0)
    device: str = Field(...)


class ModelStatsResponse(BaseModel):
    """Model for model statistics response"""
    model_name: str = Field(...)
    performance_stats: ModelPerformanceStats
    model_info: ModelInfo
    accuracy_stats: Optional[Dict[str, float]] = Field(default=None)


class HealthCheckResponse(BaseModel):
    """Model for health check response"""
    status: str = Field(..., regex=r'^(healthy|unhealthy|degraded)$')
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(...)
    environment: str = Field(...)
    uptime_seconds: float = Field(..., ge=0)
    database_status: str = Field(...)
    models_loaded: List[str] = Field(...)
    memory_usage_mb: float = Field(..., ge=0)
    cpu_usage_percent: float = Field(..., ge=0.0, le=100.0)


class AdminStatsResponse(BaseModel):
    """Model for admin statistics response"""
    database_stats: Dict[str, Any] = Field(...)
    model_stats: Dict[str, ModelStatsResponse] = Field(...)
    system_stats: Dict[str, Any] = Field(...)
    recent_activity: List[Dict[str, Any]] = Field(...)


# Marshmallow Schemas for Flask-RESTx integration
class KeystrokeDataSchema(Schema):
    """Marshmallow schema for keystroke data"""
    timing_data = fields.List(
        fields.Float(validate=validate.Range(min=1000, max=5000000)),
        required=True,
        validate=validate.Length(min=5, max=1000),
        metadata={'description': 'List of keystroke timings in microseconds'}
    )
    text_typed = fields.Str(
        required=True,
        validate=validate.Length(min=5, max=5000),
        metadata={'description': 'The text that was typed'}
    )
    session_id = fields.Str(
        required=True,
        validate=validate.Length(min=32, max=40),
        metadata={'description': 'Unique session identifier'}
    )
    
    @validates('timing_data')
    def validate_timing_data(self, value):
        """Validate timing data quality"""
        if not value:
            raise ValidationError("Timing data cannot be empty")
        
        # Check for reasonable timing distribution
        invalid_count = sum(1 for t in value if not (1000 <= t <= 5000000))
        if invalid_count > len(value) * 0.2:
            raise ValidationError("Too many unreasonable timing values")


class PredictionRequestSchema(Schema):
    """Marshmallow schema for prediction requests"""
    keystroke_data = fields.Nested(KeystrokeDataSchema, required=True)
    model_name = fields.Str(
        required=True,
        validate=validate.OneOf(['model1', 'model2', 'model3']),
        metadata={'description': 'Name of the model to use'}
    )
    include_confidence = fields.Bool(
        missing=True,
        metadata={'description': 'Whether to include confidence scores'}
    )
    user_agent = fields.Str(
        missing=None,
        validate=validate.Length(max=500),
        metadata={'description': 'Browser user agent string'}
    )


class PredictionResponseSchema(Schema):
    """Marshmallow schema for prediction responses"""
    prediction_id = fields.Str(required=True)
    age = fields.Int(required=True, validate=validate.Range(min=18, max=100))
    gender = fields.Str(required=True, validate=validate.OneOf(['Male', 'Female']))
    handedness = fields.Str(required=True, validate=validate.OneOf(['Left-handed', 'Right-handed']))
    user_class = fields.Str(required=True, validate=validate.OneOf(['Professional', 'Casual']))
    confidence_scores = fields.Dict(missing=None)
    prediction_time_ms = fields.Float(required=True, validate=validate.Range(min=0))
    model_used = fields.Str(required=True)
    data_quality_score = fields.Float(required=True, validate=validate.Range(min=0, max=1))
    timestamp = fields.DateTime(required=True)


class ContributionRequestSchema(Schema):
    """Marshmallow schema for contribution requests"""
    keystroke_data = fields.Nested(KeystrokeDataSchema, required=True)
    user_data = fields.Dict(required=True)
    prediction_id = fields.Str(missing=None)
    
    @validates('user_data')
    def validate_user_data(self, value):
        """Validate user contribution data"""
        if not isinstance(value, dict):
            raise ValidationError("User data must be a dictionary")
        
        # Validate age if provided
        age = value.get('age')
        if age is not None:
            if not isinstance(age, int) or not (18 <= age <= 100):
                raise ValidationError("Age must be between 18 and 100")
        
        # Validate gender if provided
        gender = value.get('gender')
        if gender is not None and gender not in ['Male', 'Female']:
            raise ValidationError("Gender must be 'Male' or 'Female'")
        
        # Validate rating if provided
        rating = value.get('app_rating')
        if rating is not None:
            if not isinstance(rating, int) or not (1 <= rating <= 5):
                raise ValidationError("Rating must be between 1 and 5")


class AdminStatsSchema(Schema):
    """Marshmallow schema for admin statistics"""
    database_stats = fields.Dict()
    model_stats = fields.Dict()
    system_stats = fields.Dict()
    recent_activity = fields.List(fields.Dict())


# Schema validation utilities
class SchemaValidator:
    """Utility class for schema validation"""
    
    @staticmethod
    def validate_with_pydantic(model_class: type, data: dict) -> tuple[bool, Union[BaseModel, dict]]:
        """
        Validate data using Pydantic model
        
        Args:
            model_class: Pydantic model class
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, validated_data_or_errors)
        """
        try:
            validated = model_class(**data)
            return True, validated
        except Exception as e:
            return False, {'error': str(e)}
    
    @staticmethod
    def validate_with_marshmallow(schema: Schema, data: dict) -> tuple[bool, Union[dict, dict]]:
        """
        Validate data using Marshmallow schema
        
        Args:
            schema: Marshmallow schema instance
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, validated_data_or_errors)
        """
        try:
            validated = schema.load(data)
            return True, validated
        except ValidationError as e:
            return False, e.messages
    
    @staticmethod
    def serialize_with_marshmallow(schema: Schema, data: Union[dict, BaseModel]) -> dict:
        """
        Serialize data using Marshmallow schema
        
        Args:
            schema: Marshmallow schema instance
            data: Data to serialize
            
        Returns:
            Serialized data
        """
        if isinstance(data, BaseModel):
            data = data.model_dump()
        
        return schema.dump(data) 