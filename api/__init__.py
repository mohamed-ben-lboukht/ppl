"""
Professional RESTful API Package
Provides comprehensive keystroke analytics API with documentation, validation, and monitoring.
"""

from .schemas import (
    KeystrokeDataSchema,
    PredictionRequestSchema,
    PredictionResponseSchema,
    ContributionRequestSchema,
    AdminStatsSchema
)

from .resources import (
    PredictionResource,
    ContributionResource,
    AdminResource,
    HealthCheckResource,
    ModelStatsResource
)

from .api_manager import APIManager

__all__ = [
    'KeystrokeDataSchema',
    'PredictionRequestSchema', 
    'PredictionResponseSchema',
    'ContributionRequestSchema',
    'AdminStatsSchema',
    'PredictionResource',
    'ContributionResource',
    'AdminResource',
    'HealthCheckResource',
    'ModelStatsResource',
    'APIManager'
] 