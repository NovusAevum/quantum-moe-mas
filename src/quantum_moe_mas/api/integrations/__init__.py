"""
API Integrations module for Quantum MoE MAS.

This module provides specific API integrations for 30+ free AI services
with consistent interfaces and comprehensive error handling.
"""

from quantum_moe_mas.api.integrations.base import (
    BaseAPIIntegration,
    APIResponse,
    APIError,
    APICapability,
    IntegrationConfig
)

# Language Model Integrations
from quantum_moe_mas.api.integrations.openai_playground import OpenAIPlaygroundIntegration
from quantum_moe_mas.api.integrations.hugging_face import HuggingFaceIntegration
from quantum_moe_mas.api.integrations.google_ai_studio import GoogleAIStudioIntegration
from quantum_moe_mas.api.integrations.groq import GroqIntegration
from quantum_moe_mas.api.integrations.cerebras import CerebrasIntegration
from quantum_moe_mas.api.integrations.deepseek import DeepSeekIntegration
from quantum_moe_mas.api.integrations.cohere import CohereIntegration

# Vision & Multimodal Integrations
from quantum_moe_mas.api.integrations.flux_11 import Flux11Integration
from quantum_moe_mas.api.integrations.stability_ai import StabilityAIIntegration
from quantum_moe_mas.api.integrations.replicate import ReplicateIntegration

__all__ = [
    # Base classes
    "BaseAPIIntegration",
    "APIResponse", 
    "APIError",
    "APICapability",
    "IntegrationConfig",
    
    # Language Model Integrations
    "OpenAIPlaygroundIntegration",
    "HuggingFaceIntegration",
    "GoogleAIStudioIntegration",
    "GroqIntegration",
    "CerebrasIntegration",
    "DeepSeekIntegration",
    "CohereIntegration",
    
    # Vision & Multimodal Integrations
    "Flux11Integration",
    "StabilityAIIntegration",
    "ReplicateIntegration",
]

__version__ = "0.1.0"