"""
API Integration Registry.

This module provides a registry for managing all API integrations
with dynamic loading and configuration capabilities.
"""

from typing import Dict, List, Optional, Type
from quantum_moe_mas.api.integrations.base import BaseAPIIntegration
from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class APIIntegrationRegistry:
    """
    Registry for managing API integrations.
    
    Provides centralized management of all available API integrations
    with dynamic loading and configuration capabilities.
    """
    
    def __init__(self) -> None:
        """Initialize the integration registry."""
        self.integrations: Dict[str, Type[BaseAPIIntegration]] = {}
        self.instances: Dict[str, BaseAPIIntegration] = {}
        
        logger.info("Initialized APIIntegrationRegistry")
    
    def register(
        self,
        name: str,
        integration_class: Type[BaseAPIIntegration]
    ) -> None:
        """
        Register an API integration.
        
        Args:
            name: Integration name
            integration_class: Integration class
        """
        self.integrations[name] = integration_class
        logger.info(f"Registered integration: {name}")
    
    def get_integration(self, name: str) -> Optional[Type[BaseAPIIntegration]]:
        """
        Get an integration class by name.
        
        Args:
            name: Integration name
        
        Returns:
            Integration class or None if not found
        """
        return self.integrations.get(name)
    
    def list_integrations(self) -> List[str]:
        """
        List all registered integrations.
        
        Returns:
            List of integration names
        """
        return list(self.integrations.keys())
    
    def create_instance(
        self,
        name: str,
        config: Optional[Dict] = None
    ) -> Optional[BaseAPIIntegration]:
        """
        Create an instance of an integration.
        
        Args:
            name: Integration name
            config: Optional configuration
        
        Returns:
            Integration instance or None if not found
        """
        integration_class = self.integrations.get(name)
        if not integration_class:
            return None
        
        # Create instance with config
        instance = integration_class(config or {})
        self.instances[name] = instance
        
        return instance


# Global registry instance
_registry = APIIntegrationRegistry()


def get_integration_registry() -> APIIntegrationRegistry:
    """Get the global integration registry."""
    return _registry