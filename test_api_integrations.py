#!/usr/bin/env python3
"""
Test script for API integrations.

This script tests the API integrations to ensure they are working correctly
with the provided API keys.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quantum_moe_mas.api.integration_registry import get_integration_registry, initialize_all_integrations
from quantum_moe_mas.api.orchestrator import API_Orchestrator, APIProvider
from quantum_moe_mas.core.logging_simple import setup_logging, get_logger

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)


async def test_integration(provider: APIProvider, integration) -> bool:
    """Test a specific integration."""
    try:
        logger.info(f"Testing {provider.value}...")
        
        # Test health check
        health_ok = await integration.health_check()
        logger.info(f"{provider.value} health check: {'✓' if health_ok else '✗'}")
        
        if not health_ok:
            return False
        
        # Test basic functionality based on capabilities
        capabilities = integration.get_capabilities()
        
        for capability in capabilities:
            if capability.value == "text_generation":
                try:
                    if hasattr(integration, 'generate_text'):
                        response = await integration.generate_text("Hello, world!", max_tokens=10)
                    elif hasattr(integration, 'chat_completion'):
                        response = await integration.chat_completion(
                            messages=[{"role": "user", "content": "Hello"}],
                            max_tokens=10
                        )
                    else:
                        continue
                    
                    if response.success:
                        logger.info(f"{provider.value} text generation: ✓")
                    else:
                        logger.warning(f"{provider.value} text generation failed: {response.error}")
                        
                except Exception as e:
                    logger.warning(f"{provider.value} text generation error: {e}")
                
                break  # Test only one capability per integration
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing {provider.value}: {e}")
        return False


async def main():
    """Main test function."""
    setup_logging()
    logger.info("Starting API integrations test...")
    
    # Check environment variables
    required_keys = [
        "OPENAI_API_KEY",
        "HUGGINGFACE_API_KEY", 
        "GOOGLE_AI_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "COHERE_API_KEY"
    ]
    
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        logger.warning(f"Missing API keys: {missing_keys}")
        logger.info("Some tests may fail due to missing API keys")
    
    # Initialize orchestrator
    orchestrator = API_Orchestrator()
    await orchestrator.initialize()
    
    # Initialize all integrations
    logger.info("Initializing API integrations...")
    results = await initialize_all_integrations(orchestrator)
    
    # Report initialization results
    successful = [provider.value for provider, success in results.items() if success]
    failed = [provider.value for provider, success in results.items() if not success]
    
    logger.info(f"Successfully initialized: {successful}")
    if failed:
        logger.warning(f"Failed to initialize: {failed}")
    
    # Get registry and test integrations
    registry = get_integration_registry()
    
    test_results = {}
    for provider, integration in registry.initialized_integrations.items():
        test_results[provider] = await test_integration(provider, integration)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    for provider, success in test_results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{provider.value:20} {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for success in test_results.values() if success)
    
    logger.info(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    # Cleanup
    await orchestrator.shutdown()
    await registry.shutdown_all_integrations()
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)