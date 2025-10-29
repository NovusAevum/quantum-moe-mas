"""
API Key Management and Authentication System.

This module provides secure API key storage, rotation, and authentication
for all integrated APIs with enterprise-grade security features.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
import base64

from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class KeyStatus(Enum):
    """API key status."""
    
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"
    RATE_LIMITED = "rate_limited"


@dataclass
class APICredentials:
    """API credentials with metadata."""
    
    provider: str
    api_key: str
    api_secret: Optional[str] = None
    endpoint_url: Optional[str] = None
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit_reset: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if credentials are valid and usable."""
        if self.status != KeyStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        if (self.rate_limit_reset and 
            datetime.utcnow() < self.rate_limit_reset):
            return False
        
        return True
    
    def mark_used(self) -> None:
        """Mark credentials as used."""
        self.last_used = datetime.utcnow()
        self.usage_count += 1
    
    def set_rate_limited(self, reset_time: datetime) -> None:
        """Mark credentials as rate limited."""
        self.status = KeyStatus.RATE_LIMITED
        self.rate_limit_reset = reset_time
    
    def clear_rate_limit(self) -> None:
        """Clear rate limit status."""
        if self.status == KeyStatus.RATE_LIMITED:
            self.status = KeyStatus.ACTIVE
        self.rate_limit_reset = None


class APIKeyManager:
    """
    Secure API key management system.
    
    Provides comprehensive key management including:
    - Encrypted storage of API keys
    - Key rotation and expiration
    - Usage tracking and analytics
    - Rate limit management
    - Automatic key validation
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
        auto_rotate_days: int = 90
    ) -> None:
        """
        Initialize API key manager.
        
        Args:
            storage_path: Path to encrypted key storage file
            encryption_key: Encryption key for secure storage
            auto_rotate_days: Days before automatic key rotation
        """
        self.storage_path = storage_path or os.path.expanduser("~/.quantum_moe_mas/api_keys.enc")
        self.auto_rotate_days = auto_rotate_days
        
        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            # Generate or load encryption key
            key_path = os.path.expanduser("~/.quantum_moe_mas/encryption.key")
            if os.path.exists(key_path):
                with open(key_path, 'rb') as f:
                    self.cipher = Fernet(f.read())
            else:
                # Generate new key
                key = Fernet.generate_key()
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                with open(key_path, 'wb') as f:
                    f.write(key)
                self.cipher = Fernet(key)
                logger.info(f"Generated new encryption key at {key_path}")
        
        # In-memory storage
        self.credentials: Dict[str, List[APICredentials]] = {}
        self.signup_links: Dict[str, str] = {}
        
        # Load default signup links
        self._load_default_signup_links()
        
        logger.info(
            "Initialized APIKeyManager",
            storage_path=self.storage_path,
            auto_rotate_days=auto_rotate_days
        )
    
    async def initialize(self) -> None:
        """Initialize the key manager and load stored credentials."""
        try:
            await self.load_credentials()
            logger.info("APIKeyManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize APIKeyManager: {e}")
            # Continue with empty credentials if loading fails
            self.credentials = {}
    
    def _load_default_signup_links(self) -> None:
        """Load default signup links for API providers."""
        self.signup_links = {
            "openai_playground": "https://platform.openai.com/signup",
            "hugging_face": "https://huggingface.co/join",
            "google_ai_studio": "https://makersuite.google.com/",
            "groq": "https://console.groq.com/",
            "cerebras": "https://inference.cerebras.ai/",
            "deepseek": "https://platform.deepseek.com/",
            "cohere": "https://dashboard.cohere.ai/",
            "anthropic_claude": "https://console.anthropic.com/",
            "mistral": "https://console.mistral.ai/",
            "together_ai": "https://api.together.xyz/",
            "flux_11": "https://replicate.com/black-forest-labs/flux-1.1-pro",
            "stability_ai": "https://platform.stability.ai/",
            "replicate": "https://replicate.com/",
            "perplexity": "https://www.perplexity.ai/settings/api",
            "you_com": "https://api.you.com/",
            "brave_search": "https://api.search.brave.com/",
            "serper": "https://serper.dev/",
            "tavily": "https://tavily.com/",
            "voyage_ai": "https://www.voyageai.com/",
            "jina_ai": "https://jina.ai/",
            "eleven_labs": "https://elevenlabs.io/",
        }
    
    async def add_credentials(
        self,
        provider: str,
        api_key: str,
        api_secret: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APICredentials:
        """
        Add new API credentials.
        
        Args:
            provider: API provider name
            api_key: API key
            api_secret: Optional API secret
            endpoint_url: Optional custom endpoint URL
            expires_at: Optional expiration date
            metadata: Optional metadata
        
        Returns:
            APICredentials instance
        """
        credentials = APICredentials(
            provider=provider,
            api_key=api_key,
            api_secret=api_secret,
            endpoint_url=endpoint_url,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        if provider not in self.credentials:
            self.credentials[provider] = []
        
        self.credentials[provider].append(credentials)
        
        # Save to encrypted storage
        await self.save_credentials()
        
        logger.info(f"Added credentials for {provider}")
        return credentials
    
    async def get_credentials(
        self,
        provider: str,
        prefer_unused: bool = True
    ) -> Optional[APICredentials]:
        """
        Get valid credentials for a provider.
        
        Args:
            provider: API provider name
            prefer_unused: Prefer credentials with lower usage
        
        Returns:
            Valid APICredentials or None
        """
        if provider not in self.credentials:
            logger.warning(f"No credentials found for {provider}")
            return None
        
        # Filter valid credentials
        valid_creds = [
            cred for cred in self.credentials[provider]
            if cred.is_valid()
        ]
        
        if not valid_creds:
            logger.warning(f"No valid credentials for {provider}")
            return None
        
        # Sort by usage if prefer_unused is True
        if prefer_unused:
            valid_creds.sort(key=lambda c: (c.usage_count, c.last_used or datetime.min))
        
        selected = valid_creds[0]
        selected.mark_used()
        
        # Save updated usage
        await self.save_credentials()
        
        return selected
    
    async def rotate_credentials(
        self,
        provider: str,
        old_credentials: APICredentials,
        new_api_key: str,
        new_api_secret: Optional[str] = None
    ) -> APICredentials:
        """
        Rotate API credentials.
        
        Args:
            provider: API provider name
            old_credentials: Credentials to replace
            new_api_key: New API key
            new_api_secret: Optional new API secret
        
        Returns:
            New APICredentials instance
        """
        # Mark old credentials as expired
        old_credentials.status = KeyStatus.EXPIRED
        
        # Add new credentials
        new_credentials = await self.add_credentials(
            provider=provider,
            api_key=new_api_key,
            api_secret=new_api_secret,
            endpoint_url=old_credentials.endpoint_url,
            metadata=old_credentials.metadata.copy()
        )
        
        logger.info(f"Rotated credentials for {provider}")
        return new_credentials
    
    async def revoke_credentials(
        self,
        provider: str,
        credentials: APICredentials
    ) -> None:
        """
        Revoke API credentials.
        
        Args:
            provider: API provider name
            credentials: Credentials to revoke
        """
        credentials.status = KeyStatus.REVOKED
        await self.save_credentials()
        logger.info(f"Revoked credentials for {provider}")
    
    async def handle_rate_limit(
        self,
        provider: str,
        credentials: APICredentials,
        reset_time: datetime
    ) -> None:
        """
        Handle rate limit for credentials.
        
        Args:
            provider: API provider name
            credentials: Rate limited credentials
            reset_time: When rate limit resets
        """
        credentials.set_rate_limited(reset_time)
        await self.save_credentials()
        logger.warning(f"Rate limit applied to {provider} until {reset_time}")
    
    async def check_expiring_credentials(self) -> List[APICredentials]:
        """
        Check for credentials expiring soon.
        
        Returns:
            List of credentials expiring within auto_rotate_days
        """
        expiring = []
        cutoff_date = datetime.utcnow() + timedelta(days=self.auto_rotate_days)
        
        for provider_creds in self.credentials.values():
            for cred in provider_creds:
                if (cred.expires_at and 
                    cred.expires_at <= cutoff_date and 
                    cred.status == KeyStatus.ACTIVE):
                    expiring.append(cred)
        
        return expiring
    
    async def cleanup_expired_credentials(self) -> int:
        """
        Remove expired credentials.
        
        Returns:
            Number of credentials removed
        """
        removed_count = 0
        
        for provider in list(self.credentials.keys()):
            original_count = len(self.credentials[provider])
            
            # Keep only non-expired credentials
            self.credentials[provider] = [
                cred for cred in self.credentials[provider]
                if not (cred.expires_at and datetime.utcnow() > cred.expires_at)
            ]
            
            removed = original_count - len(self.credentials[provider])
            removed_count += removed
            
            if removed > 0:
                logger.info(f"Removed {removed} expired credentials for {provider}")
        
        if removed_count > 0:
            await self.save_credentials()
        
        return removed_count
    
    def get_signup_link(self, provider: str) -> Optional[str]:
        """
        Get signup link for a provider.
        
        Args:
            provider: API provider name
        
        Returns:
            Signup URL or None
        """
        return self.signup_links.get(provider)
    
    def add_signup_link(self, provider: str, url: str) -> None:
        """
        Add or update signup link for a provider.
        
        Args:
            provider: API provider name
            url: Signup URL
        """
        self.signup_links[provider] = url
        logger.info(f"Added signup link for {provider}: {url}")
    
    async def save_credentials(self) -> None:
        """Save credentials to encrypted storage."""
        try:
            # Prepare data for serialization
            data = {}
            for provider, creds_list in self.credentials.items():
                data[provider] = []
                for cred in creds_list:
                    cred_data = {
                        "api_key": cred.api_key,
                        "api_secret": cred.api_secret,
                        "endpoint_url": cred.endpoint_url,
                        "status": cred.status.value,
                        "created_at": cred.created_at.isoformat(),
                        "expires_at": cred.expires_at.isoformat() if cred.expires_at else None,
                        "last_used": cred.last_used.isoformat() if cred.last_used else None,
                        "usage_count": cred.usage_count,
                        "rate_limit_reset": cred.rate_limit_reset.isoformat() if cred.rate_limit_reset else None,
                        "metadata": cred.metadata,
                    }
                    data[provider].append(cred_data)
            
            # Encrypt and save
            json_data = json.dumps(data).encode()
            encrypted_data = self.cipher.encrypt(json_data)
            
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.debug("Credentials saved to encrypted storage")
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise
    
    async def load_credentials(self) -> None:
        """Load credentials from encrypted storage."""
        if not os.path.exists(self.storage_path):
            logger.info("No existing credentials file found")
            return
        
        try:
            with open(self.storage_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())
            
            # Reconstruct credentials
            self.credentials = {}
            for provider, creds_list in data.items():
                self.credentials[provider] = []
                for cred_data in creds_list:
                    cred = APICredentials(
                        provider=provider,
                        api_key=cred_data["api_key"],
                        api_secret=cred_data.get("api_secret"),
                        endpoint_url=cred_data.get("endpoint_url"),
                        status=KeyStatus(cred_data["status"]),
                        created_at=datetime.fromisoformat(cred_data["created_at"]),
                        expires_at=datetime.fromisoformat(cred_data["expires_at"]) if cred_data.get("expires_at") else None,
                        last_used=datetime.fromisoformat(cred_data["last_used"]) if cred_data.get("last_used") else None,
                        usage_count=cred_data.get("usage_count", 0),
                        rate_limit_reset=datetime.fromisoformat(cred_data["rate_limit_reset"]) if cred_data.get("rate_limit_reset") else None,
                        metadata=cred_data.get("metadata", {}),
                    )
                    self.credentials[provider].append(cred)
            
            logger.info(f"Loaded credentials for {len(self.credentials)} providers")
            
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            # Initialize empty credentials on load failure
            self.credentials = {}
    
    def get_credentials_status(self) -> Dict[str, Any]:
        """
        Get comprehensive credentials status.
        
        Returns:
            Dictionary with credentials statistics
        """
        total_providers = len(self.credentials)
        total_credentials = sum(len(creds) for creds in self.credentials.values())
        
        status_counts = {}
        for creds_list in self.credentials.values():
            for cred in creds_list:
                status = cred.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_providers": total_providers,
            "total_credentials": total_credentials,
            "status_distribution": status_counts,
            "providers": {
                provider: {
                    "credential_count": len(creds),
                    "valid_credentials": len([c for c in creds if c.is_valid()]),
                    "total_usage": sum(c.usage_count for c in creds),
                }
                for provider, creds in self.credentials.items()
            },
            "signup_links_available": len(self.signup_links),
        }