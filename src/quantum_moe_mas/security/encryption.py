"""
Data Encryption and Protection Services.

This module provides comprehensive data encryption capabilities including:
- Data encryption at rest and in transit
- Secure key management and rotation
- Field-level encryption for sensitive data
- Cryptographic utilities and secure storage
"""

import base64
import hashlib
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from quantum_moe_mas.core.logging import get_logger, get_security_logger
from quantum_moe_mas.core.exceptions import QuantumMoEMASError
from quantum_moe_mas.config.settings import get_settings

logger = get_logger(__name__)
security_logger = get_security_logger(__name__)
settings = get_settings()


class EncryptionError(QuantumMoEMASError):
    """Raised when encryption/decryption operations fail."""
    pass


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    
    FERNET = "fernet"
    AES_256_GCM = "aes_256_gcm"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"


class DataClassification(Enum):
    """Data classification levels."""
    
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class EncryptionKey:
    """Encryption key with metadata."""
    
    key_id: str
    algorithm: EncryptionAlgorithm
    key_data: bytes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class EncryptedData:
    """Encrypted data with metadata."""
    
    data: bytes
    algorithm: EncryptionAlgorithm
    key_id: str
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": base64.b64encode(self.data).decode(),
            "algorithm": self.algorithm.value,
            "key_id": self.key_id,
            "iv": base64.b64encode(self.iv).decode() if self.iv else None,
            "tag": base64.b64encode(self.tag).decode() if self.tag else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedData":
        """Create from dictionary."""
        return cls(
            data=base64.b64decode(data["data"]),
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            key_id=data["key_id"],
            iv=base64.b64decode(data["iv"]) if data.get("iv") else None,
            tag=base64.b64decode(data["tag"]) if data.get("tag") else None,
            metadata=data.get("metadata", {}),
        )


class KeyManager:
    """Secure key management service."""
    
    def __init__(self, key_storage_path: Optional[str] = None):
        self.key_storage_path = key_storage_path or os.path.expanduser(
            "~/.quantum_moe_mas/encryption_keys"
        )
        self.keys: Dict[str, EncryptionKey] = {}
        self.master_key: Optional[bytes] = None
        
        # Initialize master key
        self._initialize_master_key()
    
    def _initialize_master_key(self) -> None:
        """Initialize or load master key."""
        master_key_path = os.path.join(
            os.path.dirname(self.key_storage_path),
            "master.key"
        )
        
        if os.path.exists(master_key_path):
            with open(master_key_path, "rb") as f:
                self.master_key = f.read()
        else:
            # Generate new master key
            self.master_key = Fernet.generate_key()
            os.makedirs(os.path.dirname(master_key_path), exist_ok=True)
            with open(master_key_path, "wb") as f:
                f.write(self.master_key)
            
            # Set restrictive permissions
            os.chmod(master_key_path, 0o600)
            
            logger.info("Generated new master encryption key")
    
    def generate_key(
        self,
        algorithm: EncryptionAlgorithm,
        key_id: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> EncryptionKey:
        """Generate new encryption key."""
        if not key_id:
            key_id = secrets.token_urlsafe(16)
        
        if algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
        elif algorithm == EncryptionAlgorithm.AES_256_GCM:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise EncryptionError(f"Unsupported algorithm: {algorithm}")
        
        encryption_key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            key_data=key_data,
            expires_at=expires_at
        )
        
        self.keys[key_id] = encryption_key
        self._save_key(encryption_key)
        
        logger.info(f"Generated encryption key", key_id=key_id, algorithm=algorithm.value)
        return encryption_key
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get encryption key by ID."""
        key = self.keys.get(key_id)
        if key and key.is_expired():
            logger.warning(f"Encryption key expired", key_id=key_id)
            return None
        return key
    
    def rotate_key(self, old_key_id: str) -> EncryptionKey:
        """Rotate encryption key."""
        old_key = self.get_key(old_key_id)
        if not old_key:
            raise EncryptionError(f"Key not found: {old_key_id}")
        
        # Generate new key with same algorithm
        new_key = self.generate_key(old_key.algorithm)
        
        # Deactivate old key
        old_key.is_active = False
        self._save_key(old_key)
        
        logger.info(f"Rotated encryption key", old_key_id=old_key_id, new_key_id=new_key.key_id)
        return new_key
    
    def _save_key(self, key: EncryptionKey) -> None:
        """Save key to secure storage."""
        if not self.master_key:
            raise EncryptionError("Master key not initialized")
        
        # Encrypt key data with master key
        fernet = Fernet(self.master_key)
        encrypted_key_data = fernet.encrypt(key.key_data)
        
        key_file_path = os.path.join(self.key_storage_path, f"{key.key_id}.key")
        os.makedirs(os.path.dirname(key_file_path), exist_ok=True)
        
        key_metadata = {
            "key_id": key.key_id,
            "algorithm": key.algorithm.value,
            "created_at": key.created_at.isoformat(),
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            "is_active": key.is_active,
            "metadata": key.metadata,
        }
        
        with open(key_file_path, "wb") as f:
            # Write metadata length
            metadata_bytes = str(key_metadata).encode()
            f.write(len(metadata_bytes).to_bytes(4, byteorder="big"))
            f.write(metadata_bytes)
            # Write encrypted key data
            f.write(encrypted_key_data)
        
        # Set restrictive permissions
        os.chmod(key_file_path, 0o600)


class EncryptionService:
    """Core encryption service."""
    
    def __init__(self, key_manager: Optional[KeyManager] = None):
        self.key_manager = key_manager or KeyManager()
        
        # Initialize default keys if not exist
        self._initialize_default_keys()
    
    def _initialize_default_keys(self) -> None:
        """Initialize default encryption keys."""
        # Check if default Fernet key exists
        default_key_id = "default_fernet"
        if not self.key_manager.get_key(default_key_id):
            self.key_manager.generate_key(
                EncryptionAlgorithm.FERNET,
                key_id=default_key_id
            )
        
        # Check if default AES key exists
        aes_key_id = "default_aes"
        if not self.key_manager.get_key(aes_key_id):
            self.key_manager.generate_key(
                EncryptionAlgorithm.AES_256_GCM,
                key_id=aes_key_id
            )
    
    def encrypt(
        self,
        data: Union[str, bytes],
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET,
        key_id: Optional[str] = None
    ) -> EncryptedData:
        """Encrypt data using specified algorithm."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        
        # Get or generate key
        if not key_id:
            key_id = f"default_{algorithm.value.split('_')[0]}"
        
        key = self.key_manager.get_key(key_id)
        if not key:
            key = self.key_manager.generate_key(algorithm, key_id)
        
        try:
            if algorithm == EncryptionAlgorithm.FERNET:
                fernet = Fernet(key.key_data)
                encrypted_data = fernet.encrypt(data)
                
                return EncryptedData(
                    data=encrypted_data,
                    algorithm=algorithm,
                    key_id=key_id
                )
            
            elif algorithm == EncryptionAlgorithm.AES_256_GCM:
                # Generate random IV
                iv = secrets.token_bytes(12)  # 96 bits for GCM
                
                cipher = Cipher(
                    algorithms.AES(key.key_data),
                    modes.GCM(iv)
                )
                encryptor = cipher.encryptor()
                encrypted_data = encryptor.update(data) + encryptor.finalize()
                
                return EncryptedData(
                    data=encrypted_data,
                    algorithm=algorithm,
                    key_id=key_id,
                    iv=iv,
                    tag=encryptor.tag
                )
            
            else:
                raise EncryptionError(f"Encryption not implemented for {algorithm}")
        
        except Exception as e:
            logger.error(f"Encryption failed", algorithm=algorithm.value, error=str(e))
            raise EncryptionError(f"Encryption failed: {str(e)}")
    
    def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data."""
        key = self.key_manager.get_key(encrypted_data.key_id)
        if not key:
            raise EncryptionError(f"Encryption key not found: {encrypted_data.key_id}")
        
        try:
            if encrypted_data.algorithm == EncryptionAlgorithm.FERNET:
                fernet = Fernet(key.key_data)
                return fernet.decrypt(encrypted_data.data)
            
            elif encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM:
                if not encrypted_data.iv or not encrypted_data.tag:
                    raise EncryptionError("IV and tag required for AES-GCM decryption")
                
                cipher = Cipher(
                    algorithms.AES(key.key_data),
                    modes.GCM(encrypted_data.iv, encrypted_data.tag)
                )
                decryptor = cipher.decryptor()
                return decryptor.update(encrypted_data.data) + decryptor.finalize()
            
            else:
                raise EncryptionError(f"Decryption not implemented for {encrypted_data.algorithm}")
        
        except Exception as e:
            logger.error(f"Decryption failed", algorithm=encrypted_data.algorithm.value, error=str(e))
            raise EncryptionError(f"Decryption failed: {str(e)}")
    
    def encrypt_string(self, text: str, **kwargs) -> str:
        """Encrypt string and return base64 encoded result."""
        encrypted_data = self.encrypt(text, **kwargs)
        return base64.b64encode(encrypted_data.data).decode()
    
    def decrypt_string(self, encrypted_text: str, key_id: str, algorithm: EncryptionAlgorithm) -> str:
        """Decrypt base64 encoded string."""
        encrypted_data = EncryptedData(
            data=base64.b64decode(encrypted_text),
            algorithm=algorithm,
            key_id=key_id
        )
        decrypted_bytes = self.decrypt(encrypted_data)
        return decrypted_bytes.decode("utf-8")


class DataProtectionService:
    """Service for protecting sensitive data with field-level encryption."""
    
    def __init__(self, encryption_service: Optional[EncryptionService] = None):
        self.encryption_service = encryption_service or EncryptionService()
        
        # Define sensitive field patterns
        self.sensitive_patterns = {
            "password", "secret", "key", "token", "credential",
            "ssn", "social_security", "credit_card", "card_number",
            "email", "phone", "address", "personal"
        }
    
    def is_sensitive_field(self, field_name: str) -> bool:
        """Check if field contains sensitive data."""
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in self.sensitive_patterns)
    
    def protect_data(
        self,
        data: Dict[str, Any],
        classification: DataClassification = DataClassification.CONFIDENTIAL
    ) -> Dict[str, Any]:
        """Protect sensitive fields in data dictionary."""
        protected_data = {}
        
        for key, value in data.items():
            if self.is_sensitive_field(key) and isinstance(value, (str, bytes)):
                # Encrypt sensitive field
                encrypted = self.encryption_service.encrypt(value)
                protected_data[key] = {
                    "_encrypted": True,
                    "_data": encrypted.to_dict(),
                    "_classification": classification.value
                }
                
                security_logger.data_access(
                    resource=key,
                    action="encrypt",
                    sensitive=True,
                    classification=classification.value
                )
            else:
                protected_data[key] = value
        
        return protected_data
    
    def unprotect_data(self, protected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt protected fields in data dictionary."""
        unprotected_data = {}
        
        for key, value in protected_data.items():
            if isinstance(value, dict) and value.get("_encrypted"):
                # Decrypt field
                encrypted_data = EncryptedData.from_dict(value["_data"])
                decrypted_bytes = self.encryption_service.decrypt(encrypted_data)
                unprotected_data[key] = decrypted_bytes.decode("utf-8")
                
                security_logger.data_access(
                    resource=key,
                    action="decrypt",
                    sensitive=True,
                    classification=value.get("_classification", "unknown")
                )
            else:
                unprotected_data[key] = value
        
        return unprotected_data
    
    def hash_sensitive_data(self, data: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """Hash sensitive data with salt."""
        if not salt:
            salt = secrets.token_bytes(32)
        
        # Use Scrypt for password hashing (more secure than PBKDF2)
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            n=2**14,  # CPU/memory cost parameter
            r=8,      # Block size parameter
            p=1,      # Parallelization parameter
        )
        
        key = kdf.derive(data.encode("utf-8"))
        hash_value = base64.b64encode(key).decode()
        
        return hash_value, salt


class SecureStorage:
    """Secure storage service for encrypted data persistence."""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        encryption_service: Optional[EncryptionService] = None
    ):
        self.storage_path = storage_path or os.path.expanduser(
            "~/.quantum_moe_mas/secure_storage"
        )
        self.encryption_service = encryption_service or EncryptionService()
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        os.chmod(self.storage_path, 0o700)  # Owner read/write/execute only
    
    def store(
        self,
        key: str,
        data: Union[str, bytes, Dict[str, Any]],
        classification: DataClassification = DataClassification.CONFIDENTIAL
    ) -> None:
        """Store data securely."""
        if isinstance(data, dict):
            data = str(data).encode("utf-8")
        elif isinstance(data, str):
            data = data.encode("utf-8")
        
        # Encrypt data
        encrypted_data = self.encryption_service.encrypt(data)
        
        # Store encrypted data
        file_path = os.path.join(self.storage_path, f"{key}.enc")
        with open(file_path, "wb") as f:
            # Write metadata
            metadata = {
                "classification": classification.value,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "encryption": encrypted_data.to_dict()
            }
            metadata_bytes = str(metadata).encode("utf-8")
            f.write(len(metadata_bytes).to_bytes(4, byteorder="big"))
            f.write(metadata_bytes)
        
        # Set restrictive permissions
        os.chmod(file_path, 0o600)
        
        logger.info(f"Data stored securely", key=key, classification=classification.value)
    
    def retrieve(self, key: str) -> Optional[bytes]:
        """Retrieve and decrypt stored data."""
        file_path = os.path.join(self.storage_path, f"{key}.enc")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, "rb") as f:
                # Read metadata
                metadata_length = int.from_bytes(f.read(4), byteorder="big")
                metadata_bytes = f.read(metadata_length)
                metadata = eval(metadata_bytes.decode("utf-8"))  # Note: Use json in production
                
                # Reconstruct encrypted data
                encrypted_data = EncryptedData.from_dict(metadata["encryption"])
                
                # Decrypt data
                decrypted_data = self.encryption_service.decrypt(encrypted_data)
                
                security_logger.data_access(
                    resource=key,
                    action="retrieve",
                    sensitive=True,
                    classification=metadata.get("classification", "unknown")
                )
                
                return decrypted_data
        
        except Exception as e:
            logger.error(f"Failed to retrieve secure data", key=key, error=str(e))
            return None
    
    def delete(self, key: str) -> bool:
        """Securely delete stored data."""
        file_path = os.path.join(self.storage_path, f"{key}.enc")
        
        if not os.path.exists(file_path):
            return False
        
        try:
            # Overwrite file with random data before deletion
            file_size = os.path.getsize(file_path)
            with open(file_path, "wb") as f:
                f.write(secrets.token_bytes(file_size))
            
            # Delete file
            os.remove(file_path)
            
            logger.info(f"Secure data deleted", key=key)
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete secure data", key=key, error=str(e))
            return False
    
    def list_keys(self) -> List[str]:
        """List all stored keys."""
        keys = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".enc"):
                keys.append(filename[:-4])  # Remove .enc extension
        return keys


# Global instances
encryption_service = EncryptionService()
data_protection_service = DataProtectionService(encryption_service)
secure_storage = SecureStorage(encryption_service=encryption_service)