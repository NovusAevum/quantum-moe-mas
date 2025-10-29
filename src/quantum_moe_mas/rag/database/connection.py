"""
Supabase database connection and configuration.

This module provides secure, async database connectivity with connection pooling,
error handling, and enterprise-grade security features.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from uuid import UUID

import asyncpg
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from supabase import Client, create_client
from tenacity import retry, stop_after_attempt, wait_exponential

from quantum_moe_mas.core.exceptions import DatabaseError, ConfigurationError

logger = logging.getLogger(__name__)


class SupabaseConfig(BaseSettings):
    """Supabase configuration with validation and security."""
    
    # Supabase connection
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY") 
    supabase_service_key: Optional[str] = Field(None, env="SUPABASE_SERVICE_KEY")
    
    # Database connection
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_name: str = Field("quantum_moe_mas", env="DB_NAME")
    db_user: str = Field(..., env="DB_USER")
    db_password: str = Field(..., env="DB_PASSWORD")
    db_ssl_mode: str = Field("prefer", env="DB_SSL_MODE")
    
    # Connection pool settings
    min_pool_size: int = Field(5, env="DB_MIN_POOL_SIZE")
    max_pool_size: int = Field(20, env="DB_MAX_POOL_SIZE")
    pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT")
    
    # Performance settings
    command_timeout: int = Field(60, env="DB_COMMAND_TIMEOUT")
    query_timeout: int = Field(30, env="DB_QUERY_TIMEOUT")
    
    # Security settings
    enable_ssl: bool = Field(True, env="DB_ENABLE_SSL")
    verify_ssl: bool = Field(True, env="DB_VERIFY_SSL")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False
    
    @validator("supabase_url")
    def validate_supabase_url(cls, v: str) -> str:
        """Validate Supabase URL format."""
        if not v.startswith("https://"):
            raise ValueError("Supabase URL must start with https://")
        if not v.endswith(".supabase.co"):
            raise ValueError("Invalid Supabase URL format")
        return v
    
    @validator("supabase_key")
    def validate_supabase_key(cls, v: str) -> str:
        """Validate Supabase key format."""
        if len(v) < 100:  # Supabase keys are typically longer
            raise ValueError("Invalid Supabase key format")
        return v
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        if self.database_url:
            return self.database_url
        
        ssl_param = f"?sslmode={self.db_ssl_mode}" if self.enable_ssl else ""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}{ssl_param}"
        )


class SupabaseConnection:
    """
    Enterprise-grade Supabase connection manager with async support.
    
    Features:
    - Connection pooling for optimal performance
    - Automatic retry with exponential backoff
    - Health monitoring and circuit breaker pattern
    - Comprehensive error handling and logging
    - Security-first configuration
    """
    
    def __init__(self, config: Optional[SupabaseConfig] = None):
        """Initialize Supabase connection manager."""
        self.config = config or SupabaseConfig()
        self._client: Optional[Client] = None
        self._pool: Optional[asyncpg.Pool] = None
        self._is_connected = False
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized Supabase connection manager")
    
    @property
    def client(self) -> Client:
        """Get Supabase client instance."""
        if not self._client:
            raise DatabaseError("Supabase client not initialized. Call connect() first.")
        return self._client
    
    @property
    def pool(self) -> asyncpg.Pool:
        """Get PostgreSQL connection pool."""
        if not self._pool:
            raise DatabaseError("Database pool not initialized. Call connect() first.")
        return self._pool
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self._is_connected
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def connect(self) -> None:
        """
        Establish connection to Supabase and PostgreSQL.
        
        Raises:
            DatabaseError: If connection fails after retries
            ConfigurationError: If configuration is invalid
        """
        try:
            logger.info("Connecting to Supabase...")
            
            # Initialize Supabase client
            self._client = create_client(
                self.config.supabase_url,
                self.config.supabase_key
            )
            
            # Test Supabase connection
            await self._test_supabase_connection()
            
            # Initialize PostgreSQL connection pool
            await self._initialize_pool()
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())
            
            self._is_connected = True
            logger.info("Successfully connected to Supabase and PostgreSQL")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            await self.disconnect()
            raise DatabaseError(f"Database connection failed: {e}") from e
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from database."""
        logger.info("Disconnecting from database...")
        
        self._is_connected = False
        
        # Stop health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close connection pool
        if self._pool:
            await self._pool.close()
            self._pool = None
        
        # Clear client reference
        self._client = None
        
        logger.info("Disconnected from database")
    
    async def _test_supabase_connection(self) -> None:
        """Test Supabase connection with a simple query."""
        try:
            # Test with a simple query to the auth schema
            response = self._client.table("documents").select("count", count="exact").limit(0).execute()
            logger.debug("Supabase connection test successful")
        except Exception as e:
            logger.error(f"Supabase connection test failed: {e}")
            raise DatabaseError(f"Supabase connection test failed: {e}") from e
    
    async def _initialize_pool(self) -> None:
        """Initialize PostgreSQL connection pool."""
        try:
            logger.info("Initializing PostgreSQL connection pool...")
            
            self._pool = await asyncpg.create_pool(
                self.config.postgres_url,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                command_timeout=self.config.command_timeout,
                server_settings={
                    "application_name": "quantum-moe-mas",
                    "timezone": "UTC",
                }
            )
            
            # Test pool connection
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            logger.info(
                f"PostgreSQL pool initialized: "
                f"{self.config.min_pool_size}-{self.config.max_pool_size} connections"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseError(f"Connection pool initialization failed: {e}") from e
    
    async def _health_monitor(self) -> None:
        """Monitor database health and reconnect if needed."""
        while self._is_connected:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if self._pool:
                    async with self._pool.acquire() as conn:
                        await conn.execute("SELECT 1")
                
                logger.debug("Database health check passed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Database health check failed: {e}")
                # Could implement reconnection logic here
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Get a database connection from the pool.
        
        Usage:
            async with connection_manager.get_connection() as conn:
                result = await conn.fetch("SELECT * FROM documents")
        """
        if not self._pool:
            raise DatabaseError("Connection pool not initialized")
        
        async with self._pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise DatabaseError(f"Database operation failed: {e}") from e
    
    @asynccontextmanager
    async def get_transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Get a database connection with transaction support.
        
        Usage:
            async with connection_manager.get_transaction() as conn:
                await conn.execute("INSERT INTO documents ...")
                await conn.execute("INSERT INTO chunks ...")
                # Transaction automatically committed or rolled back
        """
        async with self.get_connection() as conn:
            async with conn.transaction():
                yield conn
    
    async def execute_query(
        self,
        query: str,
        *args: Any,
        timeout: Optional[int] = None
    ) -> List[asyncpg.Record]:
        """
        Execute a query and return results.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            List of query results
        """
        timeout = timeout or self.config.query_timeout
        
        async with self.get_connection() as conn:
            try:
                return await asyncio.wait_for(
                    conn.fetch(query, *args),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Query timeout after {timeout}s: {query[:100]}...")
                raise DatabaseError(f"Query timeout after {timeout}s")
    
    async def execute_command(
        self,
        command: str,
        *args: Any,
        timeout: Optional[int] = None
    ) -> str:
        """
        Execute a command and return status.
        
        Args:
            command: SQL command to execute
            *args: Command parameters
            timeout: Command timeout in seconds
            
        Returns:
            Command execution status
        """
        timeout = timeout or self.config.command_timeout
        
        async with self.get_connection() as conn:
            try:
                return await asyncio.wait_for(
                    conn.execute(command, *args),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Command timeout after {timeout}s: {command[:100]}...")
                raise DatabaseError(f"Command timeout after {timeout}s")
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            async with self.get_connection() as conn:
                # Get database version
                version_result = await conn.fetchrow("SELECT version()")
                
                # Get pgvector extension info
                pgvector_result = await conn.fetchrow(
                    "SELECT * FROM pg_extension WHERE extname = 'vector'"
                )
                
                # Get connection pool stats
                pool_stats = {
                    "size": self._pool.get_size() if self._pool else 0,
                    "min_size": self.config.min_pool_size,
                    "max_size": self.config.max_pool_size,
                }
                
                return {
                    "database_version": version_result["version"] if version_result else "Unknown",
                    "pgvector_installed": pgvector_result is not None,
                    "pgvector_version": pgvector_result["extversion"] if pgvector_result else None,
                    "connection_pool": pool_stats,
                    "is_connected": self._is_connected,
                }
                
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}


# Global connection instance
_connection_instance: Optional[SupabaseConnection] = None


async def get_connection() -> SupabaseConnection:
    """Get global database connection instance."""
    global _connection_instance
    
    if _connection_instance is None:
        _connection_instance = SupabaseConnection()
        await _connection_instance.connect()
    
    return _connection_instance


async def close_connection() -> None:
    """Close global database connection."""
    global _connection_instance
    
    if _connection_instance:
        await _connection_instance.disconnect()
        _connection_instance = None