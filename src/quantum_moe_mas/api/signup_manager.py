"""
Automated Signup Link Generation and Key Management System.

This module provides automated signup link generation, key management workflows,
and integration assistance for API providers.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from urllib.parse import urljoin, urlparse

from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class ProviderCategory(Enum):
    """Categories of API providers."""
    
    LANGUAGE_MODEL = "language_model"
    VISION_MODEL = "vision_model"
    CODE_MODEL = "code_model"
    EMBEDDING_MODEL = "embedding_model"
    AUDIO_MODEL = "audio_model"
    SEARCH_API = "search_api"
    SPECIALIZED_API = "specialized_api"


class SignupStatus(Enum):
    """Status of signup process."""
    
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_VERIFICATION = "requires_verification"


@dataclass
class ProviderInfo:
    """Information about an API provider."""
    
    name: str
    display_name: str
    category: ProviderCategory
    description: str
    
    # URLs and links
    homepage_url: str
    signup_url: str
    docs_url: str
    pricing_url: Optional[str] = None
    
    # API details
    api_base_url: str
    auth_type: str = "api_key"  # "api_key", "oauth", "bearer_token"
    free_tier_available: bool = True
    
    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    
    # Limits and pricing
    free_tier_limits: Dict[str, Any] = field(default_factory=dict)
    pricing_model: str = "pay_per_use"  # "pay_per_use", "subscription", "freemium"
    
    # Integration details
    setup_complexity: str = "easy"  # "easy", "medium", "hard"
    setup_instructions: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SignupWorkflow:
    """Signup workflow tracking."""
    
    workflow_id: str
    provider_name: str
    user_id: Optional[str]
    
    # Status tracking
    status: SignupStatus
    current_step: int
    total_steps: int
    
    # Workflow data
    steps_completed: List[str]
    steps_remaining: List[str]
    collected_data: Dict[str, Any]
    
    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    # Results
    api_key: Optional[str] = None
    additional_credentials: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    notes: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
class S
ignupManager:
    """
    Automated signup link generation and key management system.
    
    Provides:
    - Comprehensive provider database with signup information
    - Automated signup workflow generation
    - Key management integration
    - Setup assistance and documentation
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_update_interval_hours: int = 24
    ) -> None:
        """
        Initialize signup manager.
        
        Args:
            storage_path: Path to SQLite database
            auto_update_interval_hours: Interval for auto-updating provider info
        """
        self.storage_path = storage_path or os.path.expanduser("~/.quantum_moe_mas/signup_management.db")
        self.auto_update_interval = auto_update_interval_hours
        
        # Provider database
        self.providers: Dict[str, ProviderInfo] = {}
        self.active_workflows: Dict[str, SignupWorkflow] = {}
        
        # Callbacks for workflow events
        self.workflow_callbacks: List[Callable[[SignupWorkflow, str], None]] = []
        
        # Background tasks
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None
        
        logger.info(
            "Initialized SignupManager",
            storage_path=self.storage_path,
            update_interval=auto_update_interval_hours
        )
    
    async def initialize(self) -> None:
        """Initialize the signup manager and database."""
        try:
            # Create storage directory
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Initialize database
            await self._initialize_database()
            
            # Load provider database
            await self._load_providers()
            
            # Load default providers if empty
            if not self.providers:
                await self._load_default_providers()
            
            # Start background updates
            self._update_task = asyncio.create_task(self._update_loop())
            
            self.is_running = True
            logger.info("SignupManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SignupManager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the signup manager."""
        self.is_running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SignupManager shutdown complete")
    
    async def _initialize_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            # Providers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS providers (
                    name TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    homepage_url TEXT NOT NULL,
                    signup_url TEXT NOT NULL,
                    docs_url TEXT NOT NULL,
                    pricing_url TEXT,
                    api_base_url TEXT NOT NULL,
                    auth_type TEXT NOT NULL,
                    free_tier_available BOOLEAN NOT NULL,
                    capabilities TEXT NOT NULL,
                    supported_formats TEXT NOT NULL,
                    free_tier_limits TEXT NOT NULL,
                    pricing_model TEXT NOT NULL,
                    setup_complexity TEXT NOT NULL,
                    setup_instructions TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Signup workflows table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signup_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL,
                    user_id TEXT,
                    status TEXT NOT NULL,
                    current_step INTEGER NOT NULL,
                    total_steps INTEGER NOT NULL,
                    steps_completed TEXT NOT NULL,
                    steps_remaining TEXT NOT NULL,
                    collected_data TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    last_activity TEXT NOT NULL,
                    api_key TEXT,
                    additional_credentials TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    error_messages TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_providers_category ON providers(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflows_provider ON signup_workflows(provider_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflows_status ON signup_workflows(status)")
            
            conn.commit()
            logger.info("Signup management database schema initialized")
            
        finally:
            conn.close()
    
    async def get_provider_info(self, provider_name: str) -> Optional[ProviderInfo]:
        """
        Get information about a specific provider.
        
        Args:
            provider_name: Name of the provider
        
        Returns:
            ProviderInfo instance or None if not found
        """
        return self.providers.get(provider_name)
    
    async def list_providers(
        self,
        category: Optional[ProviderCategory] = None,
        free_tier_only: bool = False,
        search_query: Optional[str] = None
    ) -> List[ProviderInfo]:
        """
        List available providers with optional filtering.
        
        Args:
            category: Optional category filter
            free_tier_only: Only include providers with free tiers
            search_query: Optional search query for name/description
        
        Returns:
            List of matching providers
        """
        providers = list(self.providers.values())
        
        # Apply filters
        if category:
            providers = [p for p in providers if p.category == category]
        
        if free_tier_only:
            providers = [p for p in providers if p.free_tier_available]
        
        if search_query:
            query_lower = search_query.lower()
            providers = [
                p for p in providers
                if (query_lower in p.name.lower() or 
                    query_lower in p.display_name.lower() or
                    query_lower in p.description.lower())
            ]
        
        # Sort by display name
        providers.sort(key=lambda p: p.display_name)
        
        return providers
    
    async def generate_signup_links(
        self,
        providers: Optional[List[str]] = None,
        include_setup_guide: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate signup links and setup information for providers.
        
        Args:
            providers: Optional list of specific providers
            include_setup_guide: Include setup instructions
        
        Returns:
            Dictionary with provider signup information
        """
        if providers is None:
            providers = list(self.providers.keys())
        
        signup_info = {}
        
        for provider_name in providers:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            
            info = {
                "display_name": provider.display_name,
                "category": provider.category.value,
                "signup_url": provider.signup_url,
                "free_tier_available": provider.free_tier_available,
                "setup_complexity": provider.setup_complexity,
            }
            
            if include_setup_guide:
                info.update({
                    "docs_url": provider.docs_url,
                    "setup_instructions": provider.setup_instructions,
                    "auth_type": provider.auth_type,
                    "api_base_url": provider.api_base_url,
                })
            
            signup_info[provider_name] = info
        
        return signup_info
    
    async def start_signup_workflow(
        self,
        provider_name: str,
        user_id: Optional[str] = None
    ) -> SignupWorkflow:
        """
        Start a guided signup workflow for a provider.
        
        Args:
            provider_name: Name of the provider
            user_id: Optional user identifier
        
        Returns:
            SignupWorkflow instance
        """
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not found")
        
        # Generate workflow steps
        steps = self._generate_workflow_steps(provider)
        
        workflow = SignupWorkflow(
            workflow_id=f"{provider_name}_{int(datetime.utcnow().timestamp())}",
            provider_name=provider_name,
            user_id=user_id,
            status=SignupStatus.IN_PROGRESS,
            current_step=0,
            total_steps=len(steps),
            steps_completed=[],
            steps_remaining=steps,
            collected_data={},
            started_at=datetime.utcnow()
        )
        
        # Save workflow
        await self._save_workflow(workflow)
        
        # Add to active workflows
        self.active_workflows[workflow.workflow_id] = workflow
        
        # Trigger callbacks
        await self._trigger_workflow_callbacks(workflow, "started")
        
        logger.info(f"Started signup workflow for {provider_name}: {workflow.workflow_id}")
        return workflow
    
    def _generate_workflow_steps(self, provider: ProviderInfo) -> List[str]:
        """Generate workflow steps for a provider."""
        steps = [
            f"Visit {provider.display_name} signup page",
            "Create account with email and password",
        ]
        
        if provider.auth_type == "api_key":
            steps.extend([
                "Navigate to API keys section",
                "Generate new API key",
                "Copy and securely store API key"
            ])
        elif provider.auth_type == "oauth":
            steps.extend([
                "Create OAuth application",
                "Configure redirect URLs",
                "Copy client ID and secret"
            ])
        
        steps.extend([
            "Test API connection",
            "Configure rate limits and quotas",
            "Complete integration setup"
        ])
        
        return steps
    
    async def _load_default_providers(self) -> None:
        """Load default provider configurations."""
        default_providers = [
            ProviderInfo(
                name="openai_playground",
                display_name="OpenAI Playground",
                category=ProviderCategory.LANGUAGE_MODEL,
                description="OpenAI's GPT models for text generation and chat",
                homepage_url="https://openai.com",
                signup_url="https://platform.openai.com/signup",
                docs_url="https://platform.openai.com/docs",
                pricing_url="https://openai.com/pricing",
                api_base_url="https://api.openai.com/v1",
                auth_type="api_key",
                free_tier_available=True,
                capabilities=["text_generation", "chat_completion", "function_calling"],
                supported_formats=["json", "text"],
                free_tier_limits={"requests_per_minute": 60, "tokens_per_minute": 10000},
                pricing_model="pay_per_use",
                setup_complexity="easy",
                setup_instructions=[
                    "Sign up at platform.openai.com",
                    "Navigate to API Keys section",
                    "Create new secret key",
                    "Set OPENAI_API_KEY environment variable"
                ]
            ),
            ProviderInfo(
                name="hugging_face",
                display_name="Hugging Face Inference API",
                category=ProviderCategory.LANGUAGE_MODEL,
                description="Access to thousands of models via Hugging Face",
                homepage_url="https://huggingface.co",
                signup_url="https://huggingface.co/join",
                docs_url="https://huggingface.co/docs/api-inference",
                api_base_url="https://api-inference.huggingface.co",
                auth_type="api_key",
                free_tier_available=True,
                capabilities=["text_generation", "text_classification", "embeddings"],
                supported_formats=["json", "text"],
                free_tier_limits={"requests_per_hour": 1000},
                pricing_model="freemium",
                setup_complexity="easy",
                setup_instructions=[
                    "Create account at huggingface.co",
                    "Go to Settings > Access Tokens",
                    "Create new token with 'read' permissions",
                    "Set HF_API_KEY environment variable"
                ]
            ),
            # Add more default providers...
        ]
        
        for provider in default_providers:
            await self._save_provider(provider)
            self.providers[provider.name] = provider
        
        logger.info(f"Loaded {len(default_providers)} default providers")
    
    async def _save_provider(self, provider: ProviderInfo) -> None:
        """Save provider to database."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO providers (
                    name, display_name, category, description,
                    homepage_url, signup_url, docs_url, pricing_url,
                    api_base_url, auth_type, free_tier_available,
                    capabilities, supported_formats, free_tier_limits,
                    pricing_model, setup_complexity, setup_instructions,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                provider.name, provider.display_name, provider.category.value, provider.description,
                provider.homepage_url, provider.signup_url, provider.docs_url, provider.pricing_url,
                provider.api_base_url, provider.auth_type, provider.free_tier_available,
                json.dumps(provider.capabilities), json.dumps(provider.supported_formats),
                json.dumps(provider.free_tier_limits), provider.pricing_model,
                provider.setup_complexity, json.dumps(provider.setup_instructions),
                provider.created_at.isoformat(), provider.updated_at.isoformat()
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _load_providers(self) -> None:
        """Load providers from database."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM providers")
            rows = cursor.fetchall()
            
            for row in rows:
                provider = ProviderInfo(
                    name=row[0],
                    display_name=row[1],
                    category=ProviderCategory(row[2]),
                    description=row[3],
                    homepage_url=row[4],
                    signup_url=row[5],
                    docs_url=row[6],
                    pricing_url=row[7],
                    api_base_url=row[8],
                    auth_type=row[9],
                    free_tier_available=bool(row[10]),
                    capabilities=json.loads(row[11]),
                    supported_formats=json.loads(row[12]),
                    free_tier_limits=json.loads(row[13]),
                    pricing_model=row[14],
                    setup_complexity=row[15],
                    setup_instructions=json.loads(row[16]),
                    created_at=datetime.fromisoformat(row[17]),
                    updated_at=datetime.fromisoformat(row[18])
                )
                
                self.providers[provider.name] = provider
            
            logger.info(f"Loaded {len(self.providers)} providers from database")
            
        finally:
            conn.close()
    
    async def _save_workflow(self, workflow: SignupWorkflow) -> None:
        """Save workflow to database."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO signup_workflows (
                    workflow_id, provider_name, user_id, status,
                    current_step, total_steps, steps_completed, steps_remaining,
                    collected_data, started_at, completed_at, last_activity,
                    api_key, additional_credentials, notes, error_messages
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow.workflow_id, workflow.provider_name, workflow.user_id, workflow.status.value,
                workflow.current_step, workflow.total_steps,
                json.dumps(workflow.steps_completed), json.dumps(workflow.steps_remaining),
                json.dumps(workflow.collected_data), workflow.started_at.isoformat(),
                workflow.completed_at.isoformat() if workflow.completed_at else None,
                workflow.last_activity.isoformat(), workflow.api_key,
                json.dumps(workflow.additional_credentials),
                json.dumps(workflow.notes), json.dumps(workflow.error_messages)
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _trigger_workflow_callbacks(self, workflow: SignupWorkflow, event: str) -> None:
        """Trigger workflow event callbacks."""
        for callback in self.workflow_callbacks:
            try:
                callback(workflow, event)
            except Exception as e:
                logger.error(f"Error in workflow callback: {e}")
    
    async def _update_loop(self) -> None:
        """Background loop for updating provider information."""
        while self.is_running:
            try:
                await asyncio.sleep(self.auto_update_interval * 3600)
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(3600)  # Wait before retrying
    
    def get_signup_manager_status(self) -> Dict[str, Any]:
        """
        Get comprehensive signup manager status.
        
        Returns:
            Dictionary with signup manager status
        """
        return {
            "is_running": self.is_running,
            "total_providers": len(self.providers),
            "active_workflows": len(self.active_workflows),
            "workflow_callbacks": len(self.workflow_callbacks),
            "storage_path": self.storage_path,
            "auto_update_interval_hours": self.auto_update_interval,
            "providers_by_category": {
                category.value: len([p for p in self.providers.values() if p.category == category])
                for category in ProviderCategory
            }
        }