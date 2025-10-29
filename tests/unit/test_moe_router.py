"""
Unit tests for Quantum MoE Router.

Tests the core routing functionality, quantum-inspired gating,
and expert selection logic.
"""

import pytest
import asyncio
from datetime import datetime

from quantum_moe_mas.moe.router import QuantumMoERouter, QuantumState
from quantum_moe_mas.moe.expert import Expert, ExpertType, ExpertStatus


class TestQuantumState:
    """Test quantum state creation and properties."""
    
    def test_from_scores_creates_valid_state(self) -> None:
        """Test quantum state creation from scores."""
        scores = {"expert1": 80.0, "expert2": 60.0, "expert3": 40.0}
        state = QuantumState.from_scores(scores)
        
        assert len(state.amplitudes) == 3
        assert len(state.probabilities) == 3
        assert state.entropy > 0
        
        # Probabilities should sum to 1
        prob_sum = sum(state.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.001
    
    def test_from_scores_handles_zero_total(self) -> None:
        """Test quantum state with zero total scores."""
        scores = {"expert1": 0.0, "expert2": 0.0}
        state = QuantumState.from_scores(scores)
        
        # Should create uniform distribution
        assert all(p == 0.5 for p in state.probabilities.values())


class TestQuantumMoERouter:
    """Test Quantum MoE Router functionality."""
    
    @pytest.fixture
    def router(self) -> QuantumMoERouter:
        """Create a router instance for testing."""
        return QuantumMoERouter(
            confidence_threshold=80.0,
            top_k=2,
            sparse_activation_ratio=0.055
        )
    
    @pytest.fixture
    def sample_experts(self) -> list[Expert]:
        """Create sample experts for testing."""
        return [
            Expert(
                id="expert1",
                name="Claude Sonnet 4",
                type=ExpertType.LANGUAGE_MODEL,
                api_endpoint="https://api.anthropic.com",
                api_key_env_var="ANTHROPIC_API_KEY",
                capabilities=["general", "reasoning", "coding"],
                cost_per_token=0.00001,
                max_tokens=8192,
                confidence_score=90.0,
                priority=80
            ),
            Expert(
                id="expert2",
                name="Qwen3 Coder Plus",
                type=ExpertType.CODE_MODEL,
                api_endpoint="https://api.qwen.com",
                api_key_env_var="QWEN_API_KEY",
                capabilities=["coding", "debugging"],
                cost_per_token=0.000005,
                max_tokens=4096,
                confidence_score=85.0,
                priority=70
            ),
            Expert(
                id="expert3",
                name="DeepSeek V3",
                type=ExpertType.REASONING_MODEL,
                api_endpoint="https://api.deepseek.com",
                api_key_env_var="DEEPSEEK_API_KEY",
                capabilities=["reasoning", "analysis"],
                cost_per_token=0.000008,
                max_tokens=4096,
                confidence_score=88.0,
                priority=75
            ),
        ]
    
    def test_router_initialization(self) -> None:
        """Test router initialization with valid parameters."""
        router = QuantumMoERouter(
            confidence_threshold=75.0,
            top_k=3,
            sparse_activation_ratio=0.1
        )
        
        assert router.confidence_threshold == 75.0
        assert router.top_k == 3
        assert router.sparse_activation_ratio == 0.1
        assert len(router.experts) == 0
    
    def test_router_initialization_invalid_confidence(self) -> None:
        """Test router initialization with invalid confidence threshold."""
        with pytest.raises(ValueError, match="Confidence threshold must be between 0 and 100"):
            QuantumMoERouter(confidence_threshold=150.0)
    
    def test_router_initialization_invalid_top_k(self) -> None:
        """Test router initialization with invalid top_k."""
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            QuantumMoERouter(top_k=0)
    
    def test_add_expert(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test adding experts to router."""
        expert = sample_experts[0]
        result = router.add_expert(expert)
        
        assert result is True
        assert expert.id in router.experts
        assert router.experts[expert.id] == expert
    
    def test_add_duplicate_expert(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test adding duplicate expert raises error."""
        expert = sample_experts[0]
        router.add_expert(expert)
        
        with pytest.raises(ValueError, match="Expert with ID .* already exists"):
            router.add_expert(expert)
    
    def test_remove_expert(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test removing expert from router."""
        expert = sample_experts[0]
        router.add_expert(expert)
        
        result = router.remove_expert(expert.id)
        
        assert result is True
        assert expert.id not in router.experts
    
    def test_remove_nonexistent_expert(self, router: QuantumMoERouter) -> None:
        """Test removing non-existent expert."""
        result = router.remove_expert("nonexistent")
        assert result is False
    
    def test_get_expert(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test getting expert by ID."""
        expert = sample_experts[0]
        router.add_expert(expert)
        
        retrieved = router.get_expert(expert.id)
        assert retrieved == expert
    
    def test_list_experts(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test listing all experts."""
        for expert in sample_experts:
            router.add_expert(expert)
        
        experts = router.list_experts()
        assert len(experts) == len(sample_experts)
    
    @pytest.mark.asyncio
    async def test_route_basic(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test basic routing functionality."""
        for expert in sample_experts:
            router.add_expert(expert)
        
        selected = await router.route(
            query="Write a Python function to sort a list",
            domain="coding"
        )
        
        assert len(selected) <= router.top_k
        assert all(isinstance(e, Expert) for e in selected)
        assert all(e.is_available() for e in selected)
    
    @pytest.mark.asyncio
    async def test_route_no_experts(self, router: QuantumMoERouter) -> None:
        """Test routing with no available experts."""
        with pytest.raises(ValueError, match="No available experts for routing"):
            await router.route("test query", "general")
    
    @pytest.mark.asyncio
    async def test_route_domain_matching(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test routing prefers experts with matching domain."""
        for expert in sample_experts:
            router.add_expert(expert)
        
        selected = await router.route(
            query="Debug this code",
            domain="coding"
        )
        
        # Should prefer coding experts
        coding_experts = [e for e in selected if "coding" in e.capabilities]
        assert len(coding_experts) > 0
    
    @pytest.mark.asyncio
    async def test_route_updates_metrics(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test routing updates metrics."""
        for expert in sample_experts:
            router.add_expert(expert)
        
        initial_requests = router.metrics.total_requests
        
        await router.route("test query", "general")
        
        assert router.metrics.total_requests == initial_requests + 1
        assert router.metrics.successful_routes > 0
    
    @pytest.mark.asyncio
    async def test_route_updates_expert_load(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test routing updates expert load factors."""
        for expert in sample_experts:
            router.add_expert(expert)
        
        initial_loads = {e.id: e.load_factor for e in sample_experts}
        
        selected = await router.route("test query", "general")
        
        # Selected experts should have increased load
        for expert in selected:
            assert expert.load_factor > initial_loads[expert.id]
    
    def test_calculate_expert_scores(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test expert score calculation."""
        scores = router._calculate_expert_scores(
            query="Write code",
            domain="coding",
            available_experts=sample_experts
        )
        
        assert len(scores) == len(sample_experts)
        assert all(score >= 0 for score in scores.values())
        
        # Coding expert should have higher score
        coding_expert = next(e for e in sample_experts if "coding" in e.capabilities)
        assert scores[coding_expert.id] > 0
    
    def test_apply_quantum_gating(self, router: QuantumMoERouter) -> None:
        """Test quantum-inspired gating."""
        scores = {"expert1": 80.0, "expert2": 60.0, "expert3": 40.0}
        
        quantum_state, probabilities = router._apply_quantum_gating(scores)
        
        assert isinstance(quantum_state, QuantumState)
        assert len(probabilities) == len(scores)
        
        # Probabilities should sum to approximately 1
        prob_sum = sum(probabilities.values())
        assert abs(prob_sum - 1.0) < 0.001
    
    def test_select_top_k_experts(self, router: QuantumMoERouter) -> None:
        """Test top-k expert selection."""
        probabilities = {
            "expert1": 0.5,
            "expert2": 0.3,
            "expert3": 0.15,
            "expert4": 0.05
        }
        
        selected = router._select_top_k_experts(probabilities, k=2)
        
        assert len(selected) == 2
        assert "expert1" in selected
        assert "expert2" in selected
    
    def test_calculate_routing_confidence(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test routing confidence calculation."""
        probabilities = {"expert1": 0.6, "expert2": 0.4}
        
        confidence = router._calculate_routing_confidence(
            sample_experts[:2],
            probabilities
        )
        
        assert 0 <= confidence <= 100
    
    def test_update_expert_weights(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test updating expert weights based on feedback."""
        for expert in sample_experts:
            router.add_expert(expert)
        
        initial_confidence = sample_experts[0].confidence_score
        
        feedback = {sample_experts[0].id: 95.0}
        router.update_expert_weights(feedback)
        
        updated_expert = router.get_expert(sample_experts[0].id)
        assert updated_expert is not None
        assert updated_expert.confidence_score != initial_confidence
    
    def test_get_routing_metrics(self, router: QuantumMoERouter) -> None:
        """Test getting routing metrics."""
        metrics = router.get_routing_metrics()
        
        assert metrics.total_requests >= 0
        assert metrics.successful_routes >= 0
        assert metrics.failed_routes >= 0
    
    def test_reset_metrics(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test resetting routing metrics."""
        for expert in sample_experts:
            router.add_expert(expert)
        
        # Simulate some routing
        router.metrics.total_requests = 10
        router.metrics.successful_routes = 8
        
        router.reset_metrics()
        
        assert router.metrics.total_requests == 0
        assert router.metrics.successful_routes == 0
    
    def test_get_expert_pool_status(self, router: QuantumMoERouter, sample_experts: list[Expert]) -> None:
        """Test getting expert pool status."""
        for expert in sample_experts:
            router.add_expert(expert)
        
        status = router.get_expert_pool_status()
        
        assert status["total_experts"] == len(sample_experts)
        assert status["available_experts"] > 0
        assert "status_distribution" in status
        assert "average_load" in status
        assert "average_confidence" in status
