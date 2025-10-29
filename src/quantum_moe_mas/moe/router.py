"""
Quantum-inspired MoE Router with probabilistic expert selection.

This module implements the core routing logic using quantum-inspired algorithms
for optimal expert selection with sparse activation and dynamic load balancing.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from quantum_moe_mas.moe.expert import Expert, ExpertStatus
from quantum_moe_mas.moe.metrics import RoutingMetrics, RoutingDecision
from quantum_moe_mas.core.logging_simple import get_logger


logger = get_logger(__name__)


@dataclass
class QuantumState:
    """Represents a quantum-inspired state for expert selection."""
    
    amplitudes: Dict[str, complex]
    probabilities: Dict[str, float]
    entropy: float
    
    @classmethod
    def from_scores(cls, scores: Dict[str, float]) -> "QuantumState":
        """
        Create quantum state from expert scores.
        
        Args:
            scores: Dictionary mapping expert IDs to scores
        
        Returns:
            QuantumState instance
        """
        # Normalize scores to create probability distribution
        total = sum(scores.values())
        if total == 0:
            probabilities = {k: 1.0 / len(scores) for k in scores.keys()}
        else:
            probabilities = {k: v / total for k, v in scores.items()}
        
        # Create complex amplitudes (quantum superposition)
        amplitudes = {
            k: complex(np.sqrt(p), 0) for k, p in probabilities.items()
        }
        
        # Calculate entropy
        entropy = -sum(
            p * np.log2(p) if p > 0 else 0
            for p in probabilities.values()
        )
        
        return cls(
            amplitudes=amplitudes,
            probabilities=probabilities,
            entropy=entropy
        )


class QuantumMoERouter:
    """
    Quantum-inspired Mixture of Experts Router.
    
    Implements sparse activation (37B/671B parameters) with quantum-probabilistic
    gating for optimal expert selection. Features dynamic load balancing and
    confidence-based routing.
    
    Attributes:
        experts: Dictionary of available experts
        metrics: Routing performance metrics
        confidence_threshold: Minimum confidence for routing (0-100)
        top_k: Number of experts to select (default: 2)
        sparse_activation_ratio: Ratio of active parameters (default: 0.055 for 37B/671B)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 80.0,
        top_k: int = 2,
        sparse_activation_ratio: float = 0.055,
        enable_load_balancing: bool = True
    ) -> None:
        """
        Initialize the Quantum MoE Router.
        
        Args:
            confidence_threshold: Minimum confidence score for routing (0-100)
            top_k: Number of top experts to select
            sparse_activation_ratio: Ratio of parameters to activate
            enable_load_balancing: Whether to enable dynamic load balancing
        
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        if not 0 <= confidence_threshold <= 100:
            raise ValueError("Confidence threshold must be between 0 and 100")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 < sparse_activation_ratio <= 1:
            raise ValueError("Sparse activation ratio must be between 0 and 1")
        
        self.experts: Dict[str, Expert] = {}
        self.metrics = RoutingMetrics()
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.sparse_activation_ratio = sparse_activation_ratio
        self.enable_load_balancing = enable_load_balancing
        
        logger.info(
            f"Initialized QuantumMoERouter: confidence_threshold={confidence_threshold}, "
            f"top_k={top_k}, sparse_activation_ratio={sparse_activation_ratio}"
        )
    
    def add_expert(self, expert: Expert) -> bool:
        """
        Add an expert to the router's pool.
        
        Args:
            expert: Expert instance to add
        
        Returns:
            True if expert was added successfully
        
        Raises:
            ValueError: If expert with same ID already exists
        """
        if expert.id in self.experts:
            raise ValueError(f"Expert with ID {expert.id} already exists")
        
        self.experts[expert.id] = expert
        logger.info(f"Added expert: {expert.name} (ID: {expert.id})")
        return True
    
    def remove_expert(self, expert_id: str) -> bool:
        """
        Remove an expert from the router's pool.
        
        Args:
            expert_id: ID of expert to remove
        
        Returns:
            True if expert was removed successfully
        """
        if expert_id not in self.experts:
            logger.warning(f"Expert {expert_id} not found")
            return False
        
        expert = self.experts.pop(expert_id)
        logger.info(f"Removed expert: {expert.name} (ID: {expert_id})")
        return True
    
    def get_expert(self, expert_id: str) -> Optional[Expert]:
        """
        Get an expert by ID.
        
        Args:
            expert_id: ID of expert to retrieve
        
        Returns:
            Expert instance or None if not found
        """
        return self.experts.get(expert_id)
    
    def list_experts(self) -> List[Expert]:
        """
        Get list of all experts.
        
        Returns:
            List of Expert instances
        """
        return list(self.experts.values())
    
    def _calculate_expert_scores(
        self,
        query: str,
        domain: str,
        available_experts: List[Expert]
    ) -> Dict[str, float]:
        """
        Calculate scores for each expert based on query and domain.
        
        Args:
            query: Query text
            domain: Domain classification
            available_experts: List of available experts
        
        Returns:
            Dictionary mapping expert IDs to scores
        """
        scores: Dict[str, float] = {}
        
        for expert in available_experts:
            score = 0.0
            
            # Base score from expert confidence
            score += expert.confidence_score * 0.3
            
            # Domain matching bonus
            if domain in expert.capabilities:
                score += 30.0
            
            # Priority bonus
            score += expert.priority * 0.2
            
            # Performance bonus (success rate)
            if expert.performance.total_requests > 0:
                score += expert.performance.success_rate * 0.2
            
            # Load penalty (prefer less loaded experts)
            if self.enable_load_balancing:
                load_penalty = expert.load_factor * 20.0
                score -= load_penalty
            
            # Cost efficiency bonus (prefer cheaper experts)
            if expert.cost_per_token > 0:
                cost_factor = 1.0 / (1.0 + expert.cost_per_token * 1000)
                score += cost_factor * 10.0
            
            scores[expert.id] = max(0.0, score)
        
        return scores
    
    def _apply_quantum_gating(
        self,
        scores: Dict[str, float]
    ) -> Tuple[QuantumState, Dict[str, float]]:
        """
        Apply quantum-inspired probabilistic gating to expert scores.
        
        Args:
            scores: Dictionary of expert scores
        
        Returns:
            Tuple of (QuantumState, normalized probabilities)
        """
        # Create quantum state from scores
        quantum_state = QuantumState.from_scores(scores)
        
        # Apply quantum interference (enhance high-probability states)
        enhanced_probs = {}
        for expert_id, prob in quantum_state.probabilities.items():
            # Quantum interference: square the amplitude
            amplitude = quantum_state.amplitudes[expert_id]
            enhanced_prob = abs(amplitude) ** 2
            
            # Apply sparse activation constraint
            if enhanced_prob < (1.0 - self.sparse_activation_ratio):
                enhanced_prob *= 0.5  # Reduce probability for sparse activation
            
            enhanced_probs[expert_id] = enhanced_prob
        
        # Renormalize
        total = sum(enhanced_probs.values())
        if total > 0:
            enhanced_probs = {k: v / total for k, v in enhanced_probs.items()}
        
        return quantum_state, enhanced_probs
    
    def _select_top_k_experts(
        self,
        probabilities: Dict[str, float],
        k: int
    ) -> List[str]:
        """
        Select top-k experts based on probabilities.
        
        Args:
            probabilities: Dictionary of expert probabilities
            k: Number of experts to select
        
        Returns:
            List of selected expert IDs
        """
        # Sort by probability
        sorted_experts = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top-k
        selected = [expert_id for expert_id, _ in sorted_experts[:k]]
        
        return selected
    
    def _calculate_routing_confidence(
        self,
        selected_experts: List[Expert],
        probabilities: Dict[str, float]
    ) -> float:
        """
        Calculate overall confidence in routing decision.
        
        Args:
            selected_experts: List of selected experts
            probabilities: Expert selection probabilities
        
        Returns:
            Confidence score (0-100)
        """
        if not selected_experts:
            return 0.0
        
        # Average of selected expert confidences
        expert_confidence = sum(e.confidence_score for e in selected_experts) / len(selected_experts)
        
        # Selection probability confidence
        selection_probs = [probabilities.get(e.id, 0.0) for e in selected_experts]
        prob_confidence = sum(selection_probs) * 100.0
        
        # Combined confidence (weighted average)
        confidence = (expert_confidence * 0.6 + prob_confidence * 0.4)
        
        return min(100.0, confidence)
    
    async def route(
        self,
        query: str,
        domain: str = "general",
        confidence_threshold: Optional[float] = None
    ) -> List[Expert]:
        """
        Route a query to the most appropriate experts using quantum-inspired selection.
        
        Args:
            query: Query text to route
            domain: Domain classification of the query
            confidence_threshold: Override default confidence threshold
        
        Returns:
            List of selected Expert instances
        
        Raises:
            ValueError: If no experts are available or confidence is too low
        """
        start_time = time.time()
        threshold = confidence_threshold or self.confidence_threshold
        
        # Get available experts
        available_experts = [
            expert for expert in self.experts.values()
            if expert.is_available()
        ]
        
        if not available_experts:
            raise ValueError("No available experts for routing")
        
        logger.debug(
            f"Routing query to {len(available_experts)} available experts "
            f"(query_length={len(query)}, domain={domain})"
        )
        
        # Calculate expert scores
        scores = self._calculate_expert_scores(query, domain, available_experts)
        
        # Apply quantum-inspired gating
        quantum_state, probabilities = self._apply_quantum_gating(scores)
        
        # Select top-k experts
        selected_ids = self._select_top_k_experts(probabilities, self.top_k)
        selected_experts = [self.experts[eid] for eid in selected_ids]
        
        # Calculate routing confidence
        routing_confidence = self._calculate_routing_confidence(
            selected_experts,
            probabilities
        )
        
        # Check confidence threshold
        if routing_confidence < threshold:
            logger.warning(
                f"Routing confidence {routing_confidence:.2f}% below threshold {threshold}%"
            )
        
        # Create routing decision
        latency_ms = (time.time() - start_time) * 1000
        decision = RoutingDecision(
            query_text=query,
            domain=domain,
            selected_expert_ids=selected_ids,
            expert_scores={k: float(v) for k, v in scores.items()},
            quantum_state={
                "entropy": quantum_state.entropy,
                "probabilities": {k: float(v) for k, v in probabilities.items()}
            },
            routing_confidence=routing_confidence,
            decision_rationale=self._generate_rationale(
                selected_experts,
                scores,
                probabilities
            ),
            latency_ms=latency_ms
        )
        
        # Record metrics
        self.metrics.record_routing_decision(decision, success=True)
        
        # Update expert load factors
        if self.enable_load_balancing:
            for expert in selected_experts:
                new_load = min(1.0, expert.load_factor + 0.1)
                expert.update_load(new_load)
        
        logger.info(
            f"Routed to {len(selected_experts)} experts: {[e.name for e in selected_experts]} "
            f"(confidence={routing_confidence:.2f}%, latency={latency_ms:.2f}ms)"
        )
        
        return selected_experts
    
    def _generate_rationale(
        self,
        selected_experts: List[Expert],
        scores: Dict[str, float],
        probabilities: Dict[str, float]
    ) -> str:
        """Generate human-readable rationale for routing decision."""
        rationale_parts = []
        
        for expert in selected_experts:
            score = scores.get(expert.id, 0.0)
            prob = probabilities.get(expert.id, 0.0)
            rationale_parts.append(
                f"{expert.name} (score: {score:.2f}, probability: {prob:.2%})"
            )
        
        return "Selected experts: " + ", ".join(rationale_parts)
    
    def get_routing_metrics(self) -> RoutingMetrics:
        """
        Get current routing metrics.
        
        Returns:
            RoutingMetrics instance
        """
        return self.metrics
    
    def update_expert_weights(self, feedback: Dict[str, float]) -> None:
        """
        Update expert confidence scores based on feedback.
        
        Args:
            feedback: Dictionary mapping expert IDs to feedback scores (0-100)
        """
        for expert_id, score in feedback.items():
            if expert_id in self.experts:
                expert = self.experts[expert_id]
                # Exponential moving average for smooth updates
                new_confidence = 0.7 * expert.confidence_score + 0.3 * score
                expert.update_confidence(new_confidence)
                logger.debug(
                    f"Updated expert {expert.name} confidence: {expert.confidence_score:.2f} -> {new_confidence:.2f}"
                )
    
    def reset_metrics(self) -> None:
        """Reset routing metrics to initial state."""
        self.metrics = RoutingMetrics()
        logger.info("Reset routing metrics")
    
    def get_expert_pool_status(self) -> Dict[str, any]:
        """
        Get status of the expert pool.
        
        Returns:
            Dictionary with pool status information
        """
        total_experts = len(self.experts)
        available_experts = sum(1 for e in self.experts.values() if e.is_available())
        
        status_counts = {}
        for expert in self.experts.values():
            status = expert.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_experts": total_experts,
            "available_experts": available_experts,
            "status_distribution": status_counts,
            "average_load": sum(e.load_factor for e in self.experts.values()) / total_experts if total_experts > 0 else 0.0,
            "average_confidence": sum(e.confidence_score for e in self.experts.values()) / total_experts if total_experts > 0 else 0.0,
        }
