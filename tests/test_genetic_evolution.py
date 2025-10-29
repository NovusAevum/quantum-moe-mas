"""
Tests for the Genetic Evolution Engine.

This module tests the genetic algorithm implementation for prompt optimization,
including mutation, crossover, selection, and A/B testing functionality.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from quantum_moe_mas.orchestration.genetic_evolution import (
    GeneticEvolutionEngine,
    PromptVariant,
    PerformanceMetrics,
    OptimizationObjective,
    MutationType,
    ABTestResult,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""
    
    def test_combined_score_default_weights(self):
        """Test combined score calculation with default weights."""
        metrics = PerformanceMetrics(
            accuracy=0.8,
            speed=2.0,  # 2 seconds
            cost=0.1,   # $0.10
            user_satisfaction=0.9
        )
        
        score = metrics.combined_score()
        
        # Should be weighted combination with speed/cost normalized
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be good score given high accuracy/satisfaction
    
    def test_combined_score_custom_weights(self):
        """Test combined score with custom weights."""
        metrics = PerformanceMetrics(accuracy=0.8, speed=1.0, cost=0.05, user_satisfaction=0.7)
        
        custom_weights = {"accuracy": 0.6, "speed": 0.2, "cost": 0.1, "user_satisfaction": 0.1}
        score = metrics.combined_score(custom_weights)
        
        assert 0.0 <= score <= 1.0


class TestPromptVariant:
    """Test PromptVariant functionality."""
    
    def test_prompt_variant_creation(self):
        """Test creating a prompt variant."""
        variant = PromptVariant(
            content="Test prompt for analysis",
            generation=1,
            mutations=[MutationType.WORD_SUBSTITUTION]
        )
        
        assert variant.content == "Test prompt for analysis"
        assert variant.generation == 1
        assert len(variant.mutations) == 1
        assert variant.id is not None
    
    def test_average_performance_empty(self):
        """Test average performance with no history."""
        variant = PromptVariant(content="Test")
        avg = variant.average_performance
        
        assert avg.accuracy == 0.0
        assert avg.speed == 0.0
        assert avg.cost == 0.0
    
    def test_average_performance_with_history(self):
        """Test average performance calculation."""
        variant = PromptVariant(content="Test")
        
        # Add performance history
        variant.performance_history = [
            PerformanceMetrics(accuracy=0.8, speed=1.0, cost=0.1),
            PerformanceMetrics(accuracy=0.9, speed=1.5, cost=0.15),
        ]
        
        avg = variant.average_performance
        assert avg.accuracy == 0.85
        assert avg.speed == 1.25
        assert avg.cost == 0.125


class TestGeneticEvolutionEngine:
    """Test GeneticEvolutionEngine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create a genetic evolution engine for testing."""
        return GeneticEvolutionEngine(
            population_size=10,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elite_size=2,
            max_generations=5
        )
    
    @pytest.fixture
    def mock_evaluation_function(self):
        """Create a mock evaluation function."""
        def evaluate(prompt: str) -> PerformanceMetrics:
            # Simple evaluation based on prompt length and content
            score = min(1.0, len(prompt) / 100.0)
            return PerformanceMetrics(
                accuracy=score,
                speed=1.0,
                cost=0.1,
                user_satisfaction=score * 0.9
            )
        return evaluate
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.population_size == 10
        assert engine.mutation_rate == 0.2
        assert engine.crossover_rate == 0.8
        assert engine.elite_size == 2
        assert engine.generation == 0
        assert len(engine.population) == 0
    
    def test_initialize_population(self, engine):
        """Test population initialization."""
        base_prompt = "Analyze the given data and provide insights."
        
        engine.initialize_population(base_prompt, "analytics")
        
        assert len(engine.population) == engine.population_size
        assert engine.population[0].content == base_prompt  # Original should be first
        assert all(variant.generation == 0 for variant in engine.population)
    
    def test_mutation_word_substitution(self, engine):
        """Test word substitution mutation."""
        original = "Please analyze the data carefully."
        mutated = engine._mutate_word_substitution(original)
        
        # Should either be unchanged or have substitutions
        assert isinstance(mutated, str)
        assert len(mutated) > 0
    
    def test_mutation_phrase_insertion(self, engine):
        """Test phrase insertion mutation."""
        original = "Analyze the data."
        mutated = engine._mutate_phrase_insertion(original)
        
        # Should be longer than original
        assert len(mutated) >= len(original)
        assert "Analyze the data" in mutated
    
    def test_evaluate_variant(self, engine, mock_evaluation_function):
        """Test variant evaluation."""
        variant = PromptVariant(content="Test prompt for evaluation")
        
        metrics = engine.evaluate_variant(variant, mock_evaluation_function)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert len(variant.performance_history) == 1
        assert variant.fitness_score > 0
    
    def test_tournament_selection(self, engine):
        """Test tournament selection."""
        # Create population with known fitness scores
        engine.population = [
            PromptVariant(content="Low fitness", fitness_score=0.2),
            PromptVariant(content="Medium fitness", fitness_score=0.5),
            PromptVariant(content="High fitness", fitness_score=0.9),
        ]
        
        # Run multiple selections to test probabilistic nature
        selections = [engine._tournament_selection() for _ in range(10)]
        
        # High fitness variant should be selected more often
        high_fitness_selections = sum(1 for s in selections if s.fitness_score == 0.9)
        assert high_fitness_selections > 0
    
    def test_crossover(self, engine):
        """Test crossover operation."""
        parent1 = PromptVariant(content="First parent prompt with multiple sentences. This is sentence two.")
        parent2 = PromptVariant(content="Second parent prompt with different content. This has different structure.")
        
        child1, child2 = engine._crossover(parent1, parent2)
        
        assert isinstance(child1, PromptVariant)
        assert isinstance(child2, PromptVariant)
        assert child1.content != parent1.content or child2.content != parent2.content
        assert parent1.id in child1.parent_ids
        assert parent2.id in child1.parent_ids
    
    def test_mutation(self, engine):
        """Test mutation operation."""
        original = PromptVariant(content="Original prompt content")
        mutated = engine._mutate(original)
        
        assert isinstance(mutated, PromptVariant)
        assert mutated.content != original.content or len(mutated.mutations) > len(original.mutations)
        assert original.id in mutated.parent_ids
    
    @pytest.mark.asyncio
    async def test_evolve_generation(self, engine, mock_evaluation_function):
        """Test generation evolution."""
        base_prompt = "Analyze the data and provide insights."
        engine.initialize_population(base_prompt)
        
        new_population = await engine.evolve_generation(mock_evaluation_function)
        
        assert len(new_population) == engine.population_size
        assert engine.generation == 1
        assert all(variant.fitness_score >= 0 for variant in new_population)
    
    def test_ab_test_creation(self, engine):
        """Test A/B test creation."""
        variant_a = PromptVariant(content="Variant A prompt")
        variant_b = PromptVariant(content="Variant B prompt")
        
        test_id = engine.start_ab_test(variant_a, variant_b, sample_size=50)
        
        assert test_id in engine.active_ab_tests
        test = engine.active_ab_tests[test_id]
        assert test.variant_a == variant_a
        assert test.variant_b == variant_b
        assert test.sample_size == 50
    
    def test_ab_test_update_and_completion(self, engine):
        """Test A/B test update and completion."""
        variant_a = PromptVariant(content="Variant A")
        variant_b = PromptVariant(content="Variant B")
        
        test_id = engine.start_ab_test(variant_a, variant_b, sample_size=2)
        
        # Add test results
        for _ in range(2):
            engine.update_ab_test(test_id, variant_a.id, PerformanceMetrics(accuracy=0.8))
            engine.update_ab_test(test_id, variant_b.id, PerformanceMetrics(accuracy=0.6))
        
        # Test should be completed and removed from active tests
        assert test_id not in engine.active_ab_tests
    
    def test_get_best_variant(self, engine):
        """Test getting the best variant."""
        engine.population = [
            PromptVariant(content="Low", fitness_score=0.3),
            PromptVariant(content="High", fitness_score=0.9),
            PromptVariant(content="Medium", fitness_score=0.6),
        ]
        
        best = engine.get_best_variant()
        assert best.fitness_score == 0.9
        assert best.content == "High"
    
    def test_evolution_summary(self, engine):
        """Test evolution summary generation."""
        # Add some evolution history
        engine.evolution_history = [
            {"generation": 0, "best_fitness": 0.5, "average_fitness": 0.4, "diversity": 0.8},
            {"generation": 1, "best_fitness": 0.7, "average_fitness": 0.6, "diversity": 0.7},
        ]
        engine.generation = 1
        engine.population = [PromptVariant(content="Best", fitness_score=0.7)]
        
        summary = engine.get_evolution_summary()
        
        assert summary["current_generation"] == 1
        assert summary["total_generations"] == 2
        assert summary["best_fitness"] == 0.7
        assert "best_variant_id" in summary
    
    @pytest.mark.asyncio
    async def test_optimize_prompts_integration(self, engine, mock_evaluation_function):
        """Test the complete optimization process."""
        base_prompt = "Analyze the data."
        
        best_variant = await engine.optimize_prompts(
            base_prompt, 
            mock_evaluation_function,
            domain="analytics",
            target_generations=2
        )
        
        assert isinstance(best_variant, PromptVariant)
        assert best_variant.fitness_score > 0
        assert engine.generation >= 1
        assert len(engine.evolution_history) >= 1


class TestABTestResult:
    """Test ABTestResult functionality."""
    
    def test_ab_test_result_creation(self):
        """Test creating an A/B test result."""
        variant_a = PromptVariant(content="A")
        variant_b = PromptVariant(content="B")
        
        result = ABTestResult(
            variant_a=variant_a,
            variant_b=variant_b,
            sample_size=100,
            confidence_level=0.95
        )
        
        assert result.variant_a == variant_a
        assert result.variant_b == variant_b
        assert result.sample_size == 100
        assert result.confidence_level == 0.95
        assert result.test_id is not None


if __name__ == "__main__":
    pytest.main([__file__])