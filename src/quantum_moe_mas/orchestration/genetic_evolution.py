"""
Genetic Evolution Engine for Prompt Optimization.

This module implements a sophisticated genetic algorithm for optimizing
agent prompts based on performance feedback, A/B testing, and multi-objective
optimization targeting accuracy, speed, and cost efficiency.

Requirements addressed: 4.5, 9.1, 9.5

Author: Wan Mohamad Hanis bin Wan Hassan
"""
import json
import os
import asyncio
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from quantum_moe_mas.core.exceptions import QuantumMoEMASError
from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class OptimizationObjective(str, Enum):
    """Optimization objectives for genetic evolution."""
    ACCURACY = "accuracy"
    SPEED = "speed"
    COST = "cost"
    COMBINED = "combined"


class MutationType(str, Enum):
    """Types of mutations for prompt evolution."""
    WORD_SUBSTITUTION = "word_substitution"
    PHRASE_INSERTION = "phrase_insertion"
    SENTENCE_REORDER = "sentence_reorder"
    STYLE_MODIFICATION = "style_modification"
    CONTEXT_ENHANCEMENT = "context_enhancement"


@dataclass
class PerformanceMetrics:
    """Performance metrics for prompt evaluation."""
    accuracy: float = 0.0  # 0-1 scale
    speed: float = 0.0  # Response time in seconds
    cost: float = 0.0  # Cost per query in USD
    user_satisfaction: float = 0.0  # 0-1 scale
    roi: float = 0.0  # Return on investment
    timestamp: datetime = field(default_factory=datetime.now)
    
    def combined_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted combined score."""
        if weights is None:
            weights = {"accuracy": 0.4, "speed": 0.3, "cost": 0.2, "user_satisfaction": 0.1}
        
        # Normalize speed and cost (lower is better)
        normalized_speed = max(0, 1 - (self.speed / 10.0))  # Assume 10s is worst case
        normalized_cost = max(0, 1 - (self.cost / 1.0))  # Assume $1 is worst case
        
        return (
            self.accuracy * weights.get("accuracy", 0.4) +
            normalized_speed * weights.get("speed", 0.3) +
            normalized_cost * weights.get("cost", 0.2) +
            self.user_satisfaction * weights.get("user_satisfaction", 0.1)
        )


@dataclass
class PromptVariant:
    """A variant of a prompt for genetic evolution."""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations: List[MutationType] = field(default_factory=list)
    performance_history: List[PerformanceMetrics] = field(default_factory=list)
    fitness_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def average_performance(self) -> PerformanceMetrics:
        """Calculate average performance across all evaluations."""
        if not self.performance_history:
            return PerformanceMetrics()
        
        return PerformanceMetrics(
            accuracy=statistics.mean(m.accuracy for m in self.performance_history),
            speed=statistics.mean(m.speed for m in self.performance_history),
            cost=statistics.mean(m.cost for m in self.performance_history),
            user_satisfaction=statistics.mean(m.user_satisfaction for m in self.performance_history),
            roi=statistics.mean(m.roi for m in self.performance_history),
        )


@dataclass
class ABTestResult:
    """Results from A/B testing prompt variants."""
    test_id: str = field(default_factory=lambda: str(uuid4()))
    variant_a: PromptVariant = None
    variant_b: PromptVariant = None
    sample_size: int = 0
    confidence_level: float = 0.95
    statistical_significance: bool = False
    winner: Optional[str] = None  # variant_a or variant_b ID
    effect_size: float = 0.0
    p_value: float = 1.0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class GeneticEvolutionEngine:
    """
    Genetic Evolution Engine for optimizing agent prompts.
    
    This engine uses genetic algorithms to evolve prompts based on
    performance feedback, implementing mutation, crossover, and selection
    operations to optimize for multiple objectives.
    """
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        max_generations: int = 100,
        convergence_threshold: float = 0.001,
        objectives_weights: Dict[str, float] = None
    ):
        """
        Initialize the Genetic Evolution Engine.
        
        Args:
            population_size: Number of prompt variants in each generation
            mutation_rate: Probability of mutation for each variant
            crossover_rate: Probability of crossover between variants
            elite_size: Number of top performers to preserve each generation
            max_generations: Maximum number of generations to evolve
            convergence_threshold: Threshold for convergence detection
            objectives_weights: Weights for multi-objective optimization
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        
        self.objectives_weights = objectives_weights or {
            "accuracy": 0.4,
            "speed": 0.3, 
            "cost": 0.2,
            "user_satisfaction": 0.1
        }
        
        self.population: List[PromptVariant] = []
        self.generation = 0
        self.evolution_history: List[Dict[str, Any]] = []
        self.active_ab_tests: Dict[str, ABTestResult] = {}
        
        # Mutation templates and strategies
        self.mutation_templates = {
            MutationType.WORD_SUBSTITUTION: [
                ("analyze", "examine"),
                ("create", "generate"),
                ("provide", "deliver"),
                ("ensure", "guarantee"),
                ("optimize", "enhance"),
            ],
            MutationType.PHRASE_INSERTION: [
                "Please",
                "Carefully",
                "Step by step",
                "In detail",
                "Systematically",
            ],
            MutationType.STYLE_MODIFICATION: [
                "Be concise and direct.",
                "Use a professional tone.",
                "Explain your reasoning.",
                "Provide examples where helpful.",
                "Focus on actionable insights.",
            ]
        }
        
        logger.info(f"Initialized GeneticEvolutionEngine with population_size={population_size}")
    
    def initialize_population(self, base_prompt: str, domain: str = "general") -> None:
        """
        Initialize the population with variants of the base prompt.
        
        Args:
            base_prompt: The initial prompt to evolve
            domain: Domain context for prompt optimization
        """
        logger.info(f"Initializing population with base prompt for domain: {domain}")
        
        self.population = []
        
        # Add the original prompt
        original = PromptVariant(
            content=base_prompt,
            generation=0,
            fitness_score=0.5  # Neutral starting fitness
        )
        self.population.append(original)
        
        # Generate variants through mutations
        for i in range(self.population_size - 1):
            variant = self._create_initial_variant(base_prompt, i + 1)
            self.population.append(variant)
        
        logger.info(f"Created initial population of {len(self.population)} variants")
    
    def _create_initial_variant(self, base_prompt: str, variant_id: int) -> PromptVariant:
        """Create an initial variant through random mutations."""
        content = base_prompt
        mutations = []
        
        # Apply 1-3 random mutations
        num_mutations = random.randint(1, 3)
        for _ in range(num_mutations):
            mutation_type = random.choice(list(MutationType))
            content = self._apply_mutation(content, mutation_type)
            mutations.append(mutation_type)
        
        return PromptVariant(
            content=content,
            generation=0,
            mutations=mutations,
            fitness_score=0.5
        )
    
    def _apply_mutation(self, prompt: str, mutation_type: MutationType) -> str:
        """Apply a specific type of mutation to a prompt."""
        try:
            if mutation_type == MutationType.WORD_SUBSTITUTION:
                return self._mutate_word_substitution(prompt)
            elif mutation_type == MutationType.PHRASE_INSERTION:
                return self._mutate_phrase_insertion(prompt)
            elif mutation_type == MutationType.SENTENCE_REORDER:
                return self._mutate_sentence_reorder(prompt)
            elif mutation_type == MutationType.STYLE_MODIFICATION:
                return self._mutate_style_modification(prompt)
            elif mutation_type == MutationType.CONTEXT_ENHANCEMENT:
                return self._mutate_context_enhancement(prompt)
            else:
                return prompt
        except Exception as e:
            logger.warning(f"Mutation failed for type {mutation_type}: {e}")
            return prompt
    
    def _mutate_word_substitution(self, prompt: str) -> str:
        """Replace words with synonyms."""
        for old_word, new_word in self.mutation_templates[MutationType.WORD_SUBSTITUTION]:
            if old_word in prompt.lower():
                # Replace with 50% probability
                if random.random() < 0.5:
                    prompt = prompt.replace(old_word, new_word)
                    break
        return prompt
    
    def _mutate_phrase_insertion(self, prompt: str) -> str:
        """Insert helpful phrases."""
        phrases = self.mutation_templates[MutationType.PHRASE_INSERTION]
        phrase = random.choice(phrases)
        
        # Insert at beginning with 70% probability, otherwise at end
        if random.random() < 0.7:
            return f"{phrase} {prompt}"
        else:
            return f"{prompt} {phrase}"
    
    def _mutate_sentence_reorder(self, prompt: str) -> str:
        """Reorder sentences in the prompt."""
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        if len(sentences) > 1:
            random.shuffle(sentences)
            return '. '.join(sentences) + '.'
        return prompt
    
    def _mutate_style_modification(self, prompt: str) -> str:
        """Add style instructions."""
        styles = self.mutation_templates[MutationType.STYLE_MODIFICATION]
        style = random.choice(styles)
        return f"{prompt} {style}"
    
    def _mutate_context_enhancement(self, prompt: str) -> str:
        """Enhance with context-specific instructions."""
        enhancements = [
            "Consider the business context.",
            "Think about security implications.",
            "Focus on user experience.",
            "Optimize for performance.",
            "Ensure scalability.",
        ]
        enhancement = random.choice(enhancements)
        return f"{prompt} {enhancement}"
    
    def evaluate_variant(
        self, 
        variant: PromptVariant, 
        evaluation_function: Callable[[str], PerformanceMetrics]
    ) -> PerformanceMetrics:
        """
        Evaluate a prompt variant using the provided evaluation function.
        
        Args:
            variant: The prompt variant to evaluate
            evaluation_function: Function that takes a prompt and returns metrics
            
        Returns:
            Performance metrics for the variant
        """
        try:
            metrics = evaluation_function(variant.content)
            variant.performance_history.append(metrics)
            
            # Update fitness score based on combined performance
            variant.fitness_score = metrics.combined_score(self.objectives_weights)
            
            logger.debug(f"Evaluated variant {variant.id[:8]}: fitness={variant.fitness_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate variant {variant.id}: {e}")
            # Return poor performance metrics on failure
            poor_metrics = PerformanceMetrics(accuracy=0.1, speed=10.0, cost=1.0)
            variant.performance_history.append(poor_metrics)
            variant.fitness_score = 0.1
            return poor_metrics
    
    async def evolve_generation(
        self, 
        evaluation_function: Callable[[str], PerformanceMetrics]
    ) -> List[PromptVariant]:
        """
        Evolve the population by one generation.
        
        Args:
            evaluation_function: Function to evaluate prompt performance
            
        Returns:
            The new generation of prompt variants
        """
        logger.info(f"Evolving generation {self.generation}")
        
        # Evaluate current population
        for variant in self.population:
            if not variant.performance_history:  # Only evaluate if not already evaluated
                self.evaluate_variant(variant, evaluation_function)
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda v: v.fitness_score, reverse=True)
        
        # Create new generation
        new_population = []
        
        # Preserve elite
        elite = self.population[:self.elite_size]
        new_population.extend(elite)
        logger.debug(f"Preserved {len(elite)} elite variants")
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            # Selection for crossover
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        
        # Update generation info
        for variant in new_population:
            if variant not in elite:  # Don't update generation for elite
                variant.generation = self.generation + 1
        
        self.population = new_population
        self.generation += 1
        
        # Record evolution history
        self._record_generation_stats()
        
        logger.info(f"Generation {self.generation} complete. Best fitness: {self.population[0].fitness_score:.3f}")
        return self.population
    
    def _tournament_selection(self, tournament_size: int = 3) -> PromptVariant:
        """Select a parent using tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda v: v.fitness_score)
    
    def _crossover(self, parent1: PromptVariant, parent2: PromptVariant) -> Tuple[PromptVariant, PromptVariant]:
        """Create offspring through crossover of two parents."""
        # Simple sentence-level crossover
        sentences1 = [s.strip() for s in parent1.content.split('.') if s.strip()]
        sentences2 = [s.strip() for s in parent2.content.split('.') if s.strip()]
        
        if len(sentences1) > 1 and len(sentences2) > 1:
            # Random crossover point
            crossover_point1 = random.randint(1, len(sentences1) - 1)
            crossover_point2 = random.randint(1, len(sentences2) - 1)
            
            child1_content = '. '.join(
                sentences1[:crossover_point1] + sentences2[crossover_point2:]
            ) + '.'
            child2_content = '. '.join(
                sentences2[:crossover_point2] + sentences1[crossover_point1:]
            ) + '.'
        else:
            # Fallback to word-level crossover
            words1 = parent1.content.split()
            words2 = parent2.content.split()
            
            crossover_point = random.randint(1, min(len(words1), len(words2)) - 1)
            
            child1_content = ' '.join(words1[:crossover_point] + words2[crossover_point:])
            child2_content = ' '.join(words2[:crossover_point] + words1[crossover_point:])
        
        child1 = PromptVariant(
            content=child1_content,
            parent_ids=[parent1.id, parent2.id],
            generation=self.generation + 1
        )
        
        child2 = PromptVariant(
            content=child2_content,
            parent_ids=[parent1.id, parent2.id],
            generation=self.generation + 1
        )
        
        return child1, child2
    
    def _mutate(self, variant: PromptVariant) -> PromptVariant:
        """Apply mutation to a variant."""
        mutation_type = random.choice(list(MutationType))
        mutated_content = self._apply_mutation(variant.content, mutation_type)
        
        mutated_variant = PromptVariant(
            content=mutated_content,
            parent_ids=[variant.id],
            mutations=variant.mutations + [mutation_type],
            generation=variant.generation
        )
        
        return mutated_variant
    
    def _record_generation_stats(self) -> None:
        """Record statistics for the current generation."""
        fitness_scores = [v.fitness_score for v in self.population]
        
        stats = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(self.population),
            "best_fitness": max(fitness_scores),
            "average_fitness": statistics.mean(fitness_scores),
            "worst_fitness": min(fitness_scores),
            "fitness_std": statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0,
            "diversity": self._calculate_diversity(),
        }
        
        self.evolution_history.append(stats)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity based on prompt content similarity."""
        if len(self.population) < 2:
            return 0.0
        
        # Simple diversity measure based on unique content
        unique_contents = set(v.content for v in self.population)
        return len(unique_contents) / len(self.population)
    
    def start_ab_test(
        self,
        variant_a: PromptVariant,
        variant_b: PromptVariant,
        sample_size: int = 100,
        confidence_level: float = 0.95
    ) -> str:
        """
        Start an A/B test between two prompt variants.
        
        Args:
            variant_a: First variant to test
            variant_b: Second variant to test
            sample_size: Number of samples per variant
            confidence_level: Statistical confidence level
            
        Returns:
            Test ID for tracking the A/B test
        """
        test = ABTestResult(
            variant_a=variant_a,
            variant_b=variant_b,
            sample_size=sample_size,
            confidence_level=confidence_level
        )
        
        self.active_ab_tests[test.test_id] = test
        
        logger.info(f"Started A/B test {test.test_id[:8]} between variants {variant_a.id[:8]} and {variant_b.id[:8]}")
        return test.test_id
    
    def update_ab_test(
        self,
        test_id: str,
        variant_id: str,
        metrics: PerformanceMetrics
    ) -> None:
        """
        Update A/B test with new performance data.
        
        Args:
            test_id: ID of the A/B test
            variant_id: ID of the variant that was tested
            metrics: Performance metrics from the test
        """
        if test_id not in self.active_ab_tests:
            logger.warning(f"A/B test {test_id} not found")
            return
        
        test = self.active_ab_tests[test_id]
        
        # Add metrics to the appropriate variant
        if test.variant_a.id == variant_id:
            test.variant_a.performance_history.append(metrics)
        elif test.variant_b.id == variant_id:
            test.variant_b.performance_history.append(metrics)
        else:
            logger.warning(f"Variant {variant_id} not part of test {test_id}")
            return
        
        # Check if test is complete
        a_samples = len(test.variant_a.performance_history)
        b_samples = len(test.variant_b.performance_history)
        
        if a_samples >= test.sample_size and b_samples >= test.sample_size:
            self._complete_ab_test(test_id)
    
    def _complete_ab_test(self, test_id: str) -> ABTestResult:
        """Complete an A/B test and determine the winner."""
        test = self.active_ab_tests[test_id]
        
        # Calculate average performance for each variant
        a_performance = test.variant_a.average_performance
        b_performance = test.variant_b.average_performance
        
        # Simple statistical test (t-test approximation)
        a_scores = [m.combined_score(self.objectives_weights) for m in test.variant_a.performance_history]
        b_scores = [m.combined_score(self.objectives_weights) for m in test.variant_b.performance_history]
        
        a_mean = statistics.mean(a_scores)
        b_mean = statistics.mean(b_scores)
        
        # Calculate effect size and statistical significance
        pooled_std = statistics.stdev(a_scores + b_scores)
        test.effect_size = abs(a_mean - b_mean) / pooled_std if pooled_std > 0 else 0
        
        # Simple significance test (assuming normal distribution)
        test.statistical_significance = test.effect_size > 0.2  # Cohen's d threshold
        test.p_value = 0.05 if test.statistical_significance else 0.5  # Simplified
        
        # Determine winner
        if test.statistical_significance:
            test.winner = test.variant_a.id if a_mean > b_mean else test.variant_b.id
        
        test.completed_at = datetime.now()
        
        logger.info(f"Completed A/B test {test_id[:8]}. Winner: {test.winner[:8] if test.winner else 'No significant difference'}")
        
        # Remove from active tests
        del self.active_ab_tests[test_id]
        
        return test
    
    def get_best_variant(self) -> Optional[PromptVariant]:
        """Get the best performing variant from the current population."""
        if not self.population:
            return None
        
        return max(self.population, key=lambda v: v.fitness_score)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the evolution process."""
        if not self.evolution_history:
            return {}
        
        latest_stats = self.evolution_history[-1]
        best_variant = self.get_best_variant()
        
        return {
            "current_generation": self.generation,
            "total_generations": len(self.evolution_history),
            "best_fitness": latest_stats["best_fitness"],
            "average_fitness": latest_stats["average_fitness"],
            "population_diversity": latest_stats["diversity"],
            "best_variant_id": best_variant.id if best_variant else None,
            "convergence_progress": self._calculate_convergence_progress(),
            "active_ab_tests": len(self.active_ab_tests),
        }
    
    def _calculate_convergence_progress(self) -> float:
        """Calculate how close the population is to convergence."""
        if len(self.evolution_history) < 5:
            return 0.0
        
        # Look at fitness improvement over last 5 generations
        recent_best = [gen["best_fitness"] for gen in self.evolution_history[-5:]]
        improvement = recent_best[-1] - recent_best[0]
        
        # Normalize improvement (0 = no improvement, 1 = significant improvement)
        return min(1.0, max(0.0, improvement / self.convergence_threshold))
    
    async def optimize_prompts(
        self,
        base_prompt: str,
        evaluation_function: Callable[[str], PerformanceMetrics],
        domain: str = "general",
        target_generations: int = None
    ) -> PromptVariant:
        """
        Run the complete optimization process.
        
        Args:
            base_prompt: Initial prompt to optimize
            evaluation_function: Function to evaluate prompt performance
            domain: Domain context for optimization
            target_generations: Number of generations to run (default: max_generations)
            
        Returns:
            The best optimized prompt variant
        """
        target_generations = target_generations or self.max_generations
        
        logger.info(f"Starting prompt optimization for domain: {domain}")
        
        # Initialize population
        self.initialize_population(base_prompt, domain)
        
        # Evolution loop
        for generation in range(target_generations):
            await self.evolve_generation(evaluation_function)
            
            # Check for convergence
            if self._check_convergence():
                logger.info(f"Converged after {generation + 1} generations")
                break
        
        best_variant = self.get_best_variant()
        logger.info(f"Optimization complete. Best fitness: {best_variant.fitness_score:.3f}")
        
        return best_variant
    
    def _check_convergence(self) -> bool:
        """Check if the population has converged."""
        if len(self.evolution_history) < 10:
            return False
        
        # Check if fitness has plateaued
        recent_best = [gen["best_fitness"] for gen in self.evolution_history[-10:]]
        improvement = recent_best[-1] - recent_best[0]
        
        return improvement < self.convergence_threshold