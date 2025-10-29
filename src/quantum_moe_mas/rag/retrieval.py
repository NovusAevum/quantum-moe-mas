"""
Advanced retrieval engine with hybrid search strategies.

This module implements the core retrieval functionality with vector similarity,
keyword search, and hybrid approaches for optimal information retrieval.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from quantum_moe_mas.core.logging_simple import get_logger
from quantum_moe_mas.rag.document import Document, DocumentChunk, DocumentType
from quantum_moe_mas.rag.embeddings import MultiModalEmbeddings

logger = get_logger(__name__)


class SearchStrategy(str, Enum):
    """Available search strategies."""
    
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    SEMANTIC_GRAPH = "semantic_graph"


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""
    
    document: Document
    chunk: Optional[DocumentChunk] = None
    score: float
    search_strategy: SearchStrategy
    
    # Detailed scoring
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    graph_score: Optional[float] = None
    
    # Metadata
    retrieval_time: float
    rank: int
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval performance."""
    
    query: str
    total_documents: int
    retrieved_count: int
    search_strategy: SearchStrategy
    
    # Performance metrics
    retrieval_time: float
    vector_search_time: Optional[float] = None
    keyword_search_time: Optional[float] = None
    
    # Quality metrics
    avg_score: float
    max_score: float
    min_score: float
    
    # Strategy-specific metrics
    vector_hits: int = 0
    keyword_hits: int = 0
    hybrid_boost: float = 0.0
    
    timestamp: float = Field(default_factory=time.time)


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    
    # General settings
    max_results: int = 10
    min_score_threshold: float = 0.1
    
    # Vector search settings
    vector_weight: float = 0.7
    similarity_threshold: float = 0.3
    
    # Keyword search settings
    keyword_weight: float = 0.3
    fuzzy_matching: bool = True
    stemming: bool = True
    
    # Hybrid search settings
    hybrid_alpha: float = 0.5  # Balance between vector and keyword
    boost_exact_matches: float = 1.2
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    parallel_search: bool = True


class BaseSearchStrategy(ABC):
    """Abstract base class for search strategies."""
    
    def __init__(self, config: SearchConfig):
        self.config = config
    
    @abstractmethod
    async def search(
        self,
        query: str,
        documents: List[Document],
        embeddings: MultiModalEmbeddings,
        **kwargs
    ) -> List[RetrievalResult]:
        """Perform search and return results."""
        pass


class VectorSearchStrategy(BaseSearchStrategy):
    """Vector similarity search strategy."""
    
    async def search(
        self,
        query: str,
        documents: List[Document],
        embeddings: MultiModalEmbeddings,
        **kwargs
    ) -> List[RetrievalResult]:
        """Perform vector similarity search."""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await embeddings.embed_text(query)
        
        results = []
        
        # Search through documents
        for doc in documents:
            if not doc.has_embedding:
                continue
            
            # Calculate similarity with document embedding
            doc_similarity = await embeddings.compute_similarity(
                query_embedding, doc.embedding
            )
            
            if doc_similarity >= self.config.similarity_threshold:
                result = RetrievalResult(
                    document=doc,
                    score=doc_similarity,
                    vector_score=doc_similarity,
                    search_strategy=SearchStrategy.VECTOR_ONLY,
                    retrieval_time=time.time() - start_time,
                    rank=0  # Will be set later
                )
                results.append(result)
            
            # Also search through chunks if available
            if doc.has_chunks:
                for chunk in doc.chunks:
                    if chunk.embedding is not None:
                        chunk_similarity = await embeddings.compute_similarity(
                            query_embedding, chunk.embedding
                        )
                        
                        if chunk_similarity >= self.config.similarity_threshold:
                            result = RetrievalResult(
                                document=doc,
                                chunk=chunk,
                                score=chunk_similarity,
                                vector_score=chunk_similarity,
                                search_strategy=SearchStrategy.VECTOR_ONLY,
                                retrieval_time=time.time() - start_time,
                                rank=0
                            )
                            results.append(result)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:self.config.max_results]
        
        # Set ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results


class KeywordSearchStrategy(BaseSearchStrategy):
    """Keyword-based search strategy."""
    
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self._stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their'
        }
    
    async def search(
        self,
        query: str,
        documents: List[Document],
        embeddings: MultiModalEmbeddings,
        **kwargs
    ) -> List[RetrievalResult]:
        """Perform keyword-based search."""
        start_time = time.time()
        
        # Process query
        query_terms = self._process_query(query)
        
        results = []
        
        # Search through documents
        for doc in documents:
            score = self._calculate_keyword_score(query_terms, doc)
            
            if score >= self.config.min_score_threshold:
                result = RetrievalResult(
                    document=doc,
                    score=score,
                    keyword_score=score,
                    search_strategy=SearchStrategy.KEYWORD_ONLY,
                    retrieval_time=time.time() - start_time,
                    rank=0
                )
                results.append(result)
            
            # Search through chunks
            if doc.has_chunks:
                for chunk in doc.chunks:
                    chunk_score = self._calculate_keyword_score_chunk(query_terms, chunk)
                    
                    if chunk_score >= self.config.min_score_threshold:
                        result = RetrievalResult(
                            document=doc,
                            chunk=chunk,
                            score=chunk_score,
                            keyword_score=chunk_score,
                            search_strategy=SearchStrategy.KEYWORD_ONLY,
                            retrieval_time=time.time() - start_time,
                            rank=0
                        )
                        results.append(result)
        
        # Sort and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:self.config.max_results]
        
        # Set ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _process_query(self, query: str) -> List[str]:
        """Process query into search terms."""
        # Convert to lowercase and split
        terms = query.lower().split()
        
        # Remove stop words
        terms = [term for term in terms if term not in self._stop_words]
        
        # Remove punctuation
        terms = [term.strip('.,!?;:"()[]{}') for term in terms]
        
        # Filter empty terms
        terms = [term for term in terms if term]
        
        return terms
    
    def _calculate_keyword_score(self, query_terms: List[str], document: Document) -> float:
        """Calculate keyword match score for document."""
        if not query_terms:
            return 0.0
        
        content = document.content if isinstance(document.content, str) else str(document.content)
        content_lower = content.lower()
        
        # Count term matches
        matches = 0
        total_terms = len(query_terms)
        
        for term in query_terms:
            if term in content_lower:
                matches += 1
                
                # Boost for exact phrase matches
                if len(query_terms) > 1 and ' '.join(query_terms) in content_lower:
                    matches += 0.5
        
        # Calculate score
        base_score = matches / total_terms if total_terms > 0 else 0.0
        
        # Apply boosts
        if matches == total_terms:  # All terms found
            base_score *= self.config.boost_exact_matches
        
        return min(base_score, 1.0)
    
    def _calculate_keyword_score_chunk(self, query_terms: List[str], chunk: DocumentChunk) -> float:
        """Calculate keyword match score for chunk."""
        if not query_terms:
            return 0.0
        
        content_lower = chunk.content.lower()
        
        matches = 0
        total_terms = len(query_terms)
        
        for term in query_terms:
            if term in content_lower:
                matches += 1
        
        base_score = matches / total_terms if total_terms > 0 else 0.0
        
        # Boost for chunks with higher importance
        if chunk.importance_score:
            base_score *= (1 + chunk.importance_score * 0.2)
        
        return min(base_score, 1.0)


class HybridSearchStrategy(BaseSearchStrategy):
    """Hybrid search combining vector and keyword approaches."""
    
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.vector_strategy = VectorSearchStrategy(config)
        self.keyword_strategy = KeywordSearchStrategy(config)
    
    async def search(
        self,
        query: str,
        documents: List[Document],
        embeddings: MultiModalEmbeddings,
        **kwargs
    ) -> List[RetrievalResult]:
        """Perform hybrid search combining vector and keyword results."""
        start_time = time.time()
        
        # Perform both searches in parallel
        if self.config.parallel_search:
            vector_task = asyncio.create_task(
                self.vector_strategy.search(query, documents, embeddings)
            )
            keyword_task = asyncio.create_task(
                self.keyword_strategy.search(query, documents, embeddings)
            )
            
            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task
            )
        else:
            vector_results = await self.vector_strategy.search(query, documents, embeddings)
            keyword_results = await self.keyword_strategy.search(query, documents, embeddings)
        
        # Combine results
        combined_results = self._combine_results(
            vector_results, keyword_results, start_time
        )
        
        return combined_results
    
    def _combine_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        start_time: float
    ) -> List[RetrievalResult]:
        """Combine vector and keyword results using hybrid scoring."""
        # Create a map of document/chunk to results
        result_map: Dict[str, RetrievalResult] = {}
        
        # Process vector results
        for result in vector_results:
            key = self._get_result_key(result)
            result_map[key] = RetrievalResult(
                document=result.document,
                chunk=result.chunk,
                score=0.0,  # Will be calculated
                vector_score=result.score,
                keyword_score=0.0,
                search_strategy=SearchStrategy.HYBRID,
                retrieval_time=time.time() - start_time,
                rank=0
            )
        
        # Process keyword results
        for result in keyword_results:
            key = self._get_result_key(result)
            if key in result_map:
                # Update existing result
                result_map[key].keyword_score = result.score
            else:
                # Create new result
                result_map[key] = RetrievalResult(
                    document=result.document,
                    chunk=result.chunk,
                    score=0.0,
                    vector_score=0.0,
                    keyword_score=result.score,
                    search_strategy=SearchStrategy.HYBRID,
                    retrieval_time=time.time() - start_time,
                    rank=0
                )
        
        # Calculate hybrid scores
        for result in result_map.values():
            vector_score = result.vector_score or 0.0
            keyword_score = result.keyword_score or 0.0
            
            # Weighted combination
            hybrid_score = (
                self.config.vector_weight * vector_score +
                self.config.keyword_weight * keyword_score
            )
            
            # Apply hybrid boost if both strategies found the result
            if vector_score > 0 and keyword_score > 0:
                hybrid_score *= (1 + self.config.hybrid_alpha)
            
            result.score = hybrid_score
        
        # Sort and limit results
        results = list(result_map.values())
        results = [r for r in results if r.score >= self.config.min_score_threshold]
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:self.config.max_results]
        
        # Set ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _get_result_key(self, result: RetrievalResult) -> str:
        """Generate unique key for result."""
        if result.chunk:
            return f"{result.document.id}_{result.chunk.id}"
        else:
            return str(result.document.id)


class RetrievalEngine:
    """Main retrieval engine coordinating different search strategies."""
    
    def __init__(
        self,
        embeddings: MultiModalEmbeddings,
        config: Optional[SearchConfig] = None
    ):
        self.embeddings = embeddings
        self.config = config or SearchConfig()
        
        # Initialize search strategies
        self.strategies = {
            SearchStrategy.VECTOR_ONLY: VectorSearchStrategy(self.config),
            SearchStrategy.KEYWORD_ONLY: KeywordSearchStrategy(self.config),
            SearchStrategy.HYBRID: HybridSearchStrategy(self.config),
        }
        
        # Cache for search results
        self._cache: Dict[str, Tuple[List[RetrievalResult], float]] = {}
        
        logger.info("Retrieval engine initialized")
    
    async def search(
        self,
        query: str,
        documents: List[Document],
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        **kwargs
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Perform search using specified strategy."""
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(query, strategy, **kwargs)
        if self.config.enable_caching and cache_key in self._cache:
            cached_results, cache_time = self._cache[cache_key]
            if time.time() - cache_time < self.config.cache_ttl:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                
                # Update retrieval time for cached results
                for result in cached_results:
                    result.retrieval_time = time.time() - start_time
                
                metrics = self._calculate_metrics(query, documents, cached_results, strategy, start_time)
                return cached_results, metrics
        
        # Perform search
        if strategy not in self.strategies:
            raise ValueError(f"Unknown search strategy: {strategy}")
        
        results = await self.strategies[strategy].search(
            query, documents, self.embeddings, **kwargs
        )
        
        # Cache results
        if self.config.enable_caching:
            self._cache[cache_key] = (results, time.time())
        
        # Calculate metrics
        metrics = self._calculate_metrics(query, documents, results, strategy, start_time)
        
        logger.info(
            f"Search completed: {len(results)} results in {metrics.retrieval_time:.3f}s "
            f"(strategy: {strategy.value})"
        )
        
        return results, metrics
    
    async def adaptive_search(
        self,
        query: str,
        documents: List[Document],
        **kwargs
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Perform adaptive search that selects optimal strategy."""
        # Simple heuristics for strategy selection
        query_length = len(query.split())
        
        if query_length <= 2:
            # Short queries work better with keyword search
            strategy = SearchStrategy.KEYWORD_ONLY
        elif query_length >= 10:
            # Long queries work better with vector search
            strategy = SearchStrategy.VECTOR_ONLY
        else:
            # Medium queries work best with hybrid
            strategy = SearchStrategy.HYBRID
        
        logger.debug(f"Adaptive search selected strategy: {strategy.value}")
        
        return await self.search(query, documents, strategy, **kwargs)
    
    def _get_cache_key(self, query: str, strategy: SearchStrategy, **kwargs) -> str:
        """Generate cache key for search."""
        import hashlib
        key_data = f"{query}_{strategy.value}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_metrics(
        self,
        query: str,
        documents: List[Document],
        results: List[RetrievalResult],
        strategy: SearchStrategy,
        start_time: float
    ) -> RetrievalMetrics:
        """Calculate retrieval metrics."""
        retrieval_time = time.time() - start_time
        
        if not results:
            return RetrievalMetrics(
                query=query,
                total_documents=len(documents),
                retrieved_count=0,
                search_strategy=strategy,
                retrieval_time=retrieval_time,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0
            )
        
        scores = [r.score for r in results]
        vector_hits = sum(1 for r in results if r.vector_score and r.vector_score > 0)
        keyword_hits = sum(1 for r in results if r.keyword_score and r.keyword_score > 0)
        
        return RetrievalMetrics(
            query=query,
            total_documents=len(documents),
            retrieved_count=len(results),
            search_strategy=strategy,
            retrieval_time=retrieval_time,
            avg_score=sum(scores) / len(scores),
            max_score=max(scores),
            min_score=min(scores),
            vector_hits=vector_hits,
            keyword_hits=keyword_hits,
            hybrid_boost=vector_hits * keyword_hits / len(results) if results else 0.0
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self.config.enable_caching,
            "cache_ttl": self.config.cache_ttl,
        }
    
    def clear_cache(self) -> None:
        """Clear search cache."""
        self._cache.clear()
        logger.info("Search cache cleared")