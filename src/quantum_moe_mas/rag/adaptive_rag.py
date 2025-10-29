"""
Adaptive Multi-Modal RAG system with hybrid vector-graph search.

This module implements the main AdaptiveMultiModalRAG class that coordinates
document processing, embedding generation, and intelligent retrieval.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import numpy as np

from quantum_moe_mas.core.logging_simple import get_logger
from quantum_moe_mas.rag.document import Document, DocumentChunk, DocumentType, ProcessingStatus
from quantum_moe_mas.rag.embeddings import EmbeddingConfig, MultiModalEmbeddings
from quantum_moe_mas.rag.retrieval import (
    RetrievalEngine,
    RetrievalMetrics,
    RetrievalResult,
    SearchConfig,
    SearchStrategy,
)
from quantum_moe_mas.rag.database import (
    SupabaseConnection,
    VectorOperations,
    BatchOperations,
    SearchOperations,
    DocumentRecord,
    ChunkRecord,
    DatabaseSchema,
)
from quantum_moe_mas.rag.retrieval_cache import RetrievalCache
from quantum_moe_mas.rag.metrics_collector import MetricsCollector, MetricType

logger = get_logger(__name__)


class AdaptiveRAGConfig:
    """Configuration for the Adaptive RAG system."""
    
    def __init__(
        self,
        embedding_config: Optional[EmbeddingConfig] = None,
        search_config: Optional[SearchConfig] = None,
        chunking_strategy: str = "semantic",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunks_per_document: int = 100,
        enable_adaptive_learning: bool = True,
        confidence_threshold: float = 0.7,
        fallback_threshold: float = 0.3,
        max_concurrent_processing: int = 10,
        # Caching configuration
        enable_caching: bool = True,
        cache_max_entries: int = 1000,
        cache_max_size_mb: int = 100,
        cache_ttl_hours: int = 1,
        # Metrics configuration
        enable_metrics: bool = True,
        metrics_history_hours: int = 24,
        # Database configuration
        use_database: bool = True,
        database_connection: Optional[SupabaseConnection] = None,
        batch_size: int = 100,
        enable_schema_init: bool = True,
    ):
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.search_config = search_config or SearchConfig()
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunks_per_document = max_chunks_per_document
        self.enable_adaptive_learning = enable_adaptive_learning
        self.confidence_threshold = confidence_threshold
        self.fallback_threshold = fallback_threshold
        self.max_concurrent_processing = max_concurrent_processing
        # Database settings
        self.use_database = use_database
        self.database_connection = database_connection
        self.batch_size = batch_size
        self.enable_schema_init = enable_schema_init
        # Caching settings
        self.enable_caching = enable_caching
        self.cache_max_entries = cache_max_entries
        self.cache_max_size_mb = cache_max_size_mb
        self.cache_ttl_hours = cache_ttl_hours
        # Metrics settings
        self.enable_metrics = enable_metrics
        self.metrics_history_hours = metrics_history_hours


class AdaptiveMultiModalRAG:
    """
    Adaptive Multi-Modal RAG system with hybrid vector-graph search.
    
    This class provides the main interface for the RAG system, coordinating
    document processing, embedding generation, and intelligent retrieval.
    """
    
    def __init__(self, config: Optional[AdaptiveRAGConfig] = None):
        self.config = config or AdaptiveRAGConfig()
        
        # Initialize core components
        self.embeddings = MultiModalEmbeddings(self.config.embedding_config)
        self.retrieval_engine = RetrievalEngine(self.embeddings, self.config.search_config)
        
        # Database components
        self.db_connection: Optional[SupabaseConnection] = None
        self.vector_ops: Optional[VectorOperations] = None
        self.batch_ops: Optional[BatchOperations] = None
        self.search_ops: Optional[SearchOperations] = None
        self.db_schema: Optional[DatabaseSchema] = None
        
        # Document storage (fallback to in-memory if database disabled)
        self.documents: Dict[UUID, Document] = {}
        self.document_index: Dict[str, UUID] = {}  # content_hash -> document_id
        
        # Performance tracking
        self.processing_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "avg_retrieval_time": 0.0,
        }
        
        # Adaptive learning data
        self.query_feedback: Dict[str, List[float]] = {}  # query -> [relevance_scores]
        self.strategy_performance: Dict[SearchStrategy, List[float]] = {
            strategy: [] for strategy in SearchStrategy
        }
        
        # Initialize caching layer
        self.cache: Optional[RetrievalCache] = None
        if self.config.enable_caching:
            self.cache = RetrievalCache(
                max_entries=self.config.cache_max_entries,
                max_size_mb=self.config.cache_max_size_mb,
                query_ttl=self.config.cache_ttl_hours * 3600,
                embedding_ttl=self.config.cache_ttl_hours * 3600 * 24,  # 24x longer for embeddings
            )
        
        # Initialize metrics collector
        self.metrics_collector: Optional[MetricsCollector] = None
        if self.config.enable_metrics:
            self.metrics_collector = MetricsCollector(
                max_history_hours=self.config.metrics_history_hours,
                aggregation_window_minutes=5,
                enable_real_time_alerts=True,
            )
        
        logger.info("Adaptive Multi-Modal RAG system initialized with caching and metrics")
    
    async def initialize_database(self) -> None:
        """Initialize database connection and schema."""
        if not self.config.use_database:
            logger.info("Database disabled, using in-memory storage")
            return
        
        try:
            # Initialize connection
            if self.config.database_connection:
                self.db_connection = self.config.database_connection
            else:
                self.db_connection = SupabaseConnection()
                await self.db_connection.connect()
            
            # Initialize database operations
            self.vector_ops = VectorOperations(self.db_connection)
            self.batch_ops = BatchOperations(self.db_connection, self.config.batch_size)
            self.search_ops = SearchOperations(self.db_connection)
            self.db_schema = DatabaseSchema(self.db_connection)
            
            # Initialize schema if enabled
            if self.config.enable_schema_init:
                await self.db_schema.initialize_schema()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            logger.warning("Falling back to in-memory storage")
            self.config.use_database = False
        
        # Start metrics collector if enabled
        if self.metrics_collector:
            await self.metrics_collector.start()
    
    async def close_database(self) -> None:
        """Close database connection and cleanup resources."""
        # Stop metrics collector
        if self.metrics_collector:
            await self.metrics_collector.stop()
        
        # Close database connection
        if self.db_connection:
            await self.db_connection.disconnect()
            logger.info("Database connection closed")
    
    async def add_document(
        self,
        content: Union[str, bytes, Path],
        document_type: Optional[DocumentType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        process_immediately: bool = True,
    ) -> Document:
        """
        Add a document to the RAG system.
        
        Args:
            content: Document content (text, bytes, or file path)
            document_type: Type of document (auto-detected if None)
            metadata: Additional metadata for the document
            process_immediately: Whether to process the document immediately
            
        Returns:
            Document object
        """
        start_time = time.time()
        
        try:
            # Create document
            if isinstance(content, Path):
                document = Document.from_file(content, document_type)
            else:
                document = Document(
                    content=content,
                    document_type=document_type or self._detect_document_type(content),
                )
            
            # Add custom metadata
            if metadata:
                for key, value in metadata.items():
                    document.metadata.custom_fields[key] = value
            
            # Check for duplicates
            content_hash = document.content_hash
            if content_hash in self.document_index:
                existing_id = self.document_index[content_hash]
                logger.info(f"Document already exists: {existing_id}")
                if self.config.use_database and self.search_ops:
                    existing_doc = await self.search_ops.get_document_by_id(existing_id)
                    if existing_doc:
                        return self._document_record_to_document(existing_doc)
                return self.documents.get(existing_id, document)
            
            # Store document
            if self.config.use_database and self.vector_ops:
                await self._store_document_in_database(document)
            else:
                # Fallback to in-memory storage
                self.documents[document.id] = document
                self.document_index[content_hash] = document.id
            
            # Process document if requested
            if process_immediately:
                await self.process_document(document.id)
            
            processing_time = time.time() - start_time
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            
            logger.info(
                f"Document added: {document.id} "
                f"(type: {document.document_type.value}, "
                f"size: {len(str(document.content))}, "
                f"time: {processing_time:.3f}s)"
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    async def add_documents(
        self,
        documents: List[Union[str, bytes, Path, Document]],
        process_immediately: bool = True,
    ) -> List[Document]:
        """Add multiple documents to the RAG system."""
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.config.max_concurrent_processing)
        
        async def add_single_document(doc_input):
            async with semaphore:
                if isinstance(doc_input, Document):
                    # Document already created
                    self.documents[doc_input.id] = doc_input
                    if process_immediately:
                        await self.process_document(doc_input.id)
                    return doc_input
                else:
                    return await self.add_document(doc_input, process_immediately=process_immediately)
        
        # Process documents concurrently
        tasks = [add_single_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_docs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to add document {i}: {result}")
            else:
                successful_docs.append(result)
        
        logger.info(f"Added {len(successful_docs)}/{len(documents)} documents successfully")
        return successful_docs
    
    async def process_document(self, document_id: UUID) -> None:
        """Process a document: chunk it and generate embeddings."""
        if document_id not in self.documents:
            raise ValueError(f"Document not found: {document_id}")
        
        document = self.documents[document_id]
        
        if document.status == ProcessingStatus.PROCESSING:
            logger.warning(f"Document {document_id} is already being processed")
            return
        
        if document.status == ProcessingStatus.COMPLETED:
            logger.debug(f"Document {document_id} already processed")
            return
        
        start_time = time.time()
        document.update_status(ProcessingStatus.PROCESSING)
        
        try:
            # Generate document embedding
            document.embedding = await self.embeddings.embed_document(document)
            
            # Chunk document
            chunks = await self._chunk_document(document)
            
            # Generate chunk embeddings
            chunk_embeddings = await self._generate_chunk_embeddings(chunks)
            
            # Update chunks with embeddings
            for chunk, embedding in zip(chunks, chunk_embeddings):
                chunk.embedding = embedding
                document.chunks.append(chunk)
            
            # Store chunks in database if enabled
            if self.config.use_database and chunks:
                await self._store_chunks_in_database(document)
            
            # Update status
            document.update_status(ProcessingStatus.COMPLETED)
            
            # Update stats
            processing_time = time.time() - start_time
            self.processing_stats["chunks_created"] += len(chunks)
            self.processing_stats["embeddings_generated"] += len(chunks) + 1  # +1 for document embedding
            
            logger.info(
                f"Document processed: {document_id} "
                f"({len(chunks)} chunks, {processing_time:.3f}s)"
            )
            
        except Exception as e:
            document.update_status(ProcessingStatus.FAILED, str(e))
            logger.error(f"Failed to process document {document_id}: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        strategy: Optional[SearchStrategy] = None,
        document_types: Optional[List[DocumentType]] = None,
        min_confidence: Optional[float] = None,
        **kwargs
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Retrieve relevant documents/chunks for a query with caching and fallback strategies.
        
        Args:
            query: Search query
            k: Number of results to return
            strategy: Search strategy (auto-selected if None)
            document_types: Filter by document types
            min_confidence: Minimum confidence threshold
            **kwargs: Additional search parameters
            
        Returns:
            Tuple of (results, metrics)
        """
        start_time = time.time()
        cache_hit = False
        fallback_triggered = False
        
        # Check cache first if enabled
        if self.cache:
            filters = {"document_types": document_types} if document_types else None
            cached_result = await self.cache.get_query_results(
                query, k, strategy.value if strategy else "auto", filters
            )
            if cached_result:
                cache_hit = True
                results, metrics = cached_result
                
                # Record cache hit metrics
                if self.metrics_collector:
                    self.metrics_collector.record_retrieval_metrics(
                        query, strategy or SearchStrategy.HYBRID, metrics, 
                        cache_hit=True, fallback_triggered=False
                    )
                
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return results, metrics
        
        # Select strategy
        if strategy is None:
            strategy = await self._select_optimal_strategy(query)
        
        # Set confidence threshold
        confidence_threshold = min_confidence or self.config.confidence_threshold
        
        # Generate query embedding (check cache first)
        query_embedding = None
        if self.cache:
            query_embedding = await self.cache.get_embedding(query, "default")
        
        if query_embedding is None:
            query_embedding = await self.embeddings.embed_text(query)
            if self.cache:
                await self.cache.cache_embedding(query, "default", query_embedding)
        
        # Perform search using database if available
        if self.config.use_database and self.vector_ops:
            # Search in database
            results = await self._retrieve_from_database(
                query_embedding=query_embedding,
                query_text=query,
                k=k,
                document_types=document_types,
                similarity_threshold=confidence_threshold,
                strategy=strategy,
            )
            
            # Create metrics
            retrieval_time = time.time() - start_time
            scores = [r.score for r in results] if results else [0.0]
            
            metrics = RetrievalMetrics(
                query=query,
                total_documents=len(results),
                retrieved_count=len(results),
                search_strategy=strategy,
                retrieval_time=retrieval_time,
                avg_score=sum(scores) / len(scores) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
                min_score=min(scores) if scores else 0.0,
            )
        else:
            # Fallback to in-memory search
            candidate_documents = self._filter_documents(document_types)
            
            if not candidate_documents:
                logger.warning("No candidate documents found for retrieval")
                empty_metrics = RetrievalMetrics(
                    query=query,
                    total_documents=0,
                    retrieved_count=0,
                    search_strategy=strategy,
                    retrieval_time=time.time() - start_time,
                    avg_score=0.0,
                    max_score=0.0,
                    min_score=0.0,
                )
                
                # Record metrics even for empty results
                if self.metrics_collector:
                    self.metrics_collector.record_retrieval_metrics(
                        query, strategy, empty_metrics, cache_hit=cache_hit
                    )
                
                return [], empty_metrics
            
            # Update search config
            search_config = self.config.search_config
            search_config.max_results = k
            if min_confidence is not None:
                search_config.min_score_threshold = min_confidence
            
            # Perform search
            results, metrics = await self.retrieval_engine.search(
                query, candidate_documents, strategy, **kwargs
            )
        
        # Apply confidence-based fallback strategies (Requirement 2.2)
        if (results and metrics.avg_score < self.config.fallback_threshold) or not results:
            logger.info(f"Applying fallback strategy for query: {query[:50]}... (confidence: {metrics.avg_score:.3f})")
            fallback_triggered = True
            
            # Try multiple fallback strategies
            fallback_results, fallback_metrics = await self._apply_comprehensive_fallback(
                query, query_embedding, document_types, k, strategy, **kwargs
            )
            
            if fallback_results and (not results or fallback_metrics.avg_score > metrics.avg_score):
                results, metrics = fallback_results, fallback_metrics
                logger.info(f"Fallback successful: improved confidence to {metrics.avg_score:.3f}")
            else:
                logger.warning(f"Fallback did not improve results for query: {query[:50]}...")
        
        # Cache results if enabled and not from cache
        if self.cache and not cache_hit and results:
            filters = {"document_types": document_types} if document_types else None
            await self.cache.cache_query_results(
                query, k, strategy.value, results, metrics, filters, query_embedding
            )
        
        # Update performance tracking
        self.processing_stats["queries_processed"] += 1
        self.processing_stats["avg_retrieval_time"] = (
            (self.processing_stats["avg_retrieval_time"] * (self.processing_stats["queries_processed"] - 1) +
             metrics.retrieval_time) / self.processing_stats["queries_processed"]
        )
        
        # Record strategy performance for adaptive learning
        if self.config.enable_adaptive_learning:
            self.strategy_performance[strategy].append(metrics.avg_score)
        
        # Record comprehensive metrics
        if self.metrics_collector:
            self.metrics_collector.record_retrieval_metrics(
                query, strategy, metrics, cache_hit=cache_hit, 
                fallback_triggered=fallback_triggered
            )
        
        logger.info(
            f"Retrieved {len(results)} results for query: {query[:50]}... "
            f"(strategy: {strategy.value}, avg_score: {metrics.avg_score:.3f}, "
            f"cache_hit: {cache_hit}, fallback: {fallback_triggered})"
        )
        
        return results, metrics
    
    async def provide_feedback(
        self,
        query: str,
        results: List[RetrievalResult],
        relevance_scores: List[float]
    ) -> None:
        """Provide feedback on retrieval results for adaptive learning (Requirement 2.5)."""
        if not self.config.enable_adaptive_learning:
            return
        
        if len(results) != len(relevance_scores):
            raise ValueError("Number of results must match number of relevance scores")
        
        # Store feedback
        if query not in self.query_feedback:
            self.query_feedback[query] = []
        
        self.query_feedback[query].extend(relevance_scores)
        
        # Update strategy performance based on feedback
        for result, score in zip(results, relevance_scores):
            strategy = result.search_strategy
            if strategy in self.strategy_performance:
                self.strategy_performance[strategy].append(score)
        
        # Record user satisfaction metrics
        if self.metrics_collector:
            avg_satisfaction = sum(relevance_scores) / len(relevance_scores)
            # Record each individual feedback score
            for score in relevance_scores:
                self.metrics_collector._record_metric(
                    MetricType.USER_SATISFACTION,
                    score,
                    time.time(),
                    {"query": query[:50], "avg_satisfaction": avg_satisfaction}
                )
        
        # Adaptive learning: adjust strategy weights based on feedback
        await self._update_strategy_weights(query, results, relevance_scores)
        
        # Enhanced adaptive learning: Update confidence thresholds based on feedback
        await self._adapt_confidence_thresholds(relevance_scores)
        
        # Update retrieval cache with feedback-based scoring
        if self.cache:
            await self._update_cache_with_feedback(query, results, relevance_scores)
        
        logger.debug(f"Feedback recorded for query: {query[:50]}... (avg satisfaction: {sum(relevance_scores)/len(relevance_scores):.3f})")
    
    async def get_document(self, document_id: UUID) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(document_id)
    
    async def remove_document(self, document_id: UUID) -> bool:
        """Remove a document from the system and invalidate related cache entries."""
        # Check if document exists in memory or database
        document = None
        if document_id in self.documents:
            document = self.documents[document_id]
        elif self.config.use_database and self.search_ops:
            # Try to get from database
            doc_record = await self.search_ops.get_document_by_id(document_id)
            if doc_record:
                document = self._document_record_to_document(doc_record)
        
        if not document:
            return False
        
        try:
            # Invalidate cache entries first
            await self.invalidate_document_cache(document_id)
            
            # Remove from database if enabled
            if self.config.use_database and self.db_connection:
                async with self.db_connection.get_transaction() as conn:
                    # Delete chunks first (foreign key constraint)
                    await conn.execute(
                        "DELETE FROM document_chunks WHERE document_id = $1",
                        document_id
                    )
                    
                    # Delete document
                    await conn.execute(
                        "DELETE FROM documents WHERE id = $1",
                        document_id
                    )
                    
                    logger.debug(f"Document removed from database: {document_id}")
            
            # Remove from in-memory storage
            if document_id in self.documents:
                content_hash = document.content_hash
                if content_hash in self.document_index:
                    del self.document_index[content_hash]
                del self.documents[document_id]
            
            logger.info(f"Document removed: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {document_id}: {e}")
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and performance metrics."""
        # Calculate strategy performance averages
        strategy_stats = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_stats[strategy.value] = {
                    "avg_score": sum(scores) / len(scores),
                    "query_count": len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                }
            else:
                strategy_stats[strategy.value] = {
                    "avg_score": 0.0,
                    "query_count": 0,
                    "max_score": 0.0,
                    "min_score": 0.0,
                }
        
        # Get database stats if available
        database_stats = {}
        if self.config.use_database and self.db_connection:
            try:
                database_stats = await self.db_connection.get_database_info()
                if self.db_schema:
                    schema_info = await self.db_schema.get_schema_info()
                    database_stats.update(schema_info)
            except Exception as e:
                database_stats = {"error": f"Failed to get database stats: {e}"}
        
        stats = {
            "documents": {
                "total": len(self.documents),
                "processed": sum(1 for doc in self.documents.values() if doc.is_processed),
                "pending": sum(1 for doc in self.documents.values() if doc.status == ProcessingStatus.PENDING),
                "failed": sum(1 for doc in self.documents.values() if doc.status == ProcessingStatus.FAILED),
            },
            "chunks": {
                "total": sum(len(doc.chunks) for doc in self.documents.values()),
                "avg_per_document": (
                    sum(len(doc.chunks) for doc in self.documents.values()) / len(self.documents)
                    if self.documents else 0
                ),
            },
            "processing_stats": self.processing_stats.copy(),
            "strategy_performance": strategy_stats,
            "embedding_stats": self.embeddings.get_embedding_stats(),
            "cache_stats": self.cache.get_cache_stats() if self.cache else {},
            "metrics_stats": self.metrics_collector.get_current_metrics() if self.metrics_collector else {},
            "database": {
                "enabled": self.config.use_database,
                "connected": self.db_connection.is_connected if self.db_connection else False,
                **database_stats
            }
        }
        
        return stats
    
    def _detect_document_type(self, content: Union[str, bytes]) -> DocumentType:
        """Auto-detect document type from content."""
        if isinstance(content, bytes):
            # Check for common binary file signatures
            if content.startswith(b'%PDF'):
                return DocumentType.PDF
            elif content.startswith((b'\xff\xd8\xff', b'\x89PNG')):
                return DocumentType.IMAGE
            else:
                return DocumentType.TEXT
        else:
            # Text content - could be code, structured data, or plain text
            content_lower = content.lower().strip()
            
            if content_lower.startswith(('def ', 'class ', 'import ', 'from ')):
                return DocumentType.CODE
            elif content_lower.startswith(('{', '[')):
                return DocumentType.STRUCTURED
            else:
                return DocumentType.TEXT
    
    def _filter_documents(
        self,
        document_types: Optional[List[DocumentType]] = None
    ) -> List[Document]:
        """Filter documents by type and processing status."""
        documents = [
            doc for doc in self.documents.values()
            if doc.is_processed  # Only include processed documents
        ]
        
        if document_types:
            documents = [
                doc for doc in documents
                if doc.document_type in document_types
            ]
        
        return documents
    
    async def _select_optimal_strategy(self, query: str) -> SearchStrategy:
        """Select optimal search strategy based on query and performance history."""
        if not self.config.enable_adaptive_learning:
            return SearchStrategy.HYBRID
        
        # Simple heuristics for strategy selection
        query_length = len(query.split())
        
        # Check strategy performance history
        strategy_scores = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_scores[strategy] = sum(scores) / len(scores)
            else:
                strategy_scores[strategy] = 0.5  # Default score
        
        # Select best performing strategy, with fallback to heuristics
        if strategy_scores:
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            
            # Use best strategy if it's significantly better
            if strategy_scores[best_strategy] > 0.7:
                return best_strategy
        
        # Fallback to heuristics
        if query_length <= 2:
            return SearchStrategy.KEYWORD_ONLY
        elif query_length >= 10:
            return SearchStrategy.VECTOR_ONLY
        else:
            return SearchStrategy.HYBRID
    
    async def _apply_comprehensive_fallback(
        self,
        query: str,
        query_embedding: np.ndarray,
        document_types: Optional[List[DocumentType]],
        k: int,
        original_strategy: SearchStrategy,
        **kwargs
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Apply comprehensive fallback strategies when confidence is below threshold."""
        fallback_strategies = []
        
        # Strategy 1: Try different search strategy
        if original_strategy == SearchStrategy.HYBRID:
            fallback_strategies.append(SearchStrategy.VECTOR_ONLY)
        elif original_strategy == SearchStrategy.VECTOR_ONLY:
            fallback_strategies.append(SearchStrategy.KEYWORD_ONLY)
        else:
            fallback_strategies.append(SearchStrategy.HYBRID)
        
        # Strategy 2: Expand search with relaxed confidence threshold
        fallback_strategies.append(original_strategy)
        
        # Strategy 3: Cross-modal search (if applicable)
        if document_types and len(document_types) == 1:
            # Try searching in all document types
            fallback_strategies.append(original_strategy)
        
        best_results = []
        best_metrics = None
        best_score = 0.0
        
        for i, strategy in enumerate(fallback_strategies):
            try:
                logger.debug(f"Trying fallback strategy {i+1}/{len(fallback_strategies)}: {strategy.value}")
                
                if i == 0:
                    # Different strategy
                    results, metrics = await self._execute_fallback_search(
                        query, query_embedding, document_types, k, strategy, **kwargs
                    )
                elif i == 1:
                    # Relaxed threshold (lower confidence requirement)
                    relaxed_threshold = max(0.3, self.config.fallback_threshold - 0.2)
                    results, metrics = await self._execute_fallback_search(
                        query, query_embedding, document_types, k * 2, strategy, 
                        min_confidence=relaxed_threshold, **kwargs
                    )
                    # Take top k results
                    results = results[:k]
                else:
                    # Cross-modal search
                    results, metrics = await self._execute_fallback_search(
                        query, query_embedding, None, k, strategy, **kwargs
                    )
                
                if results and metrics.avg_score > best_score:
                    best_results = results
                    best_metrics = metrics
                    best_score = metrics.avg_score
                    
                    # If we get good enough results, stop trying
                    if best_score >= self.config.confidence_threshold * 0.8:
                        break
                        
            except Exception as e:
                logger.warning(f"Fallback strategy {i+1} failed: {e}")
                continue
        
        if best_metrics is None:
            # Create empty metrics if all fallbacks failed
            best_metrics = RetrievalMetrics(
                query=query,
                total_documents=0,
                retrieved_count=0,
                search_strategy=original_strategy,
                retrieval_time=0.0,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
            )
        
        return best_results, best_metrics
    
    async def _execute_fallback_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        document_types: Optional[List[DocumentType]],
        k: int,
        strategy: SearchStrategy,
        min_confidence: Optional[float] = None,
        **kwargs
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Execute a specific fallback search strategy."""
        start_time = time.time()
        
        if self.config.use_database and self.vector_ops:
            # Database search
            results = await self._retrieve_from_database(
                query_embedding=query_embedding,
                query_text=query,
                k=k,
                document_types=document_types,
                similarity_threshold=min_confidence or self.config.fallback_threshold,
                strategy=strategy,
            )
            
            # Create metrics
            retrieval_time = time.time() - start_time
            scores = [r.score for r in results] if results else [0.0]
            
            metrics = RetrievalMetrics(
                query=query,
                total_documents=len(results),
                retrieved_count=len(results),
                search_strategy=strategy,
                retrieval_time=retrieval_time,
                avg_score=sum(scores) / len(scores) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
                min_score=min(scores) if scores else 0.0,
            )
        else:
            # In-memory search
            candidate_documents = self._filter_documents(document_types)
            
            if not candidate_documents:
                return [], RetrievalMetrics(
                    query=query,
                    total_documents=0,
                    retrieved_count=0,
                    search_strategy=strategy,
                    retrieval_time=time.time() - start_time,
                    avg_score=0.0,
                    max_score=0.0,
                    min_score=0.0,
                )
            
            # Update search config for fallback
            search_config = self.config.search_config
            search_config.max_results = k
            if min_confidence is not None:
                search_config.min_score_threshold = min_confidence
            
            results, metrics = await self.retrieval_engine.search(
                query, candidate_documents, strategy, **kwargs
            )
        
        return results, metrics
    
    async def _chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces."""
        if isinstance(document.content, bytes):
            # For binary content, create a single chunk with metadata
            return [
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=0,
                    content=f"Binary content: {document.metadata.title or 'Untitled'}",
                    semantic_type="binary",
                )
            ]
        
        content = str(document.content)
        
        if self.config.chunking_strategy == "semantic":
            return await self._semantic_chunking(document, content)
        else:
            return await self._simple_chunking(document, content)
    
    async def _simple_chunking(self, document: Document, content: str) -> List[DocumentChunk]:
        """Simple fixed-size chunking."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            
            if len(chunk_content.strip()) < self.config.min_chunk_size:
                continue
            
            chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=len(chunks),
                content=chunk_content.strip(),
                start_position=i,
                end_position=min(i + chunk_size, len(content)),
                semantic_type="paragraph",
            )
            chunks.append(chunk)
            
            if len(chunks) >= self.config.max_chunks_per_document:
                break
        
        return chunks
    
    async def _semantic_chunking(self, document: Document, content: str) -> List[DocumentChunk]:
        """Semantic chunking based on content structure."""
        # For now, use simple paragraph-based chunking
        # In a full implementation, this would use NLP techniques
        
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        current_position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.config.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=len(chunks),
                    content=current_chunk.strip(),
                    start_position=current_position,
                    end_position=current_position + len(current_chunk),
                    semantic_type="paragraph",
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_position += len(current_chunk)
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=len(chunks),
                content=current_chunk.strip(),
                start_position=current_position,
                end_position=current_position + len(current_chunk),
                semantic_type="paragraph",
            )
            chunks.append(chunk)
        
        return chunks[:self.config.max_chunks_per_document]
    
    async def _generate_chunk_embeddings(self, chunks: List[DocumentChunk]) -> List[np.ndarray]:
        """Generate embeddings for document chunks."""
        if not chunks:
            return []
        
        # Extract chunk contents
        chunk_contents = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = await self.embeddings.embed_texts(chunk_contents)
        
        return embeddings
    
    # Database Integration Methods
    
    async def _store_document_in_database(self, document: Document) -> None:
        """Store document in Supabase database."""
        if not self.vector_ops:
            raise RuntimeError("Database not initialized")
        
        try:
            # Convert to database record
            doc_record = self._document_to_record(document)
            
            # Store document with embedding
            stored_id = await self.vector_ops.store_document_embedding(doc_record)
            
            # Update in-memory index
            self.document_index[document.content_hash] = document.id
            
            logger.debug(f"Document stored in database: {stored_id}")
            
        except Exception as e:
            logger.error(f"Failed to store document in database: {e}")
            # Fallback to in-memory storage
            self.documents[document.id] = document
            self.document_index[document.content_hash] = document.id
            raise
    
    async def _store_chunks_in_database(self, document: Document) -> None:
        """Store document chunks in database."""
        if not self.vector_ops or not document.chunks:
            return
        
        try:
            # Convert chunks to records
            chunk_records = [
                self._chunk_to_record(chunk) for chunk in document.chunks
            ]
            
            # Store chunks in batch
            stored_ids = await self.vector_ops.store_chunk_embeddings(chunk_records)
            
            logger.debug(f"Stored {len(stored_ids)} chunks in database")
            
        except Exception as e:
            logger.error(f"Failed to store chunks in database: {e}")
            raise
    
    async def _retrieve_from_database(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int,
        document_types: Optional[List[DocumentType]] = None,
        similarity_threshold: float = 0.7,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
    ) -> List[RetrievalResult]:
        """Retrieve documents from database using vector similarity search."""
        if not self.search_ops:
            return []
        
        try:
            if strategy == SearchStrategy.HYBRID:
                # Use hybrid search
                search_results = await self.vector_ops.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=query_text,
                    similarity_threshold=similarity_threshold,
                    max_results=k,
                    document_types=document_types,
                )
            else:
                # Use vector-only search
                search_results = await self.vector_ops.similarity_search(
                    query_embedding=query_embedding,
                    similarity_threshold=similarity_threshold,
                    max_results=k,
                    document_types=document_types,
                    search_chunks=True,
                    search_documents=False,
                )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in search_results:
                retrieval_results.append(RetrievalResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.similarity,
                    metadata={
                        "document_type": result.document_type.value,
                        "title": result.title,
                        "chunk_index": result.chunk_index,
                        "section_title": result.section_title,
                        "importance_score": result.importance_score,
                    }
                ))
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Database retrieval failed: {e}")
            return []
    
    def _document_to_record(self, document: Document) -> DocumentRecord:
        """Convert Document to DocumentRecord for database storage."""
        return DocumentRecord(
            id=document.id,
            content=document.content if isinstance(document.content, str) else document.content.decode('utf-8', errors='ignore'),
            document_type=document.document_type,
            status=document.status,
            error_message=document.error_message,
            title=document.metadata.title,
            file_path=document.metadata.file_path,
            file_size=document.metadata.file_size,
            mime_type=document.metadata.mime_type,
            encoding=document.metadata.encoding,
            language=document.metadata.language,
            word_count=document.metadata.word_count,
            page_count=document.metadata.page_count,
            duration=document.metadata.duration,
            processing_time=document.metadata.processing_time,
            chunk_count=len(document.chunks),
            embedding_model=document.metadata.embedding_model,
            embedding=document.embedding,
            parent_id=document.parent_id,
            access_level=document.metadata.access_level,
            contains_pii=document.metadata.contains_pii,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processed_at=document.processed_at,
        )
    
    def _chunk_to_record(self, chunk: DocumentChunk) -> ChunkRecord:
        """Convert DocumentChunk to ChunkRecord for database storage."""
        return ChunkRecord(
            id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            start_position=chunk.start_position,
            end_position=chunk.end_position,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            semantic_type=chunk.semantic_type,
            importance_score=chunk.importance_score,
            embedding=chunk.embedding,
            created_at=chunk.created_at,
        )
    
    def _document_record_to_document(self, record: DocumentRecord) -> Document:
        """Convert DocumentRecord back to Document."""
        from quantum_moe_mas.rag.document import DocumentMetadata
        
        metadata = DocumentMetadata(
            title=record.title,
            file_path=record.file_path,
            file_size=record.file_size,
            mime_type=record.mime_type,
            encoding=record.encoding,
            language=record.language,
            word_count=record.word_count,
            page_count=record.page_count,
            duration=record.duration,
            processing_time=record.processing_time,
            embedding_model=record.embedding_model,
            access_level=record.access_level,
            contains_pii=record.contains_pii,
        )
        
        return Document(
            id=record.id,
            content=record.content,
            document_type=record.document_type,
            metadata=metadata,
            status=record.status,
            error_message=record.error_message,
            embedding=record.embedding,
            parent_id=record.parent_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
            processed_at=record.processed_at,
        )
    
    async def batch_add_documents(
        self,
        documents: List[Document],
        process_immediately: bool = True,
    ) -> List[Document]:
        """Add multiple documents using batch operations for better performance."""
        if not self.config.use_database or not self.batch_ops:
            # Fallback to regular add_documents
            return await self.add_documents(documents, process_immediately)
        
        try:
            # Process documents if needed
            if process_immediately:
                for document in documents:
                    if document.status != ProcessingStatus.COMPLETED:
                        await self.process_document_standalone(document)
            
            # Store in database using batch operations
            result = await self.batch_ops.batch_store_documents(
                documents, include_embeddings=True
            )
            
            # Store chunks for each document
            for document in documents:
                if document.chunks:
                    await self._store_chunks_in_database(document)
            
            logger.info(
                f"Batch document storage: {result.successful_items}/{result.total_items} "
                f"successful in {result.processing_time:.2f}s"
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Batch document storage failed: {e}")
            # Fallback to individual storage
            return await self.add_documents(documents, process_immediately)
    
    async def process_document_standalone(self, document: Document) -> None:
        """Process a document without requiring it to be in self.documents."""
        if document.status == ProcessingStatus.COMPLETED:
            return
        
        start_time = time.time()
        document.update_status(ProcessingStatus.PROCESSING)
        
        try:
            # Generate document embedding
            document.embedding = await self.embeddings.embed_document(document)
            
            # Chunk document
            chunks = await self._chunk_document(document)
            
            # Generate chunk embeddings
            chunk_embeddings = await self._generate_chunk_embeddings(chunks)
            
            # Update chunks with embeddings
            for chunk, embedding in zip(chunks, chunk_embeddings):
                chunk.embedding = embedding
                document.chunks.append(chunk)
            
            # Update status
            document.update_status(ProcessingStatus.COMPLETED)
            
            processing_time = time.time() - start_time
            logger.debug(f"Document processed standalone: {document.id} ({processing_time:.3f}s)")
            
        except Exception as e:
            document.update_status(ProcessingStatus.FAILED, str(e))
            logger.error(f"Failed to process document {document.id}: {e}")
            raise
    
    async def _update_strategy_weights(
        self,
        query: str,
        results: List[RetrievalResult],
        relevance_scores: List[float]
    ) -> None:
        """Update strategy performance weights based on user feedback."""
        if not results or not relevance_scores:
            return
        
        # Calculate average satisfaction for this query
        avg_satisfaction = sum(relevance_scores) / len(relevance_scores)
        
        # Update strategy performance based on which strategies were used
        strategy_scores = {}
        for result, score in zip(results, relevance_scores):
            strategy = result.search_strategy
            if strategy not in strategy_scores:
                strategy_scores[strategy] = []
            strategy_scores[strategy].append(score)
        
        # Update performance tracking with weighted scores
        for strategy, scores in strategy_scores.items():
            avg_strategy_score = sum(scores) / len(scores)
            
            # Add to strategy performance with higher weight for recent feedback
            if strategy in self.strategy_performance:
                # Keep recent performance more heavily weighted
                recent_weight = 2.0
                self.strategy_performance[strategy].append(avg_strategy_score * recent_weight)
                
                # Limit history to prevent memory growth
                if len(self.strategy_performance[strategy]) > 100:
                    self.strategy_performance[strategy] = self.strategy_performance[strategy][-50:]
        
        logger.debug(f"Updated strategy weights based on feedback (avg satisfaction: {avg_satisfaction:.3f})")
    
    async def get_performance_report(self, hours_back: int = 1) -> Optional[Any]:
        """Get comprehensive performance report from metrics collector."""
        if not self.metrics_collector:
            return None
        
        return self.metrics_collector.generate_performance_report(hours_back)
    
    async def cleanup_cache(self) -> Dict[str, int]:
        """Clean up expired cache entries."""
        if not self.cache:
            return {}
        
        return await self.cache.cleanup()
    
    async def invalidate_document_cache(self, document_id: UUID) -> None:
        """Invalidate cache entries related to a specific document."""
        if self.cache:
            await self.cache.invalidate_document(str(document_id))
    
    async def _adapt_confidence_thresholds(self, relevance_scores: List[float]) -> None:
        """Dynamically adapt confidence thresholds based on user feedback."""
        if not relevance_scores:
            return
        
        avg_satisfaction = sum(relevance_scores) / len(relevance_scores)
        
        # Adjust confidence threshold based on satisfaction
        if avg_satisfaction < 0.6:
            # Low satisfaction - increase confidence threshold to be more selective
            self.config.confidence_threshold = min(0.9, self.config.confidence_threshold + 0.05)
            logger.debug(f"Increased confidence threshold to {self.config.confidence_threshold:.3f} due to low satisfaction")
        elif avg_satisfaction > 0.8:
            # High satisfaction - can afford to lower threshold slightly for more results
            self.config.confidence_threshold = max(0.5, self.config.confidence_threshold - 0.02)
            logger.debug(f"Decreased confidence threshold to {self.config.confidence_threshold:.3f} due to high satisfaction")
        
        # Adjust fallback threshold similarly
        if avg_satisfaction < 0.5:
            self.config.fallback_threshold = min(0.5, self.config.fallback_threshold + 0.05)
        elif avg_satisfaction > 0.85:
            self.config.fallback_threshold = max(0.2, self.config.fallback_threshold - 0.02)
    
    async def _update_cache_with_feedback(
        self,
        query: str,
        results: List[RetrievalResult],
        relevance_scores: List[float]
    ) -> None:
        """Update cache entries with feedback-based relevance scoring."""
        if not self.cache or not results:
            return
        
        # Calculate weighted relevance for cache prioritization
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # If feedback is very positive, extend cache TTL for this query pattern
        if avg_relevance > 0.8:
            # Find similar cached queries and extend their TTL
            # This is a simplified implementation - in production, you'd use embedding similarity
            query_words = set(query.lower().split())
            
            # Extend TTL for high-quality results
            cache_key = self.cache._generate_query_key(query, len(results), "feedback_update", None)
            if cache_key in self.cache.query_cache._cache:
                entry = self.cache.query_cache._cache[cache_key]
                entry.ttl = entry.ttl * 1.5 if entry.ttl else self.cache.query_ttl * 1.5
                logger.debug(f"Extended cache TTL for high-quality query: {query[:50]}...")
        
        elif avg_relevance < 0.4:
            # Poor feedback - consider invalidating similar cache entries
            cache_key = self.cache._generate_query_key(query, len(results), "feedback_update", None)
            await self.cache.query_cache.remove(cache_key)
            logger.debug(f"Invalidated cache for low-quality query: {query[:50]}...")
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization and performance metrics."""
        stats = await self.get_system_stats()
        
        # Add optimization-specific metrics
        optimization_metrics = {
            "confidence_thresholds": {
                "current_confidence_threshold": self.config.confidence_threshold,
                "current_fallback_threshold": self.config.fallback_threshold,
                "adaptive_learning_enabled": self.config.enable_adaptive_learning,
            },
            "fallback_performance": {
                "fallback_triggers": sum(
                    len(scores) for scores in self.strategy_performance.values()
                    if any(score < self.config.fallback_threshold for score in scores)
                ),
                "fallback_success_rate": self._calculate_fallback_success_rate(),
            },
            "cache_optimization": self.cache.get_cache_stats() if self.cache else {},
            "adaptive_learning": {
                "queries_with_feedback": len(self.query_feedback),
                "total_feedback_points": sum(len(scores) for scores in self.query_feedback.values()),
                "avg_user_satisfaction": self._calculate_avg_user_satisfaction(),
            },
            "strategy_adaptation": {
                strategy.value: {
                    "performance_history_length": len(scores),
                    "avg_performance": sum(scores) / len(scores) if scores else 0.0,
                    "recent_performance": sum(scores[-10:]) / min(10, len(scores)) if scores else 0.0,
                }
                for strategy, scores in self.strategy_performance.items()
            },
        }
        
        # Merge with existing stats
        stats["optimization_metrics"] = optimization_metrics
        return stats
    
    def _calculate_fallback_success_rate(self) -> float:
        """Calculate the success rate of fallback strategies."""
        # This is a simplified calculation - in production, you'd track this more precisely
        total_fallbacks = 0
        successful_fallbacks = 0
        
        for scores in self.strategy_performance.values():
            for score in scores:
                if score < self.config.confidence_threshold:
                    total_fallbacks += 1
                    if score >= self.config.fallback_threshold:
                        successful_fallbacks += 1
        
        return successful_fallbacks / total_fallbacks if total_fallbacks > 0 else 0.0
    
    def _calculate_avg_user_satisfaction(self) -> float:
        """Calculate average user satisfaction from feedback."""
        all_scores = []
        for scores in self.query_feedback.values():
            all_scores.extend(scores)
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization based on collected metrics and feedback."""
        optimization_results = {
            "cache_cleanup": {},
            "threshold_adjustments": {},
            "strategy_rebalancing": {},
            "recommendations": [],
        }
        
        # 1. Clean up cache
        if self.cache:
            cleanup_stats = await self.cache.cleanup()
            optimization_results["cache_cleanup"] = cleanup_stats
        
        # 2. Analyze and adjust thresholds based on performance history
        if self.config.enable_adaptive_learning and self.query_feedback:
            all_feedback = []
            for scores in self.query_feedback.values():
                all_feedback.extend(scores)
            
            if all_feedback:
                avg_satisfaction = sum(all_feedback) / len(all_feedback)
                
                # Adjust thresholds based on overall satisfaction
                old_confidence = self.config.confidence_threshold
                old_fallback = self.config.fallback_threshold
                
                if avg_satisfaction < 0.6:
                    self.config.confidence_threshold = min(0.9, self.config.confidence_threshold + 0.1)
                    self.config.fallback_threshold = min(0.6, self.config.fallback_threshold + 0.05)
                elif avg_satisfaction > 0.8:
                    self.config.confidence_threshold = max(0.5, self.config.confidence_threshold - 0.05)
                    self.config.fallback_threshold = max(0.2, self.config.fallback_threshold - 0.02)
                
                optimization_results["threshold_adjustments"] = {
                    "confidence_threshold": {
                        "old": old_confidence,
                        "new": self.config.confidence_threshold,
                        "change": self.config.confidence_threshold - old_confidence,
                    },
                    "fallback_threshold": {
                        "old": old_fallback,
                        "new": self.config.fallback_threshold,
                        "change": self.config.fallback_threshold - old_fallback,
                    },
                    "avg_satisfaction": avg_satisfaction,
                }
        
        # 3. Strategy rebalancing based on performance
        strategy_changes = {}
        for strategy, scores in self.strategy_performance.items():
            if len(scores) >= 10:  # Only rebalance with sufficient data
                recent_performance = sum(scores[-10:]) / 10
                overall_performance = sum(scores) / len(scores)
                
                performance_trend = recent_performance - overall_performance
                strategy_changes[strategy.value] = {
                    "recent_performance": recent_performance,
                    "overall_performance": overall_performance,
                    "trend": performance_trend,
                    "sample_size": len(scores),
                }
        
        optimization_results["strategy_rebalancing"] = strategy_changes
        
        # 4. Generate optimization recommendations
        recommendations = []
        
        if self.cache:
            cache_stats = self.cache.get_cache_stats()
            if cache_stats["overall"]["total_hit_rate"] < 0.3:
                recommendations.append("Consider increasing cache size or TTL to improve hit rate")
        
        if optimization_results["threshold_adjustments"]:
            confidence_change = optimization_results["threshold_adjustments"]["confidence_threshold"]["change"]
            if abs(confidence_change) > 0.05:
                recommendations.append(f"Confidence threshold adjusted by {confidence_change:+.3f} based on user feedback")
        
        for strategy, perf in strategy_changes.items():
            if perf["trend"] < -0.1:
                recommendations.append(f"Strategy {strategy} showing declining performance - consider investigation")
            elif perf["trend"] > 0.1:
                recommendations.append(f"Strategy {strategy} showing improved performance - consider prioritizing")
        
        optimization_results["recommendations"] = recommendations
        
        logger.info(f"Performance optimization completed: {len(recommendations)} recommendations generated")
        return optimization_results