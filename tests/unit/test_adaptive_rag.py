"""
Unit tests for the Adaptive Multi-Modal RAG system.
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from quantum_moe_mas.rag.adaptive_rag import AdaptiveMultiModalRAG, AdaptiveRAGConfig
from quantum_moe_mas.rag.document import Document, DocumentType, DocumentMetadata
from quantum_moe_mas.rag.embeddings import EmbeddingConfig
from quantum_moe_mas.rag.retrieval import SearchConfig, SearchStrategy


class TestAdaptiveMultiModalRAG:
    """Test cases for AdaptiveMultiModalRAG."""
    
    @pytest.fixture
    def rag_config(self):
        """Create test RAG configuration."""
        return AdaptiveRAGConfig(
            embedding_config=EmbeddingConfig(
                text_model="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dimension=384,
                cache_embeddings=False,  # Disable caching for tests
            ),
            search_config=SearchConfig(
                max_results=5,
                min_score_threshold=0.1,
            ),
            chunk_size=500,
            chunk_overlap=100,
            enable_adaptive_learning=True,
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                content="This is a test document about artificial intelligence and machine learning.",
                document_type=DocumentType.TEXT,
                metadata=DocumentMetadata(title="AI Document")
            ),
            Document(
                content="Quantum computing represents a paradigm shift in computational capabilities.",
                document_type=DocumentType.TEXT,
                metadata=DocumentMetadata(title="Quantum Document")
            ),
            Document(
                content="Cybersecurity is crucial for protecting digital assets and infrastructure.",
                document_type=DocumentType.TEXT,
                metadata=DocumentMetadata(title="Security Document")
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_initialization(self, rag_config):
        """Test RAG system initialization."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        assert rag.config == rag_config
        assert rag.embeddings is not None
        assert rag.retrieval_engine is not None
        assert len(rag.documents) == 0
        assert len(rag.document_index) == 0
    
    @pytest.mark.asyncio
    async def test_add_document(self, rag_config):
        """Test adding a single document."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        # Mock the embedding generation to avoid loading actual models
        with patch.object(rag.embeddings, 'embed_document', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.random.rand(384).astype(np.float32)
            
            with patch.object(rag.embeddings, 'embed_texts', new_callable=AsyncMock) as mock_embed_texts:
                mock_embed_texts.return_value = [np.random.rand(384).astype(np.float32)]
                
                document = await rag.add_document(
                    content="Test document content",
                    document_type=DocumentType.TEXT,
                    metadata={"title": "Test Document"}
                )
        
        assert document.id in rag.documents
        assert document.content == "Test document content"
        assert document.document_type == DocumentType.TEXT
        assert document.metadata.custom_fields["title"] == "Test Document"
        assert document.is_processed
        assert document.has_embedding
        assert len(document.chunks) > 0
    
    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, rag_config, sample_documents):
        """Test adding multiple documents."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        # Mock embedding generation
        with patch.object(rag.embeddings, 'embed_document', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.random.rand(384).astype(np.float32)
            
            with patch.object(rag.embeddings, 'embed_texts', new_callable=AsyncMock) as mock_embed_texts:
                mock_embed_texts.return_value = [np.random.rand(384).astype(np.float32)]
                
                added_docs = await rag.add_documents(sample_documents)
        
        assert len(added_docs) == 3
        assert len(rag.documents) == 3
        
        for doc in added_docs:
            assert doc.is_processed
            assert doc.has_embedding
    
    @pytest.mark.asyncio
    async def test_retrieve_documents(self, rag_config, sample_documents):
        """Test document retrieval."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        # Mock embedding generation
        with patch.object(rag.embeddings, 'embed_document', new_callable=AsyncMock) as mock_embed_doc:
            mock_embed_doc.return_value = np.random.rand(384).astype(np.float32)
            
            with patch.object(rag.embeddings, 'embed_texts', new_callable=AsyncMock) as mock_embed_texts:
                mock_embed_texts.return_value = [np.random.rand(384).astype(np.float32)]
                
                with patch.object(rag.embeddings, 'embed_text', new_callable=AsyncMock) as mock_embed_query:
                    mock_embed_query.return_value = np.random.rand(384).astype(np.float32)
                    
                    # Add documents
                    await rag.add_documents(sample_documents)
                    
                    # Perform retrieval
                    results, metrics = await rag.retrieve(
                        query="artificial intelligence",
                        k=2,
                        strategy=SearchStrategy.HYBRID
                    )
        
        assert len(results) <= 2
        assert metrics.query == "artificial intelligence"
        assert metrics.search_strategy == SearchStrategy.HYBRID
        assert metrics.retrieval_time > 0
    
    @pytest.mark.asyncio
    async def test_document_chunking(self, rag_config):
        """Test document chunking functionality."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        # Create a long document
        long_content = "This is a test paragraph. " * 100  # ~2500 characters
        document = Document(
            content=long_content,
            document_type=DocumentType.TEXT,
        )
        
        chunks = await rag._chunk_document(document)
        
        assert len(chunks) > 1  # Should be chunked
        assert all(chunk.document_id == document.id for chunk in chunks)
        assert all(len(chunk.content) <= rag_config.chunk_size for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_selection(self, rag_config):
        """Test adaptive strategy selection."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        # Test short query (should prefer keyword)
        strategy = await rag._select_optimal_strategy("AI")
        assert strategy == SearchStrategy.KEYWORD_ONLY
        
        # Test long query (should prefer vector)
        long_query = "What are the implications of artificial intelligence on modern society and economy"
        strategy = await rag._select_optimal_strategy(long_query)
        assert strategy == SearchStrategy.VECTOR_ONLY
        
        # Test medium query (should prefer hybrid)
        medium_query = "machine learning algorithms"
        strategy = await rag._select_optimal_strategy(medium_query)
        assert strategy == SearchStrategy.HYBRID
    
    @pytest.mark.asyncio
    async def test_feedback_learning(self, rag_config):
        """Test adaptive learning from feedback."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        # Mock retrieval results
        from quantum_moe_mas.rag.retrieval import RetrievalResult
        
        mock_results = [
            RetrievalResult(
                document=Document(content="test", document_type=DocumentType.TEXT),
                score=0.8,
                search_strategy=SearchStrategy.HYBRID,
                retrieval_time=0.1,
                rank=1
            ),
            RetrievalResult(
                document=Document(content="test2", document_type=DocumentType.TEXT),
                score=0.6,
                search_strategy=SearchStrategy.HYBRID,
                retrieval_time=0.1,
                rank=2
            ),
        ]
        
        # Provide feedback
        await rag.provide_feedback(
            query="test query",
            results=mock_results,
            relevance_scores=[0.9, 0.3]
        )
        
        assert "test query" in rag.query_feedback
        assert len(rag.query_feedback["test query"]) == 2
        assert SearchStrategy.HYBRID in rag.strategy_performance
        assert len(rag.strategy_performance[SearchStrategy.HYBRID]) == 2
    
    @pytest.mark.asyncio
    async def test_system_stats(self, rag_config, sample_documents):
        """Test system statistics generation."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        # Mock embedding generation
        with patch.object(rag.embeddings, 'embed_document', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.random.rand(384).astype(np.float32)
            
            with patch.object(rag.embeddings, 'embed_texts', new_callable=AsyncMock) as mock_embed_texts:
                mock_embed_texts.return_value = [np.random.rand(384).astype(np.float32)]
                
                await rag.add_documents(sample_documents)
        
        stats = await rag.get_system_stats()
        
        assert "documents" in stats
        assert "chunks" in stats
        assert "processing_stats" in stats
        assert "strategy_performance" in stats
        assert "embedding_stats" in stats
        assert "cache_stats" in stats
        
        assert stats["documents"]["total"] == 3
        assert stats["documents"]["processed"] == 3
        assert stats["chunks"]["total"] > 0
    
    @pytest.mark.asyncio
    async def test_document_removal(self, rag_config):
        """Test document removal."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        # Mock embedding generation
        with patch.object(rag.embeddings, 'embed_document', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.random.rand(384).astype(np.float32)
            
            with patch.object(rag.embeddings, 'embed_texts', new_callable=AsyncMock) as mock_embed_texts:
                mock_embed_texts.return_value = [np.random.rand(384).astype(np.float32)]
                
                document = await rag.add_document("Test content", DocumentType.TEXT)
        
        document_id = document.id
        assert document_id in rag.documents
        
        # Remove document
        removed = await rag.remove_document(document_id)
        assert removed is True
        assert document_id not in rag.documents
        
        # Try to remove non-existent document
        removed = await rag.remove_document(document_id)
        assert removed is False
    
    @pytest.mark.asyncio
    async def test_duplicate_document_handling(self, rag_config):
        """Test handling of duplicate documents."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        content = "This is a unique test document"
        
        # Mock embedding generation
        with patch.object(rag.embeddings, 'embed_document', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.random.rand(384).astype(np.float32)
            
            with patch.object(rag.embeddings, 'embed_texts', new_callable=AsyncMock) as mock_embed_texts:
                mock_embed_texts.return_value = [np.random.rand(384).astype(np.float32)]
                
                # Add document first time
                doc1 = await rag.add_document(content, DocumentType.TEXT)
                
                # Add same document again
                doc2 = await rag.add_document(content, DocumentType.TEXT)
        
        # Should return the same document
        assert doc1.id == doc2.id
        assert len(rag.documents) == 1
    
    def test_document_type_detection(self, rag_config):
        """Test automatic document type detection."""
        rag = AdaptiveMultiModalRAG(rag_config)
        
        # Test text content
        text_type = rag._detect_document_type("This is plain text content")
        assert text_type == DocumentType.TEXT
        
        # Test code content
        code_type = rag._detect_document_type("def hello_world():\n    print('Hello, World!')")
        assert code_type == DocumentType.CODE
        
        # Test structured content
        json_type = rag._detect_document_type('{"key": "value", "number": 42}')
        assert json_type == DocumentType.STRUCTURED
        
        # Test binary content (PDF signature)
        pdf_type = rag._detect_document_type(b'%PDF-1.4\x01\x02\x03')
        assert pdf_type == DocumentType.PDF
        
        # Test binary content (PNG signature)
        png_type = rag._detect_document_type(b'\x89PNG\r\n\x1a\n')
        assert png_type == DocumentType.IMAGE