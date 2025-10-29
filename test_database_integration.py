#!/usr/bin/env python3
"""
Test script for Supabase database integration with AdaptiveMultiModalRAG.
"""

import asyncio
import os
from pathlib import Path

from quantum_moe_mas.rag.adaptive_rag import AdaptiveMultiModalRAG, AdaptiveRAGConfig
from quantum_moe_mas.rag.document import Document, DocumentType
from quantum_moe_mas.rag.database import SupabaseConnection


async def test_database_integration():
    """Test the database integration with AdaptiveMultiModalRAG."""
    print("🧪 Testing Supabase Database Integration...")
    
    # Check if database environment variables are set
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("⚠️  Database environment variables not set. Testing with in-memory storage.")
        config = AdaptiveRAGConfig(use_database=False)
    else:
        print("✅ Database environment variables found. Testing with Supabase.")
        config = AdaptiveRAGConfig(
            use_database=True,
            enable_schema_init=True,
            batch_size=50
        )
    
    # Initialize RAG system
    rag = AdaptiveMultiModalRAG(config)
    
    try:
        # Initialize database if enabled
        if config.use_database:
            print("\n🔌 Initializing database connection...")
            await rag.initialize_database()
            
            if rag.db_connection and rag.db_connection.is_connected:
                print("✅ Database connected successfully")
                
                # Get database info
                db_info = await rag.db_connection.get_database_info()
                print(f"   Database version: {db_info.get('database_version', 'Unknown')}")
                print(f"   pgvector installed: {db_info.get('pgvector_installed', False)}")
                print(f"   Connection pool size: {db_info.get('connection_pool', {}).get('size', 0)}")
            else:
                print("❌ Database connection failed, using in-memory storage")
        
        # Test document addition
        print("\n📄 Testing document addition...")
        
        # Add test documents
        doc1 = await rag.add_document(
            content="This is a test document about artificial intelligence and machine learning. "
                   "It covers various topics including neural networks, deep learning, and natural language processing.",
            document_type=DocumentType.TEXT,
            metadata={"title": "AI Test Document", "category": "technology", "source": "test"}
        )
        print(f"✅ Added document 1: {doc1.id}")
        
        doc2 = await rag.add_document(
            content="Quantum computing represents a revolutionary approach to computation. "
                   "It leverages quantum mechanical phenomena like superposition and entanglement "
                   "to perform calculations that would be impossible for classical computers.",
            document_type=DocumentType.TEXT,
            metadata={"title": "Quantum Computing", "category": "science", "source": "test"}
        )
        print(f"✅ Added document 2: {doc2.id}")
        
        doc3 = await rag.add_document(
            content="Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. "
                   "These attacks are usually aimed at accessing, changing, or destroying sensitive information.",
            document_type=DocumentType.TEXT,
            metadata={"title": "Cybersecurity Basics", "category": "security", "source": "test"}
        )
        print(f"✅ Added document 3: {doc3.id}")
        
        # Test batch document addition
        print("\n📦 Testing batch document addition...")
        batch_docs = [
            Document(
                content=f"Batch document {i}: This is test content for batch processing. "
                       f"It contains information about topic {i} and related concepts.",
                document_type=DocumentType.TEXT,
            )
            for i in range(1, 6)
        ]
        
        added_docs = await rag.batch_add_documents(batch_docs)
        print(f"✅ Added {len(added_docs)} documents in batch")
        
        # Test document retrieval
        print("\n🔍 Testing document retrieval...")
        
        # Test different queries
        test_queries = [
            "artificial intelligence machine learning",
            "quantum computing superposition",
            "cybersecurity digital attacks",
            "batch processing test content"
        ]
        
        for query in test_queries:
            results, metrics = await rag.retrieve(query, k=3)
            print(f"   Query: '{query}'")
            print(f"   Results: {len(results)}, Avg score: {metrics.avg_score:.3f}, "
                  f"Time: {metrics.retrieval_time:.3f}s, Strategy: {metrics.search_strategy.value}")
            
            for i, result in enumerate(results[:2]):  # Show top 2 results
                print(f"     {i+1}. Score: {result.score:.3f}, Content: {result.content[:100]}...")
        
        # Test system statistics
        print("\n📊 Testing system statistics...")
        stats = await rag.get_system_stats()
        
        print(f"   Total documents: {stats['documents']['total']}")
        print(f"   Processed documents: {stats['documents']['processed']}")
        print(f"   Total chunks: {stats['chunks']['total']}")
        print(f"   Queries processed: {stats['processing_stats']['queries_processed']}")
        print(f"   Database enabled: {stats['database']['enabled']}")
        print(f"   Database connected: {stats['database']['connected']}")
        
        if stats['database']['enabled'] and 'total_tables' in stats['database']:
            print(f"   Database tables: {stats['database']['total_tables']}")
            print(f"   Database indexes: {stats['database']['total_indexes']}")
        
        # Test document removal
        print("\n🗑️ Testing document removal...")
        removed = await rag.remove_document(doc3.id)
        if removed:
            print(f"✅ Document {doc3.id} removed successfully")
        else:
            print(f"❌ Failed to remove document {doc3.id}")
        
        # Final stats
        final_stats = await rag.get_system_stats()
        print(f"   Documents after removal: {final_stats['documents']['total']}")
        
        print("\n🎉 Database integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if rag.db_connection:
            await rag.close_database()
            print("🔌 Database connection closed")


if __name__ == "__main__":
    print("🚀 Starting Supabase Database Integration Test")
    print("=" * 60)
    
    # Run the test
    asyncio.run(test_database_integration())
    
    print("\n" + "=" * 60)
    print("🎯 Database integration test completed!")