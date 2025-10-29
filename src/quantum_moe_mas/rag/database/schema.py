"""
Database schema management for Supabase with pgvector extension.

This module provides schema creation, migration, and management functionality
with proper indexing for optimal vector similarity search performance.
"""

import logging
from typing import Dict, List, Optional

from quantum_moe_mas.rag.database.connection import SupabaseConnection

logger = logging.getLogger(__name__)


class DatabaseSchema:
    """
    Database schema manager for vector-enabled RAG system.
    
    Features:
    - Automatic pgvector extension setup
    - Optimized indexing for vector similarity search
    - Schema versioning and migration support
    - Performance monitoring and optimization
    """
    
    def __init__(self, connection: SupabaseConnection):
        """Initialize schema manager."""
        self.connection = connection
        self.schema_version = "1.0.0"
    
    async def initialize_schema(self) -> None:
        """Initialize complete database schema with extensions and indexes."""
        logger.info("Initializing database schema...")
        
        try:
            # Enable required extensions
            await self._enable_extensions()
            
            # Create tables
            await self._create_documents_table()
            await self._create_chunks_table()
            await self._create_embeddings_table()
            await self._create_metadata_table()
            
            # Create indexes for performance
            await self._create_indexes()
            
            # Create functions and triggers
            await self._create_functions()
            
            # Set up RLS policies
            await self._setup_rls_policies()
            
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise
    
    async def _enable_extensions(self) -> None:
        """Enable required PostgreSQL extensions."""
        extensions = [
            "vector",  # pgvector for vector operations
            "uuid-ossp",  # UUID generation
            "pg_trgm",  # Trigram matching for text search
            "btree_gin",  # GIN indexes for better performance
        ]
        
        for extension in extensions:
            try:
                await self.connection.execute_command(
                    f"CREATE EXTENSION IF NOT EXISTS {extension}"
                )
                logger.info(f"Enabled extension: {extension}")
            except Exception as e:
                logger.warning(f"Failed to enable extension {extension}: {e}")
    
    async def _create_documents_table(self) -> None:
        """Create documents table with metadata and vector support."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            content TEXT NOT NULL,
            document_type VARCHAR(50) NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            error_message TEXT,
            
            -- File metadata
            title VARCHAR(500),
            file_path TEXT,
            file_size BIGINT,
            mime_type VARCHAR(100),
            encoding VARCHAR(50),
            
            -- Content metadata
            language VARCHAR(10),
            word_count INTEGER,
            page_count INTEGER,
            duration FLOAT,
            
            -- Processing metadata
            processing_time FLOAT,
            chunk_count INTEGER DEFAULT 0,
            embedding_model VARCHAR(100),
            
            -- Document embedding (1536 dimensions for OpenAI ada-002)
            embedding VECTOR(1536),
            
            -- Relationships
            parent_id UUID REFERENCES documents(id) ON DELETE CASCADE,
            
            -- Security and access
            access_level VARCHAR(20) DEFAULT 'public',
            contains_pii BOOLEAN DEFAULT FALSE,
            
            -- Timestamps
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            processed_at TIMESTAMP WITH TIME ZONE,
            
            -- Constraints
            CONSTRAINT valid_document_type CHECK (
                document_type IN ('text', 'image', 'pdf', 'video', 'audio', 'code', 'structured')
            ),
            CONSTRAINT valid_status CHECK (
                status IN ('pending', 'processing', 'completed', 'failed', 'cached')
            ),
            CONSTRAINT valid_access_level CHECK (
                access_level IN ('public', 'private', 'restricted', 'confidential')
            )
        );
        """
        
        await self.connection.execute_command(create_table_sql)
        logger.info("Created documents table")
    
    async def _create_chunks_table(self) -> None:
        """Create document chunks table with vector embeddings."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            
            -- Chunk positioning
            start_position INTEGER,
            end_position INTEGER,
            page_number INTEGER,
            section_title VARCHAR(500),
            
            -- Semantic metadata
            semantic_type VARCHAR(50), -- paragraph, heading, table, etc.
            importance_score FLOAT CHECK (importance_score >= 0 AND importance_score <= 1),
            
            -- Vector embedding
            embedding VECTOR(1536) NOT NULL,
            
            -- Timestamps
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Constraints
            UNIQUE(document_id, chunk_index)
        );
        """
        
        await self.connection.execute_command(create_table_sql)
        logger.info("Created document_chunks table")
    
    async def _create_embeddings_table(self) -> None:
        """Create embeddings table for different models and dimensions."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS embeddings (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            content_id UUID NOT NULL, -- Can reference documents or chunks
            content_type VARCHAR(20) NOT NULL, -- 'document' or 'chunk'
            
            -- Embedding metadata
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50),
            dimensions INTEGER NOT NULL,
            
            -- Vector data (flexible dimensions)
            embedding VECTOR(1536) NOT NULL, -- Default to 1536, can be adjusted
            
            -- Performance metadata
            generation_time FLOAT,
            token_count INTEGER,
            cost DECIMAL(10, 6),
            
            -- Timestamps
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Constraints
            CONSTRAINT valid_content_type CHECK (content_type IN ('document', 'chunk')),
            UNIQUE(content_id, content_type, model_name)
        );
        """
        
        await self.connection.execute_command(create_table_sql)
        logger.info("Created embeddings table")
    
    async def _create_metadata_table(self) -> None:
        """Create flexible metadata table for custom fields."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS document_metadata (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            
            -- Flexible metadata storage
            metadata_key VARCHAR(100) NOT NULL,
            metadata_value TEXT,
            metadata_type VARCHAR(20) DEFAULT 'string', -- string, number, boolean, json
            
            -- Categorization
            category VARCHAR(50),
            tags TEXT[], -- Array of tags
            
            -- Timestamps
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Constraints
            UNIQUE(document_id, metadata_key),
            CONSTRAINT valid_metadata_type CHECK (
                metadata_type IN ('string', 'number', 'boolean', 'json', 'array')
            )
        );
        """
        
        await self.connection.execute_command(create_table_sql)
        logger.info("Created document_metadata table")
    
    async def _create_indexes(self) -> None:
        """Create optimized indexes for performance."""
        indexes = [
            # Vector similarity search indexes (HNSW for best performance)
            "CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw ON documents USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
            
            # IVFFlat indexes for larger datasets (alternative to HNSW)
            "CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
            
            # Standard B-tree indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)",
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_documents_parent_id ON documents(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_access_level ON documents(access_level)",
            
            # Chunk-specific indexes
            "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON document_chunks(document_id, chunk_index)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_semantic_type ON document_chunks(semantic_type)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_importance ON document_chunks(importance_score DESC)",
            
            # Embedding indexes
            "CREATE INDEX IF NOT EXISTS idx_embeddings_content ON embeddings(content_id, content_type)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name)",
            
            # Metadata indexes
            "CREATE INDEX IF NOT EXISTS idx_metadata_document_id ON document_metadata(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_key ON document_metadata(metadata_key)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_category ON document_metadata(category)",
            "CREATE INDEX IF NOT EXISTS idx_metadata_tags ON document_metadata USING GIN(tags)",
            
            # Full-text search indexes
            "CREATE INDEX IF NOT EXISTS idx_documents_content_fts ON documents USING GIN(to_tsvector('english', content))",
            "CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON document_chunks USING GIN(to_tsvector('english', content))",
            "CREATE INDEX IF NOT EXISTS idx_documents_title_fts ON documents USING GIN(to_tsvector('english', COALESCE(title, '')))",
            
            # Trigram indexes for fuzzy text search
            "CREATE INDEX IF NOT EXISTS idx_documents_title_trgm ON documents USING GIN(title gin_trgm_ops)",
            "CREATE INDEX IF NOT EXISTS idx_documents_content_trgm ON documents USING GIN(content gin_trgm_ops)",
        ]
        
        for index_sql in indexes:
            try:
                await self.connection.execute_command(index_sql)
                logger.debug(f"Created index: {index_sql.split()[5]}")
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
        
        logger.info("Created database indexes")
    
    async def _create_functions(self) -> None:
        """Create database functions and triggers."""
        
        # Function to update updated_at timestamp
        update_timestamp_function = """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """
        
        await self.connection.execute_command(update_timestamp_function)
        
        # Triggers for automatic timestamp updates
        triggers = [
            "CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()",
            "CREATE TRIGGER update_metadata_updated_at BEFORE UPDATE ON document_metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()",
        ]
        
        for trigger_sql in triggers:
            try:
                await self.connection.execute_command(trigger_sql)
            except Exception as e:
                logger.warning(f"Failed to create trigger: {e}")
        
        # Vector similarity search function
        similarity_function = """
        CREATE OR REPLACE FUNCTION vector_similarity_search(
            query_embedding VECTOR(1536),
            similarity_threshold FLOAT DEFAULT 0.7,
            max_results INTEGER DEFAULT 10,
            document_types TEXT[] DEFAULT NULL
        )
        RETURNS TABLE(
            document_id UUID,
            chunk_id UUID,
            content TEXT,
            similarity FLOAT,
            document_type VARCHAR(50)
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                d.id as document_id,
                c.id as chunk_id,
                c.content,
                1 - (c.embedding <=> query_embedding) as similarity,
                d.document_type
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE 
                (document_types IS NULL OR d.document_type = ANY(document_types))
                AND (1 - (c.embedding <=> query_embedding)) >= similarity_threshold
                AND d.status = 'completed'
            ORDER BY c.embedding <=> query_embedding
            LIMIT max_results;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        await self.connection.execute_command(similarity_function)
        
        logger.info("Created database functions and triggers")
    
    async def _setup_rls_policies(self) -> None:
        """Set up Row Level Security policies for data protection."""
        
        # Enable RLS on tables
        rls_commands = [
            "ALTER TABLE documents ENABLE ROW LEVEL SECURITY",
            "ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY", 
            "ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY",
            "ALTER TABLE document_metadata ENABLE ROW LEVEL SECURITY",
        ]
        
        for command in rls_commands:
            try:
                await self.connection.execute_command(command)
            except Exception as e:
                logger.warning(f"Failed to enable RLS: {e}")
        
        # Create basic policies (can be customized based on requirements)
        policies = [
            # Allow authenticated users to read public documents
            """CREATE POLICY documents_read_policy ON documents 
               FOR SELECT USING (access_level = 'public' OR auth.role() = 'authenticated')""",
            
            # Allow authenticated users to insert documents
            """CREATE POLICY documents_insert_policy ON documents 
               FOR INSERT WITH CHECK (auth.role() = 'authenticated')""",
            
            # Allow users to update their own documents
            """CREATE POLICY documents_update_policy ON documents 
               FOR UPDATE USING (auth.uid()::text = metadata->>'created_by')""",
            
            # Chunks inherit document permissions
            """CREATE POLICY chunks_read_policy ON document_chunks 
               FOR SELECT USING (
                   EXISTS (
                       SELECT 1 FROM documents d 
                       WHERE d.id = document_chunks.document_id 
                       AND (d.access_level = 'public' OR auth.role() = 'authenticated')
                   )
               )""",
        ]
        
        for policy_sql in policies:
            try:
                await self.connection.execute_command(policy_sql)
            except Exception as e:
                logger.warning(f"Failed to create RLS policy: {e}")
        
        logger.info("Set up Row Level Security policies")
    
    async def get_schema_info(self) -> Dict[str, any]:
        """Get comprehensive schema information."""
        try:
            # Get table information
            tables_query = """
            SELECT 
                table_name,
                table_type,
                is_insertable_into,
                is_typed
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('documents', 'document_chunks', 'embeddings', 'document_metadata')
            ORDER BY table_name
            """
            
            tables = await self.connection.execute_query(tables_query)
            
            # Get index information
            indexes_query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes 
            WHERE schemaname = 'public'
            AND tablename IN ('documents', 'document_chunks', 'embeddings', 'document_metadata')
            ORDER BY tablename, indexname
            """
            
            indexes = await self.connection.execute_query(indexes_query)
            
            # Get extension information
            extensions_query = """
            SELECT 
                extname,
                extversion,
                extrelocatable
            FROM pg_extension 
            WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm', 'btree_gin')
            """
            
            extensions = await self.connection.execute_query(extensions_query)
            
            return {
                "schema_version": self.schema_version,
                "tables": [dict(row) for row in tables],
                "indexes": [dict(row) for row in indexes],
                "extensions": [dict(row) for row in extensions],
                "total_tables": len(tables),
                "total_indexes": len(indexes),
            }
            
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {"error": str(e)}
    
    async def optimize_performance(self) -> None:
        """Run performance optimization commands."""
        optimization_commands = [
            # Update table statistics
            "ANALYZE documents",
            "ANALYZE document_chunks", 
            "ANALYZE embeddings",
            "ANALYZE document_metadata",
            
            # Vacuum tables to reclaim space
            "VACUUM documents",
            "VACUUM document_chunks",
            "VACUUM embeddings", 
            "VACUUM document_metadata",
        ]
        
        for command in optimization_commands:
            try:
                await self.connection.execute_command(command)
                logger.debug(f"Executed optimization: {command}")
            except Exception as e:
                logger.warning(f"Optimization command failed: {e}")
        
        logger.info("Database performance optimization completed")
    
    async def drop_schema(self) -> None:
        """Drop all schema objects (use with caution!)."""
        logger.warning("Dropping database schema...")
        
        drop_commands = [
            "DROP TABLE IF EXISTS document_metadata CASCADE",
            "DROP TABLE IF EXISTS embeddings CASCADE", 
            "DROP TABLE IF EXISTS document_chunks CASCADE",
            "DROP TABLE IF EXISTS documents CASCADE",
            "DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE",
            "DROP FUNCTION IF EXISTS vector_similarity_search(VECTOR, FLOAT, INTEGER, TEXT[]) CASCADE",
        ]
        
        for command in drop_commands:
            try:
                await self.connection.execute_command(command)
            except Exception as e:
                logger.warning(f"Failed to drop object: {e}")
        
        logger.warning("Database schema dropped")