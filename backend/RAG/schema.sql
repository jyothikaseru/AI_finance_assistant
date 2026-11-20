-- ==============================================================================
-- FinRAG PostgreSQL (pgvector) Schema & Hybrid Search Function
-- Combines Dense Vector Search (HNSW) + Full-Text Search (tsvector / GIN)
-- ==============================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Financial Chunks Table
CREATE TABLE IF NOT EXISTS financial_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_uid TEXT UNIQUE NOT NULL,
    ticker TEXT NOT NULL,
    fiscal_year INT NOT NULL,
    report_type TEXT NOT NULL,
    chunk_type TEXT NOT NULL,       -- 'text' or 'table'
    section TEXT,
    page_start INT,
    page_end INT,
    token_count INT,
    content TEXT NOT NULL,
    source TEXT,
    embedding VECTOR(768),
    fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- 3. HNSW Index for High-Performance Dense Cosine Similarity Search
CREATE INDEX IF NOT EXISTS financial_chunks_hnsw_idx 
ON financial_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. GIN Index for Fast Full-Text Keyword Search (tsvector)
CREATE INDEX IF NOT EXISTS financial_chunks_fts_idx 
ON financial_chunks USING gin (fts);

-- 5. Standard B-Tree Indexes for Metadata Filtering
CREATE INDEX IF NOT EXISTS idx_chunks_ticker ON financial_chunks(ticker);
CREATE INDEX IF NOT EXISTS idx_chunks_year ON financial_chunks(fiscal_year);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON financial_chunks(report_type);

-- 6. Hybrid Retrieval Stored Procedure (match_financial_chunks)
-- Drop previous signatures to avoid "cannot change return type of existing function" (42P13)
DROP FUNCTION IF EXISTS match_financial_chunks(text, vector, integer, text, integer, text) CASCADE;
DROP FUNCTION IF EXISTS match_financial_chunks(text, vector, int, text, int, text) CASCADE;
DROP FUNCTION IF EXISTS match_financial_chunks CASCADE;

-- Called from Next.js API route via supabase.rpc('match_financial_chunks', ...)
CREATE OR REPLACE FUNCTION match_financial_chunks(
    query_text TEXT,
    query_embedding VECTOR(768),
    match_count INT DEFAULT 8,
    filter_ticker TEXT DEFAULT NULL,
    filter_year INT DEFAULT NULL,
    filter_type TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    chunk_uid TEXT,
    ticker TEXT,
    fiscal_year BIGINT,
    report_type TEXT,
    chunk_type TEXT,
    section TEXT,
    page_start BIGINT,
    page_end BIGINT,
    content TEXT,
    source TEXT,
    similarity DOUBLE PRECISION
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        fc.id::BIGINT,
        fc.chunk_uid::TEXT,
        fc.ticker::TEXT,
        fc.fiscal_year::BIGINT,
        fc.report_type::TEXT,
        fc.chunk_type::TEXT,
        fc.section::TEXT,
        fc.page_start::BIGINT,
        fc.page_end::BIGINT,
        fc.content::TEXT,
        fc.source::TEXT,
        -- Cosine similarity: (1 - cosine distance) cast to DOUBLE PRECISION
        (1 - (fc.embedding <=> query_embedding))::DOUBLE PRECISION AS similarity
    FROM financial_chunks fc
    WHERE
        (filter_ticker IS NULL OR fc.ticker = filter_ticker)
        AND (filter_year IS NULL OR fc.fiscal_year = filter_year)
        AND (filter_type IS NULL OR fc.report_type = filter_type)
    ORDER BY 
        -- Hybrid rank score: blends dense semantic similarity with keyword match boost
        ((1 - (fc.embedding <=> query_embedding)) + 
         CASE WHEN fc.fts @@ plainto_tsquery('english', query_text) THEN 0.15 ELSE 0 END) DESC
    LIMIT match_count;
END;
$$;
