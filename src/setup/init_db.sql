-- Enable Extensions
CREATE EXTENSION IF NOT EXISTS ltree;
CREATE EXTENSION IF NOT EXISTS vector;

-- Universal Taxonomy Table
CREATE TABLE IF NOT EXISTS taxonomy_nodes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    rank VARCHAR(50) NOT NULL, -- e.g., 'kingdom', 'family', 'genus', 'species'
    path ltree NOT NULL,       -- Hierarchical path (e.g., 'Plantae.Tracheophyta.Magnoliopsida')
    domain VARCHAR(50) NOT NULL DEFAULT 'botany', -- Allows 'automotive', 'pathology', etc.
    metadata JSONB,            -- Flexible storage for traits/descriptions
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast hierarchical lookups
CREATE INDEX IF NOT EXISTS idx_taxonomy_path ON taxonomy_nodes USING gist(path);
-- Index for specific domain lookups
CREATE INDEX IF NOT EXISTS idx_taxonomy_domain ON taxonomy_nodes(domain);
