-- Product matching store: one row per supplier sheet, extracted attributes
-- in JSONB, two embeddings (raw sheet vs sheet + canonical attributes) and
-- BM25 postings for the same two texts.

create extension if not exists vector;

drop table if exists bm25_postings, bm25_docs, products cascade;

create table products (
    id          text primary key,
    category    text not null,
    title       text not null,
    raw_text    text not null,          -- the sheet as the supplier wrote it
    norm_text   text not null,          -- raw_text + canonical attributes (from the LLM mapping)
    oracle_text text not null,          -- raw_text + canonical attributes from ground truth (upper bound)
    attrs       jsonb not null,         -- normalized attributes, e.g. {"output_voltage_v": 12, "plug": "EU"}
    emb_raw     vector(384) not null,
    emb_norm    vector(384) not null,
    emb_oracle  vector(384) not null
);

-- Structured filters ("EU plug, 12 V") hit attrs directly.
create index products_attrs_gin on products using gin (attrs jsonb_path_ops);
create index products_category on products (category);

-- HNSW with cosine distance; e5 embeddings are normalized.
create index products_emb_raw_hnsw    on products using hnsw (emb_raw vector_cosine_ops)    with (m = 16, ef_construction = 64);
create index products_emb_norm_hnsw   on products using hnsw (emb_norm vector_cosine_ops)   with (m = 16, ef_construction = 64);
create index products_emb_oracle_hnsw on products using hnsw (emb_oracle vector_cosine_ops) with (m = 16, ef_construction = 64);

-- BM25: explicit postings so the scoring is real BM25 and CJK tokenization is ours.
create table bm25_docs (
    field  text not null,               -- 'raw' | 'norm' | 'oracle'
    doc_id text not null references products(id),
    len    int  not null,
    primary key (field, doc_id)
);

create table bm25_postings (
    field  text not null,
    term   text not null,
    doc_id text not null,
    tf     int  not null,
    primary key (field, term, doc_id)
);

create materialized view if not exists bm25_terms as
    select field, term, count(*) as df from bm25_postings group by field, term;
