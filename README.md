# Product matching over Chinese supplier sheets

[![CI](https://github.com/GuilhermeDidier/product-match-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/GuilhermeDidier/product-match-bench/actions/workflows/ci.yml)

A small, end-to-end prototype of hybrid retrieval with LLM extraction on Postgres + pgvector, where every layer is scored against ground truth.

**Report:** https://product-match-bench.vercel.app — built from the JSON files in `results/` by `report/build.py`.

```
supplier sheet (电池容量: 2.6Ah, 插头: 欧规, 起订量: 1万个 …)
   │
   ├─ 1. Extract & normalize   LLM maps header → canonical field | "ambiguous" | "unmapped"  (strict JSON schema)
   │                           code parses values + units → JSONB   (2.6Ah / 2600毫安时 / 9.6Wh → 2600 mAh)
   ├─ 2. Candidates            BM25 in SQL over jieba + spec-aware tokens   +   multilingual-e5-small in pgvector HNSW
   ├─ 3. Fusion & rerank       Reciprocal Rank Fusion (k=60) in one SQL statement → bge-reranker-v2-m3 on top 30
   └─ 4. Evaluation            gold relevance + hard negatives, ablations, per-language, calibration, HNSW sweep
```

## Results

526 synthetic supplier sheets across 4 categories, 104 buyer queries (70% English, 30% Chinese). Each query has 3.5 true matches on average and ~15 **near-misses**: products that match on every requested attribute but one.

| Candidate generation | Indexed text | Recall@50 | P@1 | nDCG@10 | Misorder ↓ |
|---|---|---:|---:|---:|---:|
| BM25 | raw sheet | 0.917 | 0.394 | 0.422 | 0.288 |
| Dense (HNSW) | raw sheet | 0.908 | 0.337 | 0.375 | 0.313 |
| Hybrid RRF | raw sheet | 0.940 | 0.337 | 0.405 | 0.279 |
| Hybrid RRF + rerank | raw sheet | 0.940 | 0.875 | 0.803 | 0.154 |
| **BM25** | **sheet + extracted attrs** | 0.981 | **0.971** | **0.945** | **0.046** |
| Dense (HNSW) | sheet + extracted attrs | 0.959 | 0.423 | 0.477 | 0.269 |
| Hybrid RRF | sheet + extracted attrs | **0.993** | 0.663 | 0.726 | 0.167 |
| **Hybrid RRF + rerank** | **sheet + extracted attrs** | **0.993** | 0.923 | 0.934 | **0.046** |
| *Hybrid RRF + rerank* | *sheet + ground-truth attrs (upper bound)* | *0.993* | *0.933* | *0.941* | *0.043* |

*Misorder* is the share of (true match, near-miss) pairs where the near-miss ranks higher.

What the table says:

- **Extraction is the biggest lever.** Indexing the normalized attributes next to the raw sheet takes BM25 from 0.422 to 0.945 nDCG@10. The extracted pipeline lands within 0.007 of the ground-truth upper bound.
- **Equal-weight RRF hurt ranking on this query set.** Dense retrieval adds recall (0.981 → 0.993), but it is weak on exact specs, and fusing it at equal weight drops nDCG to 0.726. Hybrid has to be measured per query type, not assumed.
- **The cross-encoder makes fusion worth it.** It restores the ranking and keeps the extra recall. It also costs ~1 s per query on an M4 (MPS), which is the next thing to optimize (smaller reranker, GPU, or rerank only when the lexical and dense lists disagree).

**Extraction** (4,594 header rows, Claude with strict JSON schema output): 100% of unambiguous headers mapped correctly. Of the 128 deliberately ambiguous ones (电流, 电压), 80% were mapped correctly, 20% were abstained on and 0% were wrong. 99.3% of canonical values came out correct after normalization; the misses are the abstentions. ECE 0.053: the model is under-confident, never over-confident, on this set.

**HNSW at 100,000 vectors** (384-d, cosine, 200 queries, recall against exact k-NN):

| m | ef_construction | build | ef 10 | ef 20 | ef 40 | ef 80 | ef 160 | ef 320 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 9 s | 0.788 | 0.870 | 0.932 | 0.957 | 0.983 | 0.993 |
| 32 | 128 | 24 s | 0.941 | 0.986 | 0.995 | 0.999 | 0.999 | 1.000 |

**Filtered search** (`category = 'motor' AND attrs @> '{"voltage_v": 24}'`, 4.9% of rows, ef_search = 40): with `hnsw.iterative_scan = off` the index returns **1.4 of 10 rows** (recall 0.144). With `relaxed_order` it returns 10/10 (recall 0.996).

## Honest limits

- The catalog is **synthetic**, generated with known ground truth (`pmatch/gen_data.py`). It imitates real sheets (header aliases, unit variants, 万 multipliers, ambiguous headers), but real data will have failure modes this doesn't.
- Queries are spec-style, which favors lexical matching. Paraphrased queries would shift weight toward dense retrieval.
- BM25 is implemented in SQL for transparency and CJK control. In production we'd benchmark ParadeDB `pg_search` / `pg_textsearch` against it.
- Queries are not yet parsed into JSONB filters. That's the obvious next step, and the misorder column is the metric it would move.

## Run it

Requires Postgres 16 with pgvector ≥ 0.8 and an Anthropic API key.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
createdb pmatch && psql -d pmatch -c "create extension vector"

.venv/bin/python -m pmatch.gen_data          # catalog + queries with ground truth
.venv/bin/python -m pmatch.extract           # LLM header mapping (resumable), ~$6
.venv/bin/python -m pmatch.index             # embeddings, HNSW, BM25 postings
.venv/bin/python -m pmatch.evaluate          # ablation + extraction/calibration → results/eval.json
.venv/bin/python -m pmatch.hnsw_sweep        # 100k-vector sweep → results/hnsw.json (~10 min)
.venv/bin/python report/build.py             # → site/index.html
.venv/bin/python -m pytest tests             # normalizer round-trip, SQL BM25 == reference BM25
```

## Layout

| Path | What |
|---|---|
| `pmatch/gen_data.py` | Synthetic catalog, families of duplicates and near-misses, queries with gold labels |
| `pmatch/normalize.py` | Deterministic unit and enum normalization, no LLM |
| `pmatch/extract.py` | LLM header mapping with strict JSON schema, abstention and confidence |
| `pmatch/text.py` | jieba + spec-aware tokenizer; canonical attribute rendering |
| `schema.sql` | Products with JSONB attrs, 3 HNSW indexes, BM25 postings |
| `pmatch/search.py` | BM25 SQL, dense SQL, hybrid RRF SQL, cross-encoder rerank |
| `pmatch/evaluate.py` | Metrics, ablation, calibration |
| `pmatch/hnsw_sweep.py` | ef_search / m sweep and the filtered-search experiment |
