"""Load products, extracted attributes, embeddings and BM25 postings into Postgres."""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from pmatch.text import canonical_text, raw_text, tokenize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DSN = os.environ.get("PMATCH_DSN", "dbname=pmatch")
EMBED_MODEL = "intfloat/multilingual-e5-small"

_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device="mps")
    return _embedder


def embed(texts, kind):
    # e5 was trained with these prefixes; dropping them costs recall
    prefix = "query: " if kind == "query" else "passage: "
    return embedder().encode([prefix + t for t in texts], batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False)


def connect():
    conn = psycopg.connect(DSN, autocommit=True)
    register_vector(conn)
    return conn


def main():
    products = [json.loads(line) for line in (DATA / "products.jsonl").open()]
    extractions = {e["id"]: e for e in map(json.loads, (DATA / "extractions.jsonl").open())}
    missing = [p["id"] for p in products if p["id"] not in extractions]
    if missing:
        raise SystemExit(f"{len(missing)} products have no extraction yet, e.g. {missing[:3]}")

    rows = []
    for p in products:
        ext = extractions[p["id"]]
        attrs = ext["attributes"]
        raw = raw_text(p)
        rows.append({
            # the extracted pipeline sees only what the model extracted, category included;
            # ground truth feeds the oracle text and nothing else
            "id": p["id"], "category": ext["mapping"]["category"], "title": p["title"], "raw": raw,
            "norm": raw + "\n" + canonical_text(ext["mapping"]["category"], attrs),
            "oracle": raw + "\n" + canonical_text(p["category"], p["truth"]),
            "attrs": attrs,
        })

    vecs = {k: embed([r[k] for r in rows], "passage") for k in ("raw", "norm", "oracle")}

    with connect() as conn:
        conn.execute((ROOT / "schema.sql").read_text())
        with conn.cursor() as cur:
            cur.executemany(
                "insert into products values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(r["id"], r["category"], r["title"], r["raw"], r["norm"], r["oracle"], json.dumps(r["attrs"]),
                  vecs["raw"][i], vecs["norm"][i], vecs["oracle"][i]) for i, r in enumerate(rows)])
            for field in ("raw", "norm", "oracle"):
                docs, postings = [], []
                for r in rows:
                    toks = tokenize(r[field])
                    docs.append((field, r["id"], len(toks)))
                    postings += [(field, t, r["id"], tf) for t, tf in Counter(toks).items()]
                cur.executemany("insert into bm25_docs values (%s,%s,%s)", docs)
                cur.executemany("insert into bm25_postings values (%s,%s,%s,%s)", postings)
        conn.execute("refresh materialized view bm25_terms")
        conn.execute("analyze")
        n = conn.execute("select count(*) from products").fetchone()[0]
        p = conn.execute("select count(*) from bm25_postings").fetchone()[0]
    print(f"indexed {n} products, {p} postings, embeddings {np.asarray(vecs['raw']).shape}")


if __name__ == "__main__":
    main()
