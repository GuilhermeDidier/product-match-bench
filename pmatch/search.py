"""Candidate generation (BM25, dense), RRF fusion and cross-encoder reranking.

Fusion happens in one SQL statement so the whole hybrid retrieval is a
single round trip to Postgres; the reranker runs in Python on the top-N.
"""

import time

from sentence_transformers import CrossEncoder

from pmatch.index import connect, embed
from pmatch.text import tokenize

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
K1, B = 1.2, 0.75
RRF_K = 60

BM25_SQL = """
with q as (select distinct unnest(%(terms)s::text[]) as term),
stats as (select count(*)::float as n, avg(len)::float as avgdl from bm25_docs where field = %(field)s)
select p.doc_id,
       sum( ln(1 + (stats.n - t.df + 0.5) / (t.df + 0.5))
            * p.tf * (%(k1)s + 1)
            / (p.tf + %(k1)s * (1 - %(b)s + %(b)s * d.len / stats.avgdl)) ) as score
from q
join bm25_terms t    on t.field = %(field)s and t.term = q.term
join bm25_postings p on p.field = %(field)s and p.term = q.term
join bm25_docs d     on d.field = %(field)s and d.doc_id = p.doc_id
cross join stats
group by p.doc_id
order by score desc
limit %(k)s
"""

# {emb} is a column name chosen from a fixed allow-list, never user input.
DENSE_SQL = """
select id, 1 - ({emb} <=> %(qvec)s) as score
from products
order by {emb} <=> %(qvec)s
limit %(k)s
"""

HYBRID_SQL = """
with lexical as (
    select doc_id, row_number() over (order by score desc) as rank
    from ({bm25}) b
),
semantic as (
    select id as doc_id, row_number() over (order by dist) as rank
    from (select id, {emb} <=> %(qvec)s as dist from products order by dist limit %(k)s) s
)
select coalesce(l.doc_id, s.doc_id) as doc_id,
       coalesce(1.0 / (%(rrf_k)s + l.rank), 0) + coalesce(1.0 / (%(rrf_k)s + s.rank), 0) as rrf,
       l.rank as lexical_rank, s.rank as semantic_rank
from lexical l
full outer join semantic s on l.doc_id = s.doc_id
order by rrf desc
limit %(k)s
"""

EMB_COLUMNS = {"raw": "emb_raw", "norm": "emb_norm", "oracle": "emb_oracle"}
TEXT_COLUMNS = {"raw": "raw_text", "norm": "norm_text", "oracle": "oracle_text"}


class Searcher:
    def __init__(self, ef_search=100):
        self.conn = connect()
        self.conn.execute(f"set hnsw.ef_search = {int(ef_search)}")
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL, device="mps", max_length=512)
        return self._reranker

    def _params(self, query, field, k, lexical=True, semantic=True):
        # only compute what the SQL uses: embedding the query is most of BM25's latency otherwise
        p = {"field": field, "k": k}
        if lexical:
            p.update(terms=tokenize(query), k1=K1, b=B)
        if semantic:
            p["qvec"] = embed([query], "query")[0]
        if lexical and semantic:
            p["rrf_k"] = RRF_K
        return p

    def bm25(self, query, field="raw", k=50):
        rows = self.conn.execute(BM25_SQL, self._params(query, field, k, semantic=False)).fetchall()
        return [r[0] for r in rows]

    def dense(self, query, field="raw", k=50):
        sql = DENSE_SQL.format(emb=EMB_COLUMNS[field])
        return [r[0] for r in self.conn.execute(sql, self._params(query, field, k, lexical=False)).fetchall()]

    def hybrid(self, query, field="raw", k=50):
        sql = HYBRID_SQL.format(bm25=BM25_SQL, emb=EMB_COLUMNS[field])
        return [r[0] for r in self.conn.execute(sql, self._params(query, field, k)).fetchall()]

    def rerank(self, query, doc_ids, field="raw", top_n=30):
        head, tail = doc_ids[:top_n], doc_ids[top_n:]
        if not head:
            return doc_ids
        texts = dict(self.conn.execute(
            f"select id, {TEXT_COLUMNS[field]} from products where id = any(%s)", (head,)).fetchall())
        scores = self.reranker.predict([(query, texts[d]) for d in head], batch_size=16, show_progress_bar=False)
        ranked = [d for _, d in sorted(zip(scores, head), key=lambda x: -x[0])]
        return ranked + tail

    def run(self, variant, query, k=50):
        """variant: '<method>_<field>' e.g. 'bm25_raw', 'rrf_norm', 'rrf_norm_rerank'."""
        parts = variant.split("_")
        method, field, rerank = parts[0], parts[1], "rerank" in parts
        t0 = time.perf_counter()
        ids = {"bm25": self.bm25, "dense": self.dense, "rrf": self.hybrid}[method](query, field, k)
        t1 = time.perf_counter()
        if rerank:
            ids = self.rerank(query, ids, field)
        t2 = time.perf_counter()
        return ids, {"retrieve_ms": (t1 - t0) * 1000, "rerank_ms": (t2 - t1) * 1000}
