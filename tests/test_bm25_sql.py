"""The SQL BM25 must agree with a straightforward Python BM25 over the same postings."""
import math
from collections import Counter

import psycopg
import pytest

from pmatch.index import DSN
from pmatch.search import B, BM25_SQL, K1
from pmatch.text import tokenize

QUERIES = ["12V 1.5A power adapter, AU plug, CE certified", "磷酸铁锂电池 12.8V 6000mAh", "24V brushless DC motor 3000 rpm"]


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(DSN)
    except psycopg.OperationalError:
        pytest.skip("no database")
    if not c.execute("select to_regclass('bm25_docs')").fetchone()[0]:
        pytest.skip("index not built")
    yield c
    c.close()


def python_bm25(conn, query, field="norm"):
    docs = {d: n for d, n in conn.execute("select doc_id, len from bm25_docs where field=%s", (field,))}
    avgdl, n = sum(docs.values()) / len(docs), len(docs)
    scores = Counter()
    for term in set(tokenize(query)):
        post = conn.execute("select doc_id, tf from bm25_postings where field=%s and term=%s", (field, term)).fetchall()
        idf = math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
        for d, tf in post:
            scores[d] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * docs[d] / avgdl))
    return scores


@pytest.mark.parametrize("query", QUERIES)
def test_sql_matches_reference(conn, query):
    ref = python_bm25(conn, query)
    rows = conn.execute(BM25_SQL, {"terms": tokenize(query), "field": "norm", "k": 20, "k1": K1, "b": B}).fetchall()
    assert rows
    for doc_id, score in rows:
        assert score == pytest.approx(ref[doc_id], rel=1e-9)
    assert [r[0] for r in rows][:5] == [d for d, _ in ref.most_common(5)]
