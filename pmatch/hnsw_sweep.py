"""HNSW at scale: recall vs latency across ef_search, index build cost across
m/ef_construction, and the filtered-search trap (with and without
pgvector 0.8 iterative index scans).

Ground truth is exact k-NN computed in numpy over the same vectors, so
"recall" here means: of the true 10 nearest neighbours, how many did the
approximate index return.
"""

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from pmatch.gen_data import CATEGORIES, Q_EN, render_sheet, sample_spec
from pmatch.index import connect, embed
from pmatch.text import raw_text

ROOT = Path(__file__).resolve().parent.parent
N_QUERIES, K = 200, 10
EF_SEARCH = [10, 20, 40, 80, 160, 320]
BUILDS = [(16, 64), (32, 128)]


def corpus(n, rng):
    cats = list(CATEGORIES)
    items = []
    for i in range(n):
        cat = cats[i % len(cats)]
        spec = sample_spec(cat, rng)
        items.append((f"S{i:06d}", cat, spec, raw_text(render_sheet(f"S{i:06d}", cat, spec, rng))))
    return items


def timed_knn(conn, sql, params):
    t0 = time.perf_counter()
    rows = conn.execute(sql, params).fetchall()
    return rows, (time.perf_counter() - t0) * 1000


def main(n=100_000):
    rng = random.Random(11)
    out = {"n": n, "dim": 384}
    t0 = time.perf_counter()
    items = corpus(n, rng)
    vecs = embed([t for *_, t in items], "passage").astype(np.float32)
    out["embed_seconds"] = time.perf_counter() - t0
    print(f"embedded {n} in {out['embed_seconds']:.0f}s", file=sys.stderr)

    q_specs = [(c, sample_spec(c, rng)) for c in rng.choices(list(CATEGORIES), k=N_QUERIES)]
    q_vecs = embed([Q_EN[c](s) for c, s in q_specs], "query").astype(np.float32)
    exact = np.argsort(-(q_vecs @ vecs.T), axis=1)[:, :K]
    ids = np.array([i for i, *_ in items])

    conn = connect()
    conn.execute("drop table if exists scale_items")
    conn.execute("create table scale_items (id text primary key, category text, attrs jsonb, emb vector(384))")
    with conn.cursor() as cur, cur.copy("copy scale_items (id, category, attrs, emb) from stdin") as cp:
        for (i, c, s, _), v in zip(items, vecs):
            cp.write_row((i, c, json.dumps(s), "[" + ",".join(f"{x:.6f}" for x in v) + "]"))
    conn.execute("create index on scale_items (category)")
    conn.execute("analyze scale_items")
    conn.execute("set maintenance_work_mem = '1GB'")

    knn = "select id from scale_items order by emb <=> %s limit %s"
    out["builds"] = []
    for m, efc in BUILDS:
        conn.execute("drop index if exists scale_hnsw")
        t0 = time.perf_counter()
        conn.execute(f"create index scale_hnsw on scale_items using hnsw (emb vector_cosine_ops) "
                     f"with (m = {m}, ef_construction = {efc})")
        build_s = time.perf_counter() - t0
        size = conn.execute("select pg_relation_size('scale_hnsw')").fetchone()[0]
        sweep = []
        for ef in EF_SEARCH:
            conn.execute(f"set hnsw.ef_search = {ef}")
            recalls, lats = [], []
            for qi, qv in enumerate(q_vecs):
                rows, ms = timed_knn(conn, knn, (qv, K))
                recalls.append(len({r[0] for r in rows} & set(ids[exact[qi]])) / K)
                lats.append(ms)
            lats.sort()
            sweep.append({"ef_search": ef, "recall@10": float(np.mean(recalls)),
                          "p50_ms": lats[len(lats) // 2], "p95_ms": lats[int(len(lats) * 0.95)]})
            print(f"m={m} efc={efc} ef={ef} recall={sweep[-1]['recall@10']:.3f} p50={sweep[-1]['p50_ms']:.2f}ms",
                  file=sys.stderr)
        out["builds"].append({"m": m, "ef_construction": efc, "build_seconds": build_s,
                              "index_mb": size / 2**20, "sweep": sweep})

    # exact scan for reference
    conn.execute("set enable_indexscan = off")
    lats = sorted(timed_knn(conn, knn, (qv, K))[1] for qv in q_vecs[:50])
    out["exact_scan_p50_ms"] = lats[len(lats) // 2]
    conn.execute("set enable_indexscan = on")

    # Filtered search: category + one attribute ≈ 5% selectivity. HNSW walks the
    # graph first and filters after, so with a small ef_search it runs out of
    # candidates before it finds 10 that pass the filter.
    conn.execute("drop index if exists scale_hnsw")
    conn.execute("create index scale_hnsw on scale_items using hnsw (emb vector_cosine_ops) with (m = 16, ef_construction = 64)")
    conn.execute("set hnsw.ef_search = 40")
    # Make the planner use HNSW here; otherwise at 5% selectivity it may (rightly)
    # pick an exact scan of the filtered rows. Both plans are recorded.
    conn.execute("drop index if exists scale_items_category_idx")
    conn.execute("set enable_seqscan = off")
    filt_sql = ("select id from scale_items where category = 'motor' and attrs @> '{\"voltage_v\": 24}' "
                "order by emb <=> %s limit %s")
    mask = np.array([c == "motor" and s["voltage_v"] == 24 for _, c, s, _ in items])
    sel = float(mask.mean())
    sub_idx = np.where(mask)[0]
    motor_q = [qv for (c, _), qv in zip(q_specs, q_vecs) if c == "motor"][:50]
    filtered = {"selectivity": sel, "ef_search": 40, "modes": [],
                "plan": [re.sub(r"'\[[^\]]*\]'::vector", "'[…384 floats…]'::vector", r[0]) for r in conn.execute("explain " + filt_sql.replace("%s", "%s::vector", 1), (motor_q[0], K)).fetchall()]}
    for mode in ("off", "relaxed_order", "strict_order"):
        conn.execute(f"set hnsw.iterative_scan = {mode}")
        got, rec, lats = [], [], []
        for qv in motor_q:
            truth = set(ids[sub_idx[np.argsort(-(vecs[sub_idx] @ qv))[:K]]])
            rows, ms = timed_knn(conn, filt_sql, (qv, K))
            got.append(len(rows))
            rec.append(len({r[0] for r in rows} & truth) / K)
            lats.append(ms)
        lats.sort()
        filtered["modes"].append({"iterative_scan": mode, "avg_rows_returned": float(np.mean(got)),
                                  "recall@10": float(np.mean(rec)), "p50_ms": lats[len(lats) // 2]})
        print(f"filtered iterative_scan={mode}: rows={np.mean(got):.1f}/10 recall={np.mean(rec):.3f}", file=sys.stderr)
    conn.execute("set hnsw.iterative_scan = off")
    conn.execute("set enable_seqscan = on")
    out["filtered"] = filtered

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "hnsw.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
