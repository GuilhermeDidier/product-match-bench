"""Evaluation harness: extraction quality + calibration, and a retrieval ablation.

Everything is scored against the generator's ground truth, never eyeballed.
"""

import json
import math
import statistics
from pathlib import Path

from pmatch.normalize import same
from pmatch.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"

VARIANTS = [
    "bm25_raw", "dense_raw", "rrf_raw", "rrf_raw_rerank",
    "bm25_norm", "dense_norm", "rrf_norm", "rrf_norm_rerank",
    "rrf_oracle_rerank",
]
BINS = [0.0, 0.8, 0.9, 0.95, 0.98, 1.0001]


# ------------------------------------------------------------ extraction --

def evaluate_extraction(products, extractions):
    rows_out, field_hits, field_total = [], 0, 0
    for p in products:
        e = extractions[p["id"]]
        pred = {m["index"]: m for m in e["mapping"]["rows"]}
        for i, r in enumerate(p["rows"]):
            m = pred.get(i, {"field": "unmapped", "confidence": 0.0})
            truth = r["field"] or "unmapped"
            outcome = "abstain" if m["field"] == "ambiguous" else ("correct" if m["field"] == truth else "wrong")
            rows_out.append({"product": p["id"], "header": r["header"], "value": r["value"], "truth": truth,
                             "pred": m["field"], "confidence": float(m["confidence"]),
                             "ambiguous_header": r["ambiguous"], "outcome": outcome})
        for field, value in p["truth"].items():
            if any(r["field"] == field for r in p["rows"]):
                field_total += 1
                field_hits += field in e["attributes"] and same(e["attributes"][field], value)

    def rate(rs, what):
        return sum(r["outcome"] == what for r in rs) / max(len(rs), 1)

    amb = [r for r in rows_out if r["ambiguous_header"]]
    clear = [r for r in rows_out if not r["ambiguous_header"]]
    answered = [r for r in rows_out if r["outcome"] != "abstain"]

    bins = []
    for lo, hi in zip(BINS, BINS[1:]):
        b = [r for r in answered if lo <= r["confidence"] < hi]
        if b:
            bins.append({"lo": lo, "hi": min(hi, 1.0), "n": len(b),
                         "mean_conf": statistics.mean(r["confidence"] for r in b),
                         "accuracy": rate(b, "correct")})
    ece = sum(b["n"] * abs(b["mean_conf"] - b["accuracy"]) for b in bins) / max(len(answered), 1)

    return {
        "rows": len(rows_out),
        "mapping_accuracy_clear_headers": rate(clear, "correct"),
        "wrong_rate_clear_headers": rate(clear, "wrong"),
        "ambiguous_headers": {"n": len(amb), "correct": rate(amb, "correct"), "abstain": rate(amb, "abstain"),
                              "wrong": rate(amb, "wrong")},
        "value_accuracy": field_hits / field_total,
        "value_total": field_total,
        "normalization_errors": sum(len(e["normalization_errors"]) for e in extractions.values()),
        "calibration_bins": bins,
        "ece": ece,
        "errors": [r for r in rows_out if r["outcome"] == "wrong"][:40],
    }


# ------------------------------------------------------------- retrieval --

def metrics(ranked, relevant, hard_negatives):
    rel, hard = set(relevant), set(hard_negatives)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    first = next((i for i, d in enumerate(ranked) if d in rel), None)
    return {
        "recall@50": len(rel & set(ranked[:50])) / len(rel),
        "recall@10": len(rel & set(ranked[:10])) / len(rel),
        "p@1": float(bool(ranked) and ranked[0] in rel),
        "p@5": len(rel & set(ranked[:5])) / 5,
        "ndcg@10": dcg / idcg,
        "mrr": 0.0 if first is None else 1 / (first + 1),
        # share of (true match, near-miss) pairs where the near-miss is ranked higher;
        # anything outside the returned list counts as ranked last
        "misorder": misorder(ranked, rel, hard),
    }


def misorder(ranked, rel, hard):
    pos = {d: i for i, d in enumerate(ranked)}
    last = len(ranked)
    pairs = [(pos.get(r, last), pos.get(h, last)) for r in rel for h in hard]
    pairs = [(r, h) for r, h in pairs if not (r == last and h == last)]
    return sum(h < r for r, h in pairs) / len(pairs) if pairs else 0.0


def evaluate_retrieval(queries):
    s = Searcher()
    s.reranker  # load once, outside the timings
    s.run("rrf_raw_rerank", queries[0]["text"])  # warm up MPS + caches
    table, per_query, examples = {}, {}, {}
    for v in VARIANTS:
        ms, lat = [], []
        for q in queries:
            ids, t = s.run(v, q["text"])
            m = metrics(ids, q["relevant"], q["hard_negatives"])
            m["lang"] = q["lang"]
            ms.append(m)
            lat.append(t["retrieve_ms"] + t["rerank_ms"])
            per_query.setdefault(q["id"], {})[v] = ids[:5]
        keys = [k for k in ms[0] if k != "lang"]
        row = {k: statistics.mean(m[k] for m in ms) for k in keys}
        for lang in ("en", "zh"):
            sub = [m for m in ms if m["lang"] == lang]
            row[f"ndcg@10_{lang}"] = statistics.mean(m["ndcg@10"] for m in sub)
        lat.sort()
        row["p50_ms"], row["p95_ms"] = lat[len(lat) // 2], lat[int(len(lat) * 0.95)]
        table[v] = row
        print(f"{v:20s} " + "  ".join(f"{k}={row[k]:.3f}" for k in ("recall@50", "p@1", "ndcg@10", "mrr", "misorder")))
    for q in queries:
        examples[q["id"]] = {"text": q["text"], "lang": q["lang"], "category": q["category"], "relevant": q["relevant"],
                             "hard_negatives": q["hard_negatives"], "top5": per_query[q["id"]]}
    return {"variants": table, "n_queries": len(queries), "examples": examples}


def main():
    products = [json.loads(line) for line in (DATA / "products.jsonl").open()]
    queries = [json.loads(line) for line in (DATA / "queries.jsonl").open()]
    extractions = {e["id"]: e for e in map(json.loads, (DATA / "extractions.jsonl").open())}
    RESULTS.mkdir(exist_ok=True)
    ext = evaluate_extraction(products, extractions)
    print(json.dumps({k: v for k, v in ext.items() if k not in ("errors", "calibration_bins")}, indent=1))
    ret = evaluate_retrieval(queries)
    (RESULTS / "eval.json").write_text(json.dumps({"extraction": ext, "retrieval": ret}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
