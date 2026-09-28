"""Render results/*.json into a single static page: site/index.html."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SITE = ROOT / "site"

LABELS = {
    "bm25_raw": ("BM25", "raw sheet"),
    "dense_raw": ("Dense (HNSW)", "raw sheet"),
    "rrf_raw": ("Hybrid RRF", "raw sheet"),
    "rrf_raw_rerank": ("Hybrid RRF + rerank", "raw sheet"),
    "bm25_norm": ("BM25", "sheet + extracted attrs"),
    "dense_norm": ("Dense (HNSW)", "sheet + extracted attrs"),
    "rrf_norm": ("Hybrid RRF", "sheet + extracted attrs"),
    "rrf_norm_rerank": ("Hybrid RRF + rerank", "sheet + extracted attrs"),
    "rrf_oracle_rerank": ("Hybrid RRF + rerank", "sheet + ground-truth attrs"),
}
FIELD_NAMES = {"raw": "raw sheet", "norm": "extracted", "oracle": "oracle"}

e = html.escape


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def f3(x):
    return f"{x:.3f}"


def ms(x):
    # sub-10 ms latencies keep one decimal, or BM25 would read as "0"
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


# ---------------------------------------------------------------- charts --

def reliability_svg(bins):
    W, H, P = 420, 320, 48
    lo = 0.5
    sx = lambda v: P + (v - lo) / (1 - lo) * (W - P - 16)
    sy = lambda v: H - P - (v - lo) / (1 - lo) * (H - P - 16)
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    g = [f'<line x1="{sx(t)}" y1="{sy(lo)}" x2="{sx(t)}" y2="{sy(1)}" class="grid"/>'
         f'<line x1="{sx(lo)}" y1="{sy(t)}" x2="{sx(1)}" y2="{sy(t)}" class="grid"/>'
         f'<text x="{sx(t)}" y="{H - P + 18}" class="tick" text-anchor="middle">{t:.1f}</text>'
         f'<text x="{P - 8}" y="{sy(t) + 4}" class="tick" text-anchor="end">{t:.1f}</text>' for t in ticks]
    g.append(f'<line x1="{sx(lo)}" y1="{sy(lo)}" x2="{sx(1)}" y2="{sy(1)}" class="ref"/>')
    g.append(f'<text x="{sx(0.62)}" y="{sy(0.66)}" class="note" transform="rotate(-36 {sx(0.62)} {sy(0.66)})">perfectly calibrated</text>')
    total = sum(b["n"] for b in bins)
    for b in bins:
        cx, cy = sx(max(b["mean_conf"], lo)), sy(max(b["accuracy"], lo))
        r = 4 + 10 * (b["n"] / total) ** 0.5
        g.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" class="dot s1"><title>'
                 f'confidence {b["lo"]:.2f}–{b["hi"]:.2f}: n={b["n"]}, mean confidence {b["mean_conf"]:.3f}, '
                 f'observed accuracy {b["accuracy"]:.3f}</title></circle>')
    g.append(f'<text x="{(W + P) / 2}" y="{H - 8}" class="axis" text-anchor="middle">model confidence</text>')
    g.append(f'<text x="14" y="{(H - P) / 2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(H - P) / 2})">observed accuracy</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Reliability diagram">{"".join(g)}</svg>'


def hnsw_svg(builds, exact_ms):
    W, H, P = 560, 320, 52
    pts = [(p["p50_ms"], p["recall@10"]) for b in builds for p in b["sweep"]]
    xmax = max(x for x, _ in pts) * 1.12
    ymin = min(0.8, min(y for _, y in pts) - 0.02)
    sx = lambda v: P + v / xmax * (W - P - 24)
    sy = lambda v: H - P - (v - ymin) / (1 - ymin) * (H - P - 20)
    g = []
    step = 0.05 if 1 - ymin > 0.1 else 0.02
    t = 1.0
    while t >= ymin - 1e-9:
        g.append(f'<line x1="{sx(0)}" y1="{sy(t)}" x2="{sx(xmax)}" y2="{sy(t)}" class="grid"/>'
                 f'<text x="{P - 8}" y="{sy(t) + 4}" class="tick" text-anchor="end">{t:.2f}</text>')
        t = round(t - step, 4)
    for i in range(6):
        v = xmax * i / 5
        g.append(f'<text x="{sx(v)}" y="{H - P + 18}" class="tick" text-anchor="middle">{v:.1f}</text>')
    for si, b in enumerate(builds, 1):
        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(p['p50_ms']):.1f},{sy(p['recall@10']):.1f}" for j, p in enumerate(b["sweep"]))
        g.append(f'<path d="{path}" class="line s{si}"/>')
        for p in b["sweep"]:
            g.append(f'<circle cx="{sx(p["p50_ms"]):.1f}" cy="{sy(p["recall@10"]):.1f}" r="4.5" class="dot s{si}">'
                     f'<title>m={b["m"]}, ef_construction={b["ef_construction"]}, ef_search={p["ef_search"]}: '
                     f'recall@10 {p["recall@10"]:.3f}, p50 {p["p50_ms"]:.2f} ms, p95 {p["p95_ms"]:.2f} ms</title></circle>')
        if si == 1:
            for p in b["sweep"]:
                g.append(f'<text x="{sx(p["p50_ms"]) + 7:.1f}" y="{sy(p["recall@10"]) + 14:.1f}" class="lbl">ef {p["ef_search"]}</text>')
        # series identity lives in the legend above the chart; the curves end too close together for end labels
    g.append(f'<text x="{(W + P) / 2}" y="{H - 8}" class="axis" text-anchor="middle">p50 query latency (ms) — exact scan: {exact_ms:.1f} ms</text>')
    g.append(f'<text x="14" y="{(H - P) / 2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(H - P) / 2})">recall@10 vs exact k-NN</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="HNSW recall versus latency">{"".join(g)}</svg>'


def bar(v, vmax=1.0):
    return f'<span class="bar"><span style="width:{max(v / vmax, 0) * 100:.1f}%"></span></span>'


# ---------------------------------------------------------------- page --

def main():
    ev = json.loads((RES / "eval.json").read_text())
    hn = json.loads((RES / "hnsw.json").read_text())
    ext, ret = ev["extraction"], ev["retrieval"]
    V = ret["variants"]
    products = {p["id"]: p for p in map(json.loads, (ROOT / "data" / "products.jsonl").open())}
    n_products = len(products)
    queries = [json.loads(line) for line in (ROOT / "data" / "queries.jsonl").open()]

    base, best, lex = V["bm25_raw"], V["rrf_norm_rerank"], V["bm25_norm"]

    # ablation table
    best_ndcg = max(v["ndcg@10"] for k, v in V.items() if k != "rrf_oracle_rerank")
    rows = []
    for k, v in V.items():
        name, field = LABELS[k]
        cls = ' class="oracle"' if "oracle" in k else (' class="best"' if v["ndcg@10"] == best_ndcg else "")
        rows.append(f"<tr{cls}><td>{e(name)}</td><td class='muted'>{e(field)}</td>"
                    f"<td class='num'>{f3(v['recall@50'])}</td><td class='num'>{f3(v['p@1'])}</td>"
                    f"<td class='num'>{f3(v['mrr'])}</td><td class='num nd'>{f3(v['ndcg@10'])}{bar(v['ndcg@10'])}</td>"
                    f"<td class='num'>{f3(v['ndcg@10_en'])}</td><td class='num'>{f3(v['ndcg@10_zh'])}</td>"
                    f"<td class='num'>{f3(v['misorder'])}</td><td class='num'>{ms(v['p50_ms'])}</td></tr>")
    ablation = "\n".join(rows)

    # examples
    ex_html = []
    # one English query per category, then two Chinese ones; first by id, deterministic
    pick, seen = [], set()
    for qid, x in ret["examples"].items():
        if x["lang"] == "en" and x["category"] not in seen:
            pick.append(qid)
            seen.add(x["category"])
    pick += [qid for qid, x in ret["examples"].items() if x["lang"] == "zh"][:2]
    for qid in pick:
        x = ret["examples"][qid]
        rel, hard = set(x["relevant"]), set(x["hard_negatives"])
        cols = []
        for v in ("bm25_raw", "dense_raw", "rrf_norm_rerank"):
            items = []
            for d in x["top5"][v]:
                kind = "rel" if d in rel else ("hard" if d in hard else "other")
                mark = {"rel": "✓", "hard": "✕", "other": "·"}[kind]
                items.append(f'<li class="{kind}"><span class="mk">{mark}</span><span class="t">{e(products[d]["title"])}</span></li>')
            name, field = LABELS[v]
            cols.append(f'<div class="excol"><div class="exh">{e(name)} <span class="muted">· {e(field)}</span></div><ol>{"".join(items)}</ol></div>')
        ex_html.append(f'<div class="ex"><div class="exq"><span class="lang">{x["lang"]}</span>{e(x["text"])}'
                       f'<span class="muted"> — {len(rel)} true matches, {len(hard)} near-misses in the catalog</span></div>'
                       f'<div class="excols">{"".join(cols)}</div></div>')

    # extraction errors sample
    err_rows = "".join(
        f"<tr><td>{e(r['header'])}</td><td class='mono'>{e(r['value'])}</td><td>{e(r['truth'])}</td>"
        f"<td>{e(r['pred'])}</td><td class='num'>{r['confidence']:.2f}</td></tr>" for r in ext["errors"][:10]) or (
        "<tr><td colspan='5' class='muted'>None: every unambiguous header was mapped to the right field. "
        "The only non-correct outcomes are abstentions on ambiguous headers.</td></tr>")

    b1 = hn["builds"][0]
    sweep_rows = "".join(
        f"<tr><td class='num'>{b['m']}</td><td class='num'>{b['ef_construction']}</td><td class='num'>{b['build_seconds']:.0f}s</td>"
        f"<td class='num'>{b['index_mb']:.0f} MB</td>"
        + "".join(f"<td class='num'>{p['recall@10']:.3f}</td>" for p in b["sweep"]) + "</tr>" for b in hn["builds"])
    ef_head = "".join(f"<th class='num'>ef {p['ef_search']}</th>" for p in b1["sweep"])
    filt = hn["filtered"]
    filt_rows = "".join(
        f"<tr><td class='mono'>hnsw.iterative_scan = {m['iterative_scan']}</td><td class='num'>{m['avg_rows_returned']:.1f} / 10</td>"
        f"<td class='num'>{m['recall@10']:.3f}</td><td class='num'>{m['p50_ms']:.1f} ms</td></tr>" for m in filt["modes"])

    amb = ext["ambiguous_headers"]

    src = (ROOT / "pmatch" / "search.py").read_text()
    concrete = lambda q: (q.replace("{emb}", "emb_norm").replace("%(k)s", "50").replace("%(rrf_k)s", "60")
                          .replace("%(qvec)s", ":query_embedding").replace("%(terms)s", ":query_terms")
                          .replace("%(field)s", "'norm'").replace("%(k1)s", "1.2").replace("%(b)s", "0.75"))
    hybrid_sql = concrete(src.split('HYBRID_SQL = """')[1].split('"""')[0].strip()).replace("({bm25})", "(/* BM25 subquery below */)")
    bm25_sql = concrete(src.split('BM25_SQL = """')[1].split('"""')[0].strip())
    answered_bins = ext["calibration_bins"]
    under = all(b["accuracy"] >= b["mean_conf"] for b in answered_bins)
    calib_read = ("Every bin sits on or above the diagonal: the model is under-confident, never over-confident. "
                  "Because it made no mapping errors on this set, the curve can't show where it would start to over-claim; "
                  "that needs your real sheets and a few hundred labeled rows."
                  if under else "Bins below the diagonal are ranges where the model over-claims; the auto-accept threshold goes above them.")

    page = TEMPLATE.format(
        n_products=n_products, n_queries=len(queries), n_rows=ext["rows"],
        n_scale=f"{hn['n']:,}",
        base_ndcg=f3(base["ndcg@10"]), best_ndcg=f3(best["ndcg@10"]), lex_ndcg=f3(lex["ndcg@10"]),
        rrf_norm_ndcg=f3(V["rrf_norm"]["ndcg@10"]), rrf_raw_rr_ndcg=f3(V["rrf_raw_rerank"]["ndcg@10"]),
        lex_r50=f3(lex["recall@50"]), rrf_r50=f3(V["rrf_norm"]["recall@50"]),
        best_mis=f3(best["misorder"]), base_mis=f3(base["misorder"]), lex_mis=f3(lex["misorder"]),
        best_p1=pct(best["p@1"], 0), base_p1=pct(base["p@1"], 0),
        map_acc=pct(ext["mapping_accuracy_clear_headers"]), wrong=pct(ext["wrong_rate_clear_headers"]),
        val_acc=pct(ext["value_accuracy"]), val_total=ext["value_total"],
        amb_n=amb["n"], amb_correct=pct(amb["correct"], 0), amb_abstain=pct(amb["abstain"], 0), amb_wrong=pct(amb["wrong"], 0),
        ece=f"{ext['ece']:.3f}", norm_errors=ext["normalization_errors"],
        reliability=reliability_svg(ext["calibration_bins"]),
        hnsw=hnsw_svg(hn["builds"], hn["exact_scan_p50_ms"]),
        ablation=ablation, examples="\n".join(ex_html), err_rows=err_rows,
        sweep_rows=sweep_rows, ef_head=ef_head, filt_rows=filt_rows,
        filt_sel=pct(filt["selectivity"]), filt_ef=filt["ef_search"],
        rerank_ms=f"{best['p50_ms']:.0f}", rrf_ms=f"{V['rrf_norm']['p50_ms']:.0f}",
        oracle_ndcg=f3(V["rrf_oracle_rerank"]["ndcg@10"]),
        hybrid_sql=e(hybrid_sql), bm25_sql=e(bm25_sql),
        exact_ms=f"{hn['exact_scan_p50_ms']:.1f}", calib_read=e(calib_read),
    )
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(page)
    print("wrote", SITE / "index.html")


TEMPLATE = (Path(__file__).parent / "template.html").read_text()

if __name__ == "__main__":
    main()
