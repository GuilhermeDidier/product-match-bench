"""LLM header mapping + deterministic normalization.

The model sees a supplier sheet and answers one question per row: which
canonical field does this header mean? It may say "unmapped" (not in our
schema) or "ambiguous" (can't tell) instead of guessing, and it gives a
confidence we later check against ground truth (calibration).

Values are never produced by the model. normalize.py parses them.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from pmatch.normalize import FIELDS, NormalizationError, normalize

DATA = Path(__file__).resolve().parent.parent / "data"
MODEL = "claude-opus-5"
CATS = ["adapter", "battery", "led_driver", "motor"]

FIELD_DOCS = {
    "output_voltage_v": "adapter DC output voltage",
    "output_current_a": "adapter output current",
    "power_w": "rated/output power (adapter or LED driver)",
    "input_voltage_v": "AC input voltage range",
    "plug": "plug standard (US/EU/UK/AU/CN)",
    "cert": "safety certification (CE, UL, CCC/3C, FCC)",
    "moq_units": "minimum order quantity",
    "chemistry": "battery cell chemistry",
    "voltage_v": "nominal/rated voltage of a battery or motor",
    "capacity_mah": "battery capacity (any unit: mAh, Ah, Wh)",
    "max_discharge_a": "battery max continuous discharge current",
    "weight_g": "weight",
    "op_temp_c": "operating temperature range",
    "output_current_ma": "LED driver constant output current",
    "ip_rating": "ingress protection rating",
    "dimmable": "whether the LED driver is dimmable",
    "motor_type": "motor type (brushless, brushed, stepper)",
    "speed_rpm": "motor rated/no-load speed",
    "rated_torque_nm": "motor rated torque",
    "no_load_current_a": "motor no-load current",
}

SYSTEM = f"""You map supplier spec-sheet headers (often Chinese) to a canonical product schema.

Canonical fields:
{chr(10).join(f"- {k}: {v}" for k, v in FIELD_DOCS.items())}

For every row, return exactly one of:
- a canonical field name,
- "unmapped" when the row is real information our schema does not cover (packaging, origin, lead time, model number...),
- "ambiguous" when the header could mean two different canonical fields and you cannot tell which with confidence.

Use the value and the product category as evidence, not only the header text. Never map two rows to the same field.
confidence is your probability (0-1) that the answer is correct. Be calibrated: 0.9 should be right about 9 times in 10."""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATS},
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "field": {"type": "string", "enum": FIELDS + ["unmapped", "ambiguous"]},
                    "confidence": {"type": "number"},
                },
                "required": ["index", "field", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["category", "rows"],
    "additionalProperties": False,
}

client = anthropic.Anthropic()


def map_headers(product):
    sheet = "\n".join(f"[{i}] {r['header']}: {r['value']}" for i, r in enumerate(product["rows"]))
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"Title: {product['title']}\n\n{sheet}"}],
    )
    if msg.stop_reason != "end_turn":
        raise RuntimeError(f"{product['id']}: stop_reason={msg.stop_reason}")
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text), msg.usage


def build_attributes(product, mapping):
    """Normalize mapped rows into a JSONB-ready dict. Voltage first: Wh capacity needs it."""
    by_field = {}
    for m in mapping["rows"]:
        if m["field"] in ("unmapped", "ambiguous") or not 0 <= m["index"] < len(product["rows"]):
            continue
        prev = by_field.get(m["field"])
        if prev is None or m["confidence"] > prev["confidence"]:
            by_field[m["field"]] = m
    attrs, errors = {}, []
    for field in sorted(by_field, key=lambda f: f != "voltage_v"):
        raw = product["rows"][by_field[field]["index"]]["value"]
        try:
            attrs[field] = normalize(field, raw, attrs)
        except NormalizationError as e:
            errors.append({"field": field, "raw": raw, "error": str(e)})
    return attrs, errors


def main(limit=None, workers=8):
    products = [json.loads(line) for line in (DATA / "products.jsonl").open()][:limit]
    out_path = DATA / "extractions.jsonl"
    done = {}
    if out_path.exists():
        done = {json.loads(line)["id"]: 1 for line in out_path.open()}
    todo = [p for p in products if p["id"] not in done]
    print(f"{len(done)} cached, {len(todo)} to extract", file=sys.stderr)
    tokens = [0, 0]
    with ThreadPoolExecutor(workers) as pool, out_path.open("a") as out:
        futures = {pool.submit(map_headers, p): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                mapping, usage = fut.result()
            except Exception as e:  # keep going; failures are reported, not hidden
                print(f"FAIL {p['id']}: {e}", file=sys.stderr)
                continue
            tokens[0] += usage.input_tokens
            tokens[1] += usage.output_tokens
            attrs, errors = build_attributes(p, mapping)
            out.write(json.dumps({"id": p["id"], "mapping": mapping, "attributes": attrs,
                                  "normalization_errors": errors}, ensure_ascii=False) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}  in={tokens[0]} out={tokens[1]}", file=sys.stderr)
    print(f"done. tokens in={tokens[0]} out={tokens[1]}", file=sys.stderr)


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
