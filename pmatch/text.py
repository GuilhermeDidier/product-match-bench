"""Tokenization and document text.

Postgres full-text search has no Chinese parser out of the box (zhparser is
an extra extension most managed hosts don't offer), and its ts_rank is not
BM25. So we tokenize in Python — jieba for Chinese, a spec-aware splitter
for "12V1.5A" style strings — and store postings in plain tables that SQL
scores with real BM25.
"""

import re
import unicodedata

import jieba

jieba.setLogLevel(60)

# "12v1.5a" -> "12v", "1.5a"; units stay glued to their number, which is
# what makes "12v" a precise lexical term.
SPEC = re.compile(r"\d+(?:\.\d+)?(?:mah|ah|wh|ma|kg|rpm|mn·m|n·m|n\.m|kgf·cm|v|a|w|g|k|%)?")
WORD = re.compile(r"[a-z][a-z0-9\-]*")
CJK = re.compile(r"[一-鿿]+")
STOP = {"的", "和", "与", "及", "通用", "a", "an", "the", "for", "with", "and", "of"}


def tokenize(text):
    s = unicodedata.normalize("NFKC", text).lower()
    s = s.replace("°c", "c").replace("℃", "c")
    # "dc12v" / "ac100~240v": split the current-type prefix off, or WORD would
    # swallow the number and "12v" would never match
    s = re.sub(r"(?<![a-z])(dc|ac)(?=\d)", r"\1 ", s)
    tokens = []
    for chunk in re.split(r"[\s,;:/()（）\[\]|、，：]+", s):
        if not chunk:
            continue
        for m in re.finditer(rf"{CJK.pattern}|{SPEC.pattern}|{WORD.pattern}", chunk):
            t = m.group(0)
            if CJK.fullmatch(t):
                tokens += [w for w in jieba.cut_for_search(t) if w.strip()]
            else:
                tokens.append(t)
    return [t for t in tokens if t not in STOP]


def raw_text(product):
    return product["title"] + "\n" + "\n".join(f"{r['header']}: {r['value']}" for r in product["rows"])


CAT_WORDS = {
    "adapter": "power adapter power supply charger",
    "battery": "battery pack lithium battery",
    "led_driver": "led driver constant current",
    "motor": "motor",
}
MOTOR_WORDS = {"BLDC": "brushless dc bldc", "brushed_dc": "brushed dc", "stepper": "stepper"}


def g(x):
    return f"{x:g}"


def canonical_text(category, attrs):
    """English, canonical-unit rendering of the extracted attributes.

    Appending this to the raw sheet is what lets an English buyer query
    ("12V 1.5A, AU plug") match a sheet that says "1.5安 / 澳规".
    """
    parts = [CAT_WORDS.get(category, "")]
    fmt = {
        "output_voltage_v": lambda v: f"{g(v)}v output", "voltage_v": lambda v: f"{g(v)}v",
        "output_current_a": lambda v: f"{g(v)}a", "power_w": lambda v: f"{g(v)}w",
        "plug": lambda v: f"{v} plug", "cert": lambda v: f"{v} certified",
        "chemistry": lambda v: v, "capacity_mah": lambda v: f"{g(v)}mah",
        "max_discharge_a": lambda v: f"{g(v)}a max discharge", "weight_g": lambda v: f"{g(v)}g",
        "output_current_ma": lambda v: f"{g(v)}ma", "ip_rating": lambda v: v,
        "dimmable": lambda v: "dimmable" if v else "non-dimmable",
        "motor_type": lambda v: MOTOR_WORDS[v], "speed_rpm": lambda v: f"{g(v)} rpm",
        "rated_torque_nm": lambda v: f"{g(v)}n·m torque",
    }
    for field, value in attrs.items():
        if field in fmt:
            parts.append(fmt[field](value))
    return " | ".join(p for p in parts if p)
