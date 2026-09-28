"""Deterministic value normalization. No LLM in here, on purpose.

The LLM decides *which canonical field* a supplier header means. Turning
"2.6Ah", "2600毫安时" or "9.6Wh" into 2600 mAh is arithmetic, and arithmetic
should be code: testable, repeatable and free.
"""

import re

NUM = r"[-+]?\d+(?:\.\d+)?"

UNITS = {
    "voltage": {"v": 1, "伏": 1, "kv": 1000, "mv": 0.001},
    "current": {"a": 1, "安": 1, "ma": 0.001, "毫安": 0.001},
    "current_ma": {"ma": 1, "毫安": 1, "a": 1000, "安": 1000},
    "power": {"w": 1, "瓦": 1, "kw": 1000},
    "mass": {"g": 1, "克": 1, "kg": 1000, "公斤": 1000},
    "speed": {"rpm": 1, "r/min": 1, "转/分": 1, "转": 1},
    "torque": {"n·m": 1, "n.m": 1, "nm": 1, "mn·m": 0.001, "kgf·cm": 0.0980665, "kg·cm": 0.0980665, "kg.cm": 0.0980665},
}

FIELD_KIND = {
    "output_voltage_v": "voltage", "voltage_v": "voltage", "input_voltage_v": "voltage_range",
    "output_current_a": "current", "max_discharge_a": "current", "no_load_current_a": "current",
    "output_current_ma": "current_ma", "power_w": "power", "weight_g": "mass",
    "speed_rpm": "speed", "rated_torque_nm": "torque", "capacity_mah": "capacity",
    "op_temp_c": "temp_range", "moq_units": "count",
    "plug": "enum", "cert": "enum", "chemistry": "enum", "motor_type": "enum",
    "ip_rating": "ip", "dimmable": "bool",
}

ENUMS = {
    "plug": {"US": ["美规", "us", "美标插头", "美标"], "EU": ["欧规", "eu", "欧标插头", "欧标"],
             "UK": ["英规", "uk", "英标插头", "英标"], "AU": ["澳规", "au", "澳标"], "CN": ["国标", "cn", "中规"]},
    "cert": {"CE": ["ce认证", "ce"], "UL": ["ul认证", "ul"], "CCC": ["3c认证", "ccc", "3c"], "FCC": ["fcc认证", "fcc"]},
    "chemistry": {"LiFePO4": ["磷酸铁锂", "lifepo4", "磷酸铁锂 lfp", "lfp"], "Li-ion": ["三元锂", "锂离子", "li-ion", "nmc"],
                  "LiPo": ["聚合物锂", "锂聚合物", "lipo"]},
    "motor_type": {"BLDC": ["无刷直流电机", "bldc", "无刷", "brushless dc"], "brushed_dc": ["有刷直流电机", "有刷", "brushed dc"],
                   "stepper": ["步进电机", "stepper", "步进"]},
}

FIELDS = list(FIELD_KIND)


class NormalizationError(ValueError):
    pass


def _clean(s):
    return s.strip().lower().replace(" ", "").replace("约", "")


def _number_with_unit(raw, table):
    s = re.sub(r"^(dc|ac)", "", _clean(raw))
    # longest unit first so "mah" never matches "a" and "kgf·cm" beats "g"
    for unit in sorted(table, key=len, reverse=True):
        m = re.fullmatch(rf"({NUM}){re.escape(unit)}", s)
        if m:
            return float(m.group(1)) * table[unit]
    raise NormalizationError(f"no unit match for {raw!r}")


def _range(raw):
    nums = [float(x) for x in re.findall(NUM, raw.replace("+", ""))]
    # "100-240V": the dash is a separator, not a minus sign
    if re.search(r"\d\s*-\s*\d", raw) and len(nums) == 2 and nums[1] < 0:
        nums[1] = -nums[1]
    if len(nums) < 2:
        raise NormalizationError(f"not a range: {raw!r}")
    return [nums[0], nums[1]]


def normalize(field, raw, context=None):
    """Canonical value for `field` from the raw supplier string.

    `context` holds already-normalized fields of the same product; capacity
    given in Wh needs the nominal voltage to become mAh.
    """
    kind = FIELD_KIND[field]
    s = _clean(raw)
    if kind in UNITS:
        return round(_number_with_unit(raw, UNITS[kind]), 4)
    if kind == "capacity":
        for unit, mult in (("mah", 1), ("毫安时", 1), ("ah", 1000)):
            m = re.fullmatch(rf"({NUM}){unit}", s)
            if m:
                return round(float(m.group(1)) * mult, 2)
        m = re.fullmatch(rf"({NUM})wh", s)
        if m:
            volts = (context or {}).get("voltage_v")
            if not volts:
                raise NormalizationError("capacity in Wh needs voltage_v")
            return round(float(m.group(1)) / volts * 1000, -1)   # Wh printed at 2 decimals
        raise NormalizationError(f"bad capacity {raw!r}")
    if kind == "voltage_range":
        return _range(raw.split("50/60")[0])
    if kind == "temp_range":
        return _range(raw.replace("至", "~").replace("to", "~"))
    if kind == "count":
        m = re.fullmatch(rf"({NUM})(万|k)?(个|pcs|units)?", s)
        if not m:
            raise NormalizationError(f"bad count {raw!r}")
        mult = {"万": 10000, "k": 1000}.get(m.group(2), 1)
        return int(round(float(m.group(1)) * mult))
    if kind == "enum":
        for canon, syns in ENUMS[field].items():
            if s in [x.replace(" ", "") for x in syns]:
                return canon
        raise NormalizationError(f"unknown {field} {raw!r}")
    if kind == "ip":
        m = re.search(r"ip(\d\d)", s)
        if not m:
            raise NormalizationError(f"bad ip {raw!r}")
        return f"IP{m.group(1)}"
    if kind == "bool":
        if s.startswith(("否", "不", "no", "非")):
            return False
        if s in ("是", "yes", "可调光") or "调光" in s:
            return True
        raise NormalizationError(f"bad bool {raw!r}")
    raise NormalizationError(kind)


def same(a, b, rel=0.02):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= rel * max(abs(b), 1e-9)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b
