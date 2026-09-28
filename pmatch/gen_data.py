"""Synthetic supplier catalog with known ground truth.

Every product is a "supplier sheet": a title plus header/value rows, rendered
the way Chinese supplier spreadsheets actually look — header aliases in
Chinese, English or mixed, the same quantity in different units
(2600mAh / 2.6Ah / 2600毫安时 / 9.6Wh), 万 multipliers, kgf·cm torque, and a
few genuinely ambiguous headers. Because we generate it, we know the true
canonical value of every row, which is what makes the evaluation honest.

Products come in families: an anchor, 1-2 exact duplicates from other
suppliers (true matches) and 2-3 near-misses that differ in exactly one
attribute the query asks for (hard negatives).
"""

import json
import random
from pathlib import Path

SEED = 7
OUT = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------- schema --

CATEGORIES = {
    "adapter": {
        "zh": ["电源适配器", "开关电源适配器", "充电器电源"],
        "en": ["Power Adapter", "AC/DC Adapter", "Switching Power Supply"],
    },
    "battery": {
        "zh": ["锂电池组", "电池包", "可充电锂电池"],
        "en": ["Battery Pack", "Rechargeable Battery", "Lithium Battery"],
    },
    "led_driver": {
        "zh": ["LED驱动电源", "恒流驱动电源", "LED恒流电源"],
        "en": ["LED Driver", "Constant Current LED Driver"],
    },
    "motor": {
        "zh": ["直流电机", "电机", "马达"],
        "en": ["DC Motor", "Motor", "Gear Motor"],
    },
}

CHEMISTRY_VOLTAGES = {
    "LiFePO4": [3.2, 6.4, 12.8, 25.6],
    "Li-ion": [3.7, 7.4, 11.1, 14.8],
    "LiPo": [3.7, 7.4, 11.1],
}

# What a buyer's query constrains, per category. Near-misses change one of these.
QUERY_FIELDS = {
    "adapter": ["output_voltage_v", "output_current_a", "plug", "cert"],
    "battery": ["chemistry", "voltage_v", "capacity_mah"],
    "led_driver": ["power_w", "output_current_ma", "ip_rating"],
    "motor": ["motor_type", "voltage_v", "speed_rpm"],
}


def sample_spec(cat, rng):
    if cat == "adapter":
        v = rng.choice([5, 9, 12, 15, 19, 24])
        a = rng.choice([1, 1.5, 2, 3, 5])
        return {
            "output_voltage_v": v,
            "output_current_a": a,
            "power_w": round(v * a, 1),
            "input_voltage_v": [100, 240],
            "plug": rng.choice(["US", "EU", "UK", "AU", "CN"]),
            "cert": rng.choice(["CE", "UL", "CCC", "FCC"]),
            "moq_units": rng.choice([500, 1000, 3000, 10000]),
        }
    if cat == "battery":
        chem = rng.choice(list(CHEMISTRY_VOLTAGES))
        return {
            "chemistry": chem,
            "voltage_v": rng.choice(CHEMISTRY_VOLTAGES[chem]),
            "capacity_mah": rng.choice([1000, 2000, 2600, 3000, 5000, 6000, 10000, 20000]),
            "max_discharge_a": rng.choice([1, 2, 5, 10, 20]),
            "weight_g": rng.choice([45, 90, 180, 350, 700, 1500]),
            "op_temp_c": rng.choice([[-20, 60], [-10, 45], [0, 45], [-20, 55]]),
            "moq_units": rng.choice([200, 500, 1000, 5000]),
        }
    if cat == "led_driver":
        return {
            "power_w": rng.choice([10, 20, 30, 50, 100, 150]),
            "output_current_ma": rng.choice([300, 350, 700, 1050, 1400]),
            "ip_rating": rng.choice(["IP20", "IP65", "IP67"]),
            "dimmable": rng.choice([True, False]),
            "op_temp_c": rng.choice([[-20, 50], [-30, 60], [-25, 55]]),
            "moq_units": rng.choice([500, 1000, 2000]),
        }
    return {
        "motor_type": rng.choice(["BLDC", "brushed_dc", "stepper"]),
        "voltage_v": rng.choice([6, 12, 24, 36, 48]),
        "speed_rpm": rng.choice([1000, 2000, 3000, 4000, 6000]),
        "rated_torque_nm": rng.choice([0.1, 0.3, 0.5, 1.0, 2.0]),
        "no_load_current_a": rng.choice([0.1, 0.2, 0.5, 1.0]),
        "weight_g": rng.choice([120, 250, 500, 900]),
        "moq_units": rng.choice([100, 500, 1000]),
    }


def mutate(cat, spec, field, rng):
    """Copy of spec with exactly one query field changed."""
    s = json.loads(json.dumps(spec))
    while True:
        alt = sample_spec(cat, rng)
        if field == "voltage_v" and cat == "battery":
            options = [v for v in CHEMISTRY_VOLTAGES[s["chemistry"]] if v != s[field]]
            s[field] = rng.choice(options)
            return s
        if alt[field] != s[field]:
            s[field] = alt[field]
            if cat == "battery" and field == "chemistry":
                # keep it a real chemistry/voltage pair, voltage then differs too
                s["voltage_v"] = rng.choice(CHEMISTRY_VOLTAGES[s["chemistry"]])
            if cat == "adapter":
                s["power_w"] = round(s["output_voltage_v"] * s["output_current_a"], 1)
            return s


# ------------------------------------------------------------- rendering --

def fmt(x):
    return f"{x:g}"


def r_volts(v, rng):
    return rng.choice([f"{fmt(v)}V", f"{fmt(v)}伏", f"DC{fmt(v)}V", f"{fmt(v)} V"])


def r_amps(a, rng):
    return rng.choice([f"{fmt(a)}A", f"{fmt(a * 1000)}mA", f"{fmt(a)}安", f"{fmt(a)} A"])


def r_watts(w, rng):
    return rng.choice([f"{fmt(w)}W", f"{fmt(w)}瓦", f"{fmt(w)} W"])


def r_mah(mah, volts, rng):
    wh = round(mah / 1000 * volts, 2)
    return rng.choice([
        f"{fmt(mah)}mAh", f"{fmt(mah / 1000)}Ah", f"{fmt(mah)}毫安时",
        f"{fmt(mah)} mAh", f"{fmt(wh)}Wh",
    ])


def r_ma(ma, rng):
    return rng.choice([f"{fmt(ma)}mA", f"{fmt(ma / 1000)}A", f"{fmt(ma)}毫安"])


def r_grams(g, rng):
    return rng.choice([f"{fmt(g)}g", f"{fmt(g / 1000)}kg", f"{fmt(g)}克", f"约{fmt(g)}g"])


def r_temp(rng_pair, rng):
    lo, hi = rng_pair
    return rng.choice([f"{lo}~{hi}℃", f"{lo}℃至{hi}℃", f"{lo}°C to +{hi}°C", f"{lo}~+{hi}°C"])


def r_moq(n, rng):
    opts = [f"{n}pcs", f"{n}个", f"{n} units"]
    if n >= 10000:
        opts.append(f"{fmt(n / 10000)}万个")
    if n >= 1000:
        opts.append(f"{fmt(n / 1000)}K")
    return rng.choice(opts)


def r_rpm(r, rng):
    return rng.choice([f"{r}rpm", f"{r}转/分", f"{r} r/min", f"{r}RPM"])


def r_torque(nm, rng):
    return rng.choice([
        f"{fmt(nm)}N·m", f"{fmt(nm * 1000)}mN·m", f"{fmt(round(nm * 10.197, 1))}kgf·cm", f"{fmt(nm)}N.m",
    ])


PLUG = {"US": ["美规", "US", "美标插头"], "EU": ["欧规", "EU", "欧标插头"],
        "UK": ["英规", "UK", "英标插头"], "AU": ["澳规", "AU", "澳标"], "CN": ["国标", "CN", "中规"]}
CERT = {"CE": ["CE认证", "CE"], "UL": ["UL认证", "UL"], "CCC": ["3C认证", "CCC", "3C"], "FCC": ["FCC认证", "FCC"]}
CHEM = {"LiFePO4": ["磷酸铁锂", "LiFePO4", "磷酸铁锂 LFP"], "Li-ion": ["三元锂", "锂离子", "Li-ion"],
        "LiPo": ["聚合物锂", "锂聚合物", "LiPo"]}
MOTOR = {"BLDC": ["无刷直流电机", "BLDC", "无刷"], "brushed_dc": ["有刷直流电机", "有刷", "Brushed DC"],
         "stepper": ["步进电机", "Stepper", "步进"]}

# field -> (header aliases, value renderer). Headers are what the LLM must map.
HEADERS = {
    "output_voltage_v": (["输出电压(V)", "输出电压", "Output Voltage", "Vout", "额定输出电压"], lambda s, r: r_volts(s["output_voltage_v"], r)),
    "output_current_a": (["输出电流(A)", "输出电流", "Output Current", "Iout", "输出电流A"], lambda s, r: r_amps(s["output_current_a"], r)),
    "power_w": (["功率(W)", "额定功率", "Power", "输出功率", "Pout"], lambda s, r: r_watts(s["power_w"], r)),
    "input_voltage_v": (["输入电压", "Input Voltage", "输入电压范围", "AC输入"], lambda s, r: r.choice(["100-240V", "AC100~240V", "100-240VAC 50/60Hz"])),
    "plug": (["插头类型", "插头", "Plug Type", "插脚规格"], lambda s, r: r.choice(PLUG[s["plug"]])),
    "cert": (["认证", "Certification", "认证标准", "安规认证"], lambda s, r: r.choice(CERT[s["cert"]])),
    "moq_units": (["起订量", "MOQ", "最小起订量", "最低订货量"], lambda s, r: r_moq(s["moq_units"], r)),
    "chemistry": (["电芯类型", "电池类型", "Cell Type", "化学体系"], lambda s, r: r.choice(CHEM[s["chemistry"]])),
    "voltage_v": (["标称电压(V)", "标称电压", "额定电压", "Nominal Voltage", "Rated Voltage"], lambda s, r: r_volts(s["voltage_v"], r)),
    "capacity_mah": (["容量(mAh)", "额定容量", "Capacity", "电池容量", "容量"], None),
    "max_discharge_a": (["最大放电电流", "最大持续放电电流(A)", "Max Discharge Current", "放电电流"], lambda s, r: r_amps(s["max_discharge_a"], r)),
    "weight_g": (["重量(g)", "重量", "净重", "Weight"], lambda s, r: r_grams(s["weight_g"], r)),
    "op_temp_c": (["工作温度(℃)", "工作温度", "Operating Temp", "使用温度"], lambda s, r: r_temp(s["op_temp_c"], r)),
    "output_current_ma": (["输出电流(mA)", "恒流输出电流", "Output Current", "输出电流"], lambda s, r: r_ma(s["output_current_ma"], r)),
    "ip_rating": (["防护等级", "防水等级", "IP Rating", "IP等级"], lambda s, r: r.choice([s["ip_rating"], f"防护等级{s['ip_rating']}", s["ip_rating"].lower()])),
    "dimmable": (["是否调光", "调光", "Dimmable", "调光方式"], lambda s, r: r.choice(["是", "0-10V调光", "Yes", "可调光"] if s["dimmable"] else ["否", "不可调光", "No", "非调光"])),
    "motor_type": (["电机类型", "类型", "Motor Type", "马达类型"], lambda s, r: r.choice(MOTOR[s["motor_type"]])),
    "speed_rpm": (["额定转速", "转速(rpm)", "Rated Speed", "空载转速"], lambda s, r: r_rpm(s["speed_rpm"], r)),
    "rated_torque_nm": (["额定扭矩", "额定转矩", "Rated Torque", "扭矩"], lambda s, r: r_torque(s["rated_torque_nm"], r)),
    "no_load_current_a": (["空载电流", "No-load Current", "空载电流(A)"], lambda s, r: r_amps(s["no_load_current_a"], r)),
}

# Headers that are genuinely ambiguous out of context. The field is still
# knowable from the value/category for a careful reader, but a model that
# says "ambiguous" here is being honest, not wrong.
AMBIGUOUS = {
    ("battery", "max_discharge_a"): "电流",          # charge or discharge current?
    ("adapter", "input_voltage_v"): "电压",          # input or output voltage?
    ("motor", "no_load_current_a"): "电流",          # no-load, rated or stall?
}

NOISE_ROWS = [
    ("包装方式", ["纸箱", "吸塑盒", "彩盒"]), ("产地", ["广东深圳", "浙江宁波", "江苏苏州"]),
    ("交期", ["7-15天", "15天", "现货"]), ("质保", ["1年", "2年", "18个月"]),
    ("外壳材质", ["PC", "铝合金", "ABS"]), ("型号", None),
]


def render_sheet(pid, cat, spec, rng):
    style = rng.choices(["zh", "en", "mixed"], weights=[70, 15, 15])[0]
    rows = []
    for field, value in spec.items():
        aliases, renderer = HEADERS[field]
        if field == "output_current_a" and cat == "led_driver":
            continue
        if style == "en":
            cands = [a for a in aliases if a.isascii()] or aliases
        elif style == "zh":
            cands = [a for a in aliases if not a.isascii()] or aliases
        else:
            cands = aliases
        header = rng.choice(cands)
        ambiguous = False
        if (cat, field) in AMBIGUOUS and rng.random() < 0.35:
            header, ambiguous = AMBIGUOUS[(cat, field)], True
        if field == "capacity_mah":
            raw = r_mah(spec["capacity_mah"], spec["voltage_v"], rng)
        else:
            raw = renderer(spec, rng)
        rows.append({"header": header, "value": raw, "field": field, "ambiguous": ambiguous})
    for header, vals in rng.sample(NOISE_ROWS, k=rng.randint(1, 3)):
        v = vals and rng.choice(vals) or f"{cat[:2].upper()}-{rng.randint(1000, 9999)}"
        rows.append({"header": header, "value": v, "field": None, "ambiguous": False})
    rng.shuffle(rows)
    return {"id": pid, "category": cat, "supplier": f"Supplier S{rng.randint(1, 60):03d}",
            "title": render_title(cat, spec, style, rng), "rows": rows, "truth": spec, "style": style}


def render_title(cat, spec, style, rng):
    name = rng.choice(CATEGORIES[cat]["zh" if style != "en" else "en"])
    bits = []
    if cat == "adapter":
        bits = [f"{fmt(spec['output_voltage_v'])}V{fmt(spec['output_current_a'])}A",
                rng.choice(PLUG[spec["plug"]]) if style != "en" else f"{spec['plug']} plug"]
    elif cat == "battery":
        bits = [f"{fmt(spec['voltage_v'])}V", rng.choice(CHEM[spec["chemistry"]])]
    elif cat == "led_driver":
        bits = [f"{spec['power_w']}W", spec["ip_rating"]]
    else:
        bits = [f"{spec['voltage_v']}V", rng.choice(MOTOR[spec["motor_type"]])]
    extras_zh = ["厂家直销", "现货批发", "工业级", "高品质", "路由器监控通用", "OEM定制"]
    extras_en = ["Factory Direct", "OEM", "Wholesale", "Industrial Grade"]
    bits.append(rng.choice(extras_zh if style != "en" else extras_en))
    rng.shuffle(bits)
    sep = " " if rng.random() < 0.6 else ""
    return sep.join([name] + bits) if rng.random() < 0.5 else sep.join(bits + [name])


# --------------------------------------------------------------- queries --

Q_EN = {
    "adapter": lambda s: f"{fmt(s['output_voltage_v'])}V {fmt(s['output_current_a'])}A power adapter, {s['plug']} plug, {s['cert']} certified",
    "battery": lambda s: f"{s['chemistry']} battery pack {fmt(s['voltage_v'])}V {fmt(s['capacity_mah'])}mAh",
    "led_driver": lambda s: f"{s['power_w']}W constant current LED driver {s['output_current_ma']}mA {s['ip_rating']}",
    "motor": lambda s: f"{s['voltage_v']}V {dict(BLDC='brushless DC', brushed_dc='brushed DC', stepper='stepper')[s['motor_type']]} motor {s['speed_rpm']} rpm",
}
Q_ZH = {
    "adapter": lambda s: f"{fmt(s['output_voltage_v'])}V{fmt(s['output_current_a'])}A {PLUG[s['plug']][0]} 电源适配器 {CERT[s['cert']][0]}",
    "battery": lambda s: f"{CHEM[s['chemistry']][0]}电池 {fmt(s['voltage_v'])}V {fmt(s['capacity_mah'])}mAh",
    "led_driver": lambda s: f"{s['power_w']}W LED恒流驱动 {s['output_current_ma']}mA {s['ip_rating']}",
    "motor": lambda s: f"{s['voltage_v']}V {MOTOR[s['motor_type']][0]} {s['speed_rpm']}转",
}


def matches(q_spec, p_spec, cat):
    return all(p_spec[f] == q_spec[f] for f in QUERY_FIELDS[cat])


def main(families_per_cat=26):
    rng = random.Random(SEED)
    products, anchors = [], []
    pid = 0
    for cat in CATEGORIES:
        for _ in range(families_per_cat):
            spec = sample_spec(cat, rng)
            fam = [spec]
            fam += [json.loads(json.dumps(spec)) for _ in range(rng.randint(1, 2))]   # same product, other suppliers
            for f in rng.sample(QUERY_FIELDS[cat], k=rng.randint(2, 3)):              # near-misses
                fam.append(mutate(cat, spec, f, rng))
            for i, s in enumerate(fam):
                if i and s is not spec:
                    # duplicates keep the query fields but a supplier may differ in minor specs
                    for minor in ("moq_units", "weight_g"):
                        if minor in s and rng.random() < 0.5:
                            s[minor] = sample_spec(cat, rng)[minor]
                products.append(render_sheet(f"P{pid:04d}", cat, s, rng))
                pid += 1
            anchors.append((cat, spec))

    queries = []
    for qi, (cat, spec) in enumerate(anchors):
        lang = "zh" if rng.random() < 0.3 else "en"
        text = (Q_ZH if lang == "zh" else Q_EN)[cat](spec)
        rel = [p["id"] for p in products if p["category"] == cat and matches(spec, p["truth"], cat)]
        hard = [p["id"] for p in products if p["category"] == cat and not matches(spec, p["truth"], cat)
                and sum(p["truth"][f] == spec[f] for f in QUERY_FIELDS[cat]) == len(QUERY_FIELDS[cat]) - 1]
        queries.append({"id": f"Q{qi:03d}", "category": cat, "lang": lang, "text": text,
                        "constraints": {f: spec[f] for f in QUERY_FIELDS[cat]},
                        "relevant": rel, "hard_negatives": hard})

    OUT.mkdir(exist_ok=True)
    with open(OUT / "products.jsonl", "w") as f:
        for p in products:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(OUT / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    n_amb = sum(r["ambiguous"] for p in products for r in p["rows"])
    print(f"{len(products)} products, {len(queries)} queries, "
          f"{sum(len(q['relevant']) for q in queries) / len(queries):.1f} relevant/query, "
          f"{sum(len(q['hard_negatives']) for q in queries) / len(queries):.1f} hard negatives/query, "
          f"{n_amb} ambiguous rows")


if __name__ == "__main__":
    main()
