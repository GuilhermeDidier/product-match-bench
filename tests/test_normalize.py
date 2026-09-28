"""With the correct field given, every generated row must normalize back to its ground truth."""
import json
from pathlib import Path

from pmatch.normalize import normalize, same

DATA = Path(__file__).resolve().parent.parent / "data" / "products.jsonl"


def test_every_row_roundtrips_to_truth():
    failures = []
    for line in DATA.open():
        p = json.loads(line)
        for r in p["rows"]:
            if r["field"] is None:
                continue
            got = normalize(r["field"], r["value"], p["truth"])
            if not same(got, p["truth"][r["field"]]):
                failures.append((p["id"], r["field"], r["value"], got, p["truth"][r["field"]]))
    assert not failures, failures[:10]


def test_units():
    assert normalize("capacity_mah", "2.6Ah") == 2600
    assert normalize("capacity_mah", "9.62Wh", {"voltage_v": 3.7}) == 2600
    assert normalize("moq_units", "1万个") == 10000
    assert normalize("moq_units", "5K") == 5000
    assert same(normalize("rated_torque_nm", "5.1kgf·cm"), 0.5)
    assert normalize("input_voltage_v", "100-240VAC 50/60Hz") == [100, 240]
    assert normalize("op_temp_c", "-20°C to +60°C") == [-20, 60]
    assert normalize("motor_type", "Brushed DC") == "brushed_dc"
