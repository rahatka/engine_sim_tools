import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Regex helpers
# ----------------------------

LABEL_RE = re.compile(
    r"""^\s*label\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(.*?)\s*\)\s*(?://.*)?$""",
    re.MULTILINE,
)

HEADER_LINE_RE = re.compile(r"^\s*//\s*(.+?)\s*$", re.MULTILINE)
YEAR_RE = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")

# Example:
# // lobes.py v1.0 | 202.3 248.3 LSA 113.97 ADV -1.28 OVL -2.6
LOBE_SUMMARY_PIPE_RE = re.compile(
    r"lobes\.py.*?\|\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+LSA\s*:?\s*([-+]?\d+(?:\.\d+)?)\s*,?\s*ADV\s*:?\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Newer lobes.py output, often split across comment lines:
# // ID: 200.0 ED: 207.0 LSA: 116.50, ADV: 2.00
LOBE_SUMMARY_NAMED_RE = re.compile(
    r"\bID\s*:\s*([-+]?\d+(?:\.\d+)?)"
    r".*?\bED\s*:\s*([-+]?\d+(?:\.\d+)?)"
    r".*?\bLSA\s*:\s*([-+]?\d+(?:\.\d+)?)"
    r"\s*,?\s*\bADV\s*:\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)

LEADING_NUM_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\b")


def safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_labels(text: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for m in LABEL_RE.finditer(text):
        name, expr = m.group(1), m.group(2).strip()
        labels[name] = expr
    return labels


def expr_leading_number(expr: Optional[str]) -> Optional[float]:
    if not expr:
        return None
    m = LEADING_NUM_RE.match(expr)
    if not m:
        return None
    return safe_float(m.group(1))


def parse_lobe_summary(text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    for pattern in (LOBE_SUMMARY_PIPE_RE, LOBE_SUMMARY_NAMED_RE):
        m = pattern.search(text)
        if m:
            return (
                safe_float(m.group(1)),  # intake duration
                safe_float(m.group(2)),  # exhaust duration
                safe_float(m.group(3)),  # LSA
                safe_float(m.group(4)),  # advance
            )
    return None, None, None, None


def compute_cam_metrics_from_labels(
    labels: Dict[str, str],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute durations, LSA and cam advance from valve timing events.

    Timing convention used by these engine files:
      IVO: degrees BTDC, IVC: degrees ABDC
      EVO: degrees BBDC, EVC: degrees ATDC

    Negative values naturally represent events on the opposite side of TDC/BDC.
    """
    ivo = expr_leading_number(labels.get("IVO"))
    ivc = expr_leading_number(labels.get("IVC"))
    evo = expr_leading_number(labels.get("EVO"))
    evc = expr_leading_number(labels.get("EVC"))

    intake_duration = None
    exhaust_duration = None
    intake_centerline = None
    exhaust_centerline = None

    if ivo is not None and ivc is not None:
        intake_duration = 180.0 + ivo + ivc
        intake_centerline = intake_duration / 2.0 - ivo

    if evo is not None and evc is not None:
        exhaust_duration = 180.0 + evo + evc
        exhaust_centerline = exhaust_duration / 2.0 - evc

    lsa = None
    advance = None
    if intake_centerline is not None and exhaust_centerline is not None:
        lsa = (intake_centerline + exhaust_centerline) / 2.0
        advance = lsa - intake_centerline

    return intake_duration, exhaust_duration, lsa, advance


def displacement_liters(bore_mm: Optional[float], stroke_mm: Optional[float], cyl: Optional[float]) -> Optional[float]:
    if bore_mm is None or stroke_mm is None or cyl is None:
        return None
    # mm^3 -> liters
    return (math.pi / 4.0) * (bore_mm ** 2) * stroke_mm * cyl / 1_000_000.0


def parse_engine_file(text: str, rel_path: str) -> Dict[str, object]:
    labels = parse_labels(text)

    # Core labels
    bore = expr_leading_number(labels.get("bore"))
    stroke = expr_leading_number(labels.get("stroke"))
    cyl = expr_leading_number(labels.get("cyl"))
    compression_ratio = expr_leading_number(labels.get("compression_ratio"))
    con_rod = expr_leading_number(labels.get("con_rod"))

    # Valve diameters
    intake_valve_dia = expr_leading_number(labels.get("intake_valve_diameter"))
    exhaust_valve_dia = expr_leading_number(labels.get("exhaust_valve_diameter"))

    # Valve lifts (common labels)
    intake_valve_lift = expr_leading_number(labels.get("IVL"))
    exhaust_valve_lift = expr_leading_number(labels.get("EVL"))

    # Preferred source for duration/LSA/ADV
    intake_duration, exhaust_duration, lsa, advance = parse_lobe_summary(text)

    # Fallback from valve timing labels. This also fills LSA/advance when
    # the file has no lobes.py summary comment.
    if any(v is None for v in (intake_duration, exhaust_duration, lsa, advance)):
        d_i, d_e, computed_lsa, computed_advance = compute_cam_metrics_from_labels(labels)
        if intake_duration is None:
            intake_duration = d_i
        if exhaust_duration is None:
            exhaust_duration = d_e
        if lsa is None:
            lsa = computed_lsa
        if advance is None:
            advance = computed_advance

    disp = displacement_liters(bore, stroke, cyl)
    rod_ratio = (con_rod / stroke) if (con_rod is not None and stroke not in (None, 0)) else None

    return {
        "source_file": rel_path,
        "cylinders": int(cyl) if cyl is not None and float(cyl).is_integer() else cyl,
        "bore_mm": bore,
        "stroke_mm": stroke,
        "displacement_l": disp,
        "displacement_cyl_l": disp / int(cyl) if cyl is not None and float(cyl).is_integer() else cyl,
        "compression_ratio": compression_ratio,
        "con_rod_length_mm": con_rod,
        "con_rod_to_stroke_ratio": rod_ratio,
        "intake_valve_dia_mm": intake_valve_dia,
        "intake_valve_lift_mm": intake_valve_lift,
        "exhaust_valve_dia_mm": exhaust_valve_dia,
        "exhaust_valve_lift_mm": exhaust_valve_lift,
        "intake_duration_deg": intake_duration,
        "exhaust_duration_deg": exhaust_duration,
        "LSA_deg": lsa,
        "advance_deg": advance,
    }


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def collect_from_directory(root: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Collect all .mr files under root, grouped by first-level folder:
      engines/10s/*.mr        -> section "10s"
      engines/aircraft/*.mr   -> section "aircraft"
      engines/experimental/*.mr -> section "experimental"
    Skips engines/chassis/*
    """
    out: Dict[str, List[Tuple[str, str]]] = {}

    for p in root.rglob("*.mr"):
        rel = p.relative_to(root).as_posix()

        # Skip chassis subtree
        if rel.startswith("chassis/") or "/chassis/" in rel:
            continue

        parts = rel.split("/")
        if len(parts) < 2:
            # top-level .mr helper file, not in a section folder
            continue

        section = parts[0]
        # Skip hidden / special dirs just in case
        if section.startswith(".") or section.startswith("_"):
            continue

        text = p.read_text(encoding="utf-8", errors="replace")
        out.setdefault(section, []).append((rel, text))

    return out


def write_combined_csv(section_map: Dict[str, List[Tuple[str, str]]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "cylinders",
        "bore_mm",
        "stroke_mm",
        "displacement_l",
        "displacement_cyl_l",
        "compression_ratio",
        "con_rod_length_mm",
        "con_rod_to_stroke_ratio",
        "intake_valve_dia_mm",
        "intake_valve_lift_mm",
        "exhaust_valve_dia_mm",
        "exhaust_valve_lift_mm",
        "intake_duration_deg",
        "exhaust_duration_deg",
        "LSA_deg",
        "advance_deg",
    ]

    rows: List[Dict[str, object]] = []

    for section in sorted(section_map.keys()):
        for rel_path, text in section_map[section]:
            try:
                row = parse_engine_file(text, rel_path)
            except Exception as e:
                print(f"[WARN] Failed parsing {rel_path}: {e}", file=sys.stderr)
                row = {"source_file": rel_path}
            rows.append(row)

    rows.sort(key=lambda r: str(r.get("source_file", "")).lower())

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: fmt(r.get(k)) for k in fieldnames})

    print(f"Wrote {out_csv} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(
        description="Extract engine metadata from .mr files into one combined CSV."
    )
    ap.add_argument(
        "engines_dir",
        help="Path to the 'engines' directory",
    )
    ap.add_argument(
        "-o", "--out",
        default="engines_combined.csv",
        help="Output CSV file path",
    )
    args = ap.parse_args()

    root = Path(args.engines_dir)
    if not root.exists() or not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    section_map = collect_from_directory(root)
    if not section_map:
        print("No .mr engine files found.", file=sys.stderr)
        sys.exit(1)

    write_combined_csv(section_map, Path(args.out))


if __name__ == "__main__":
    main()