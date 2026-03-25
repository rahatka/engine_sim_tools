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
LOBE_SUMMARY_RE = re.compile(
    r"lobes\.py.*?\|\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+LSA\s+([-+]?\d+(?:\.\d+)?)\s+ADV\s+([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
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


def parse_year_and_engine_name(text: str, file_stem: str) -> Tuple[Optional[int], str]:
    """
    Tries to parse a comment line like:
      // 1914 Buick C-54, 55 48 @
    Returns (year, engine_name)
    """
    header_lines = [m.group(1).strip() for m in HEADER_LINE_RE.finditer(text)]

    # Prefer a comment line with a year (skip generic banner lines)
    for line in header_lines:
        low = line.lower()
        if low.startswith("engine sim"):
            continue
        ym = YEAR_RE.search(line)
        if ym:
            year = int(ym.group(1))
            engine_name = re.sub(rf"^\s*{year}\s*", "", line).strip()
            engine_name = engine_name.strip("-–— ").strip()
            if engine_name:
                return year, engine_name

    # Fallback: any year in comments + filename
    for line in header_lines:
        ym = YEAR_RE.search(line)
        if ym:
            return int(ym.group(1)), file_stem.replace("_", " ")

    return None, file_stem.replace("_", " ")


def parse_lobe_summary(text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    m = LOBE_SUMMARY_RE.search(text)
    if not m:
        return None, None, None, None
    return (
        safe_float(m.group(1)),  # intake duration
        safe_float(m.group(2)),  # exhaust duration
        safe_float(m.group(3)),  # LSA
        safe_float(m.group(4)),  # advance
    )


def compute_duration_from_labels(labels: Dict[str, str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Fallback if lobes.py summary is absent:
      intake_duration = 180 + IVO + IVC
      exhaust_duration = 180 + EVO + EVC
    """
    ivo = expr_leading_number(labels.get("IVO"))
    ivc = expr_leading_number(labels.get("IVC"))
    evo = expr_leading_number(labels.get("EVO"))
    evc = expr_leading_number(labels.get("EVC"))

    intake_duration = None
    exhaust_duration = None

    if ivo is not None and ivc is not None:
        intake_duration = 180.0 + ivo + ivc
    if evo is not None and evc is not None:
        exhaust_duration = 180.0 + evo + evc

    return intake_duration, exhaust_duration


def displacement_cc(bore_mm: Optional[float], stroke_mm: Optional[float], cyl: Optional[float]) -> Optional[float]:
    if bore_mm is None or stroke_mm is None or cyl is None:
        return None
    # mm^3 -> cm^3
    return (math.pi / 4.0) * (bore_mm ** 2) * stroke_mm * cyl / 1000.0


def parse_engine_file(text: str, rel_path: str) -> Dict[str, object]:
    file_stem = Path(rel_path).stem
    labels = parse_labels(text)
    year, engine_name = parse_year_and_engine_name(text, file_stem)

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

    # Fallback durations from timing labels
    if intake_duration is None or exhaust_duration is None:
        d_i, d_e = compute_duration_from_labels(labels)
        if intake_duration is None:
            intake_duration = d_i
        if exhaust_duration is None:
            exhaust_duration = d_e

    disp = displacement_cc(bore, stroke, cyl)
    rod_ratio = (con_rod / stroke) if (con_rod is not None and stroke not in (None, 0)) else None

    return {
        "year": year,
        "engine_name": engine_name,
        "cylinders": int(cyl) if cyl is not None and float(cyl).is_integer() else cyl,
        "bore_mm": bore,
        "stroke_mm": stroke,
        "displacement_cm3": disp,
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
        "source_file": rel_path,
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


def write_section_csvs(section_map: Dict[str, List[Tuple[str, str]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "year",
        "engine_name",
        "cylinders",
        "bore_mm",
        "stroke_mm",
        "displacement_cm3",
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
        "source_file",
    ]

    for section in sorted(section_map.keys()):
        rows: List[Dict[str, object]] = []

        for rel_path, text in section_map[section]:
            try:
                row = parse_engine_file(text, rel_path)
            except Exception as e:
                print(f"[WARN] Failed parsing {rel_path}: {e}", file=sys.stderr)
                row = {
                    "year": None,
                    "engine_name": Path(rel_path).stem.replace("_", " "),
                    "source_file": rel_path,
                }
            rows.append(row)

        rows.sort(key=lambda r: (
            9999 if r.get("year") in (None, "") else int(r["year"]),
            str(r.get("engine_name", "")).lower()
        ))

        out_csv = out_dir / f"{section}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: fmt(r.get(k)) for k in fieldnames})

        print(f"Wrote {out_csv} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(
        description="Extract engine metadata from .mr files into per-section CSVs."
    )
    ap.add_argument(
        "engines_dir",
        help="Path to the 'engines' directory",
    )
    ap.add_argument(
        "-o", "--out",
        default="engine_csv_out",
        help="Output directory for CSV files (one per section folder)",
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

    write_section_csvs(section_map, Path(args.out))


if __name__ == "__main__":
    main()