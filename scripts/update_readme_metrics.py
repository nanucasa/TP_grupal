
# Actualiza la tabla entre <!-- METRICS_START --> y <!-- METRICS_END --> en README.md
# Lee métricas de: metrics*.json y umbrales en artifacts/best_threshold*.json
# Mantiene estilo simple y robusto para valores faltantes.

import json, os, re
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
README_PATH = os.path.join(ROOT, "README.md")

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _metric(d: Dict[str, Any], *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default

def _fmt(x: Optional[float]) -> str:
    if x is None: return "-"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    pairs = [
        ("LogReg", "metrics.json",         "valid",  "artifacts/best_threshold.json"),
        ("LogReg", "metrics_test.json",    "test",   "artifacts/best_threshold.json"),
        ("RF",     "metrics_rf.json",      "valid",  "artifacts/best_threshold_rf.json"),
        ("RF",     "metrics_test_rf.json", "test",   "artifacts/best_threshold_rf.json"),
        ("FE",     "metrics_fe.json",      "valid",  "artifacts/best_threshold_fe.json"),
        ("FE",     "metrics_test_fe.json", "test",   "artifacts/best_threshold_fe.json"),
        ("XGB",    "metrics_xgb.json",     "valid",  "artifacts/best_threshold_xgb.json"),
        ("XGB",    "metrics_test_xgb.json","test",   "artifacts/best_threshold_xgb.json"),
    ]

    for model, metrics_file, split, thr_file in pairs:
        mpath = os.path.join(ROOT, metrics_file)
        if not os.path.exists(mpath):
            continue
        m = _read_json(mpath) or {}
        acc = _metric(m, "accuracy", "acc")
        prec= _metric(m, "precision", "prec")
        rec = _metric(m, "recall", "rec")
        f1  = _metric(m, "f1", "f1_score")

        thr_val = None
        tpath = os.path.join(ROOT, thr_file)
        if os.path.exists(tpath):
            t = _read_json(tpath) or {}
            thr_val = _metric(t, "best_threshold", "threshold", "best_thr", default=None)

        rows.append({
            "Modelo": model,
            "Split": split,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Threshold": thr_val
        })
    return rows

def build_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "| Modelo | Split | Accuracy | Precision | Recall | F1 | Threshold |\n|---|---|---:|---:|---:|---:|---:|\n| - | - | - | - | - | - | - |\n"
    # Orden: test primero, luego valid; dentro, por modelo
    order = {"test": 0, "valid": 1}
    rows_sorted = sorted(rows, key=lambda r: (order.get(r["Split"], 9), r["Modelo"]))
    lines = []
    lines.append("| Modelo | Split | Accuracy | Precision | Recall | F1 | Threshold |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows_sorted:
        lines.append(
            f"| {r['Modelo']} | {r['Split']} | {_fmt(r['Accuracy'])} | {_fmt(r['Precision'])} | {_fmt(r['Recall'])} | {_fmt(r['F1'])} | {_fmt(r['Threshold'])} |"
        )
    return "\n".join(lines) + "\n"

def update_readme(table_md: str):
    if not os.path.exists(README_PATH):
        raise FileNotFoundError(f"No existe README.md en {README_PATH}")
    with open(README_PATH, "r", encoding="utf-8") as f:
        txt = f.read()

    start_tag = "<!-- METRICS_START -->"
    end_tag   = "<!-- METRICS_END -->"

    if start_tag in txt and end_tag in txt:
        # Reemplazar contenido entre marcadores
        new_txt = re.sub(
            rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}",
            f"{start_tag}\n\n{table_md}\n{end_tag}",
            txt,
            flags=re.DOTALL
        )
    else:
        # Anexar al final si faltan marcadores
        new_txt = txt.rstrip() + f"\n\n{start_tag}\n\n{table_md}\n{end_tag}\n"

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_txt)

if __name__ == "__main__":
    rows = collect_rows()
    table = build_table(rows)
    update_readme(table)
    print("[OK] README.md actualizado con métricas.")
