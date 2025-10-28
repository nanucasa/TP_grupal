# scripts/update_readme_metrics.py
# Lee métricas de JSONs y actualiza una sección del README entre marcadores.
import json, os, re
from pathlib import Path

root = Path(".")
files = {
    "logreg_valid": root/"metrics.json",              # {"valid": {...}}
    "logreg_test":  root/"metrics_test.json",         # {"test": {..., "threshold": x}}
    "rf_valid":     root/"metrics_rf.json",           # {"valid": {...}}
    "rf_test":      root/"metrics_test_rf.json",      # {"test": {..., "threshold": x}}
    "thr_valid":    root/"metrics_threshold.json",    # {"valid_threshold": {...}}
    "thr_value":    root/"artifacts"/"best_threshold.json",  # {"best_threshold": x}
}

def ld(p):
    if p and Path(p).exists():
        with open(p, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

M = {k: ld(v) for k,v in files.items()}
def pick(d, k, default=""):
    try:
        v = d.get(k, default)
        return "" if v is None else v
    except Exception:
        return ""

def row(model, split, dct, key, thr=None):
    blk = dct.get(key, {})
    return [
        model, split,
        f"{thr:.4f}" if isinstance(thr, (int,float)) else (f"{thr}" if thr not in (None, "") else ""),
        f"{pick(blk,'accuracy'):.4f}" if 'accuracy' in blk else "",
        f"{pick(blk,'precision'):.4f}" if 'precision' in blk else "",
        f"{pick(blk,'recall'):.4f}" if 'recall' in blk else "",
        f"{pick(blk,'f1'):.4f}" if 'f1' in blk else "",
        f"{pick(blk,'roc_auc'):.4f}" if 'roc_auc' in blk else "",
        f"{pick(blk,'pr_ap'):.4f}" if 'pr_ap' in blk else "",
    ]

thr_val = None
if M["thr_value"]:
    thr_val = M["thr_value"].get("best_threshold", None)
test_thr_logreg = M["logreg_test"].get("test", {}).get("threshold", thr_val)
test_thr_rf     = M["rf_test"].get("test", {}).get("threshold", "")

rows = []
rows.append(row("LogReg", "valid", {"valid": M["thr_valid"].get("valid_threshold", {})}, "valid", thr_val))
rows.append(row("LogReg", "test",  M["logreg_test"], "test", test_thr_logreg))
rows.append(row("RF",     "valid", M["rf_valid"],    "valid", ""))  # RF valid no usa umbral óptimo
rows.append(row("RF",     "test",  M["rf_test"],     "test",  test_thr_rf))

header = ["Model","Split","Threshold","Accuracy","Precision","Recall","F1","ROC_AUC","PR_AP"]
table = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"]*len(header)) + "|"]
for r in rows:
    table.append("| " + " | ".join(str(x) if x is not None else "" for x in r) + " |")
md_table = "\n".join(table)

section = (
    "\n## Resultados (Benchmark)\n\n"
    "_Las métricas provienen de los JSON versionados por DVC (valid/test)._\n\n"
    + md_table + "\n"
)

readme = Path("README.md")
content = readme.read_text(encoding="utf-8") if readme.exists() else "# TP_grupal\n\n"

start_tag = "<!-- METRICS_START -->"
end_tag   = "<!-- METRICS_END -->"
block = f"{start_tag}\n{section}{end_tag}\n"

if start_tag in content and end_tag in content:
    content = re.sub(rf"{start_tag}.*?{end_tag}\n?", block, content, flags=re.S)
else:
    content += "\n" + block

readme.write_text(content, encoding="utf-8")
print("[OK] README.md actualizado con tabla de métricas.")
