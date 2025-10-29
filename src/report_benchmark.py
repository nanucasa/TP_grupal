# Genera un reporte comparativo (MD + PNG) con métricas de test de LogReg, RF y XGB.

import json, os
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT_MD = ROOT / "reports" / "benchmark.md"
OUT_IMG = ROOT / "reports" / "f1_bench.png"

files = {
    "LogReg": ROOT / "metrics_test.json",
    "RF":     ROOT / "metrics_test_rf.json",
    "XGB":    ROOT / "metrics_test_xgb.json",
}

def load_metrics(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        return j.get("test", {})
    except Exception:
        return {}

rows = []
for model, path in files.items():
    m = load_metrics(path)
    rows.append({
        "model": model,
        "threshold": m.get("threshold", ""),
        "accuracy": m.get("accuracy", ""),
        "precision": m.get("precision", ""),
        "recall": m.get("recall", ""),
        "f1": m.get("f1", ""),
        "roc_auc": m.get("roc_auc", ""),
        "pr_ap": m.get("pr_ap", ""),
    })

# Markdown
os.makedirs(OUT_MD.parent, exist_ok=True)
header = ["Model","Threshold","Accuracy","Precision","Recall","F1","ROC_AUC","PR_AP"]
md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"]*len(header)) + "|"]
for r in rows:
    md.append("| " + " | ".join(str(r[k]) if r[k] != "" else "" for k in ["model","threshold","accuracy","precision","recall","f1","roc_auc","pr_ap"]) + " |")
OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

# Gráfico F1
labels = [r["model"] for r in rows]
f1_vals = [float(r["f1"]) if r["f1"] != "" else 0.0 for r in rows]
plt.figure(figsize=(6,4))
plt.bar(labels, f1_vals)
plt.title("F1 (Test) por Modelo")
plt.ylabel("F1")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT_IMG, dpi=150)
print("[OK] Reporte generado:", OUT_MD, "|", OUT_IMG)
