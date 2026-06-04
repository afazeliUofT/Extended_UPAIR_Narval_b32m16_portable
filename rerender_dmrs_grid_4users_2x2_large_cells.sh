#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"

PLOT_DIR="${ROOT}/TWC_plots_comprehensive/comprehensive_plots"
CSV_IN="${PLOT_DIR}/dmrs_grid_4users_14sym_12sc_values.csv"

# Figure tuning. Increase FIG_W/FIG_H if you want even larger REs.
FIG_W="${FIG_W:-13.2}"
FIG_H="${FIG_H:-10.8}"
VALUE_FONTSIZE="${VALUE_FONTSIZE:-4.8}"

cd "${ROOT}"

if [[ -d "${VENV_PATH}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

if [[ ! -f "${CSV_IN}" ]]; then
  echo "ERROR: missing DMRS values CSV:"
  echo "  ${CSV_IN}"
  echo
  echo "Run make_dmrs_grid_4users_12sc.sh first to extract actual DMRS values."
  exit 2
fi

echo "============================================================"
echo "RERENDER 4-USER DMRS GRID: 2x2 LARGE-CELL VERSION"
echo "============================================================"
echo "ROOT=${ROOT}"
echo "PLOT_DIR=${PLOT_DIR}"
echo "CSV_IN=${CSV_IN}"
echo "FIG_W=${FIG_W}"
echo "FIG_H=${FIG_H}"
echo "VALUE_FONTSIZE=${VALUE_FONTSIZE}"
echo

echo "Removing old bad DMRS figure files..."
rm -f \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc.png" \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc.pdf" \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc_2x2.png" \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc_2x2.pdf"

export ROOT PLOT_DIR CSV_IN FIG_W FIG_H VALUE_FONTSIZE
export MPLBACKEND=Agg

python - <<'PY'
from __future__ import annotations

from pathlib import Path
import csv
import os
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


ROOT = Path(os.environ["ROOT"])
PLOT_DIR = Path(os.environ["PLOT_DIR"])
CSV_IN = Path(os.environ["CSV_IN"])
FIG_W = float(os.environ["FIG_W"])
FIG_H = float(os.environ["FIG_H"])
VALUE_FONTSIZE = float(os.environ["VALUE_FONTSIZE"])

PNG_MAIN = PLOT_DIR / "dmrs_grid_4users_14sym_12sc.png"
PDF_MAIN = PLOT_DIR / "dmrs_grid_4users_14sym_12sc.pdf"

PNG_2X2 = PLOT_DIR / "dmrs_grid_4users_14sym_12sc_2x2.png"
PDF_2X2 = PLOT_DIR / "dmrs_grid_4users_14sym_12sc_2x2.pdf"

CSV_USED = PLOT_DIR / "dmrs_grid_4users_14sym_12sc_2x2_values_used.csv"
REPORT = PLOT_DIR / "dmrs_grid_4users_14sym_12sc_2x2_report.txt"


def parse_port_set(s: str) -> str:
    s = str(s).strip()
    return s if s else "[]"


def fmt_component(x: float) -> str:
    if abs(x) < 5e-3:
        return "0.00"
    return f"{x:+.2f}"


def fmt_dmrs_value(real: float, imag: float) -> str:
    """
    Compact two-line complex format that fits inside one RE.
    Example:
      +0.71
      -0.71j
    """
    return f"{fmt_component(real)}\n{fmt_component(imag)}j"


def load_dmrs_csv(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "user",
            "dmrs_port_set",
            "ofdm_symbol",
            "subcarrier_local_0_to_11",
            "subcarrier_global",
            "real",
            "imag",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {path}: {sorted(missing)}")

        for r in reader:
            rows.append({
                "user": int(float(r["user"])),
                "dmrs_port_set": parse_port_set(r["dmrs_port_set"]),
                "ofdm_symbol": int(float(r["ofdm_symbol"])),
                "sc_local": int(float(r["subcarrier_local_0_to_11"])),
                "sc_global": int(float(r["subcarrier_global"])),
                "real": float(r["real"]),
                "imag": float(r["imag"]),
                "formatted": fmt_dmrs_value(float(r["real"]), float(r["imag"])),
                "extraction_mode": r.get("extraction_mode", ""),
            })
    return rows


rows = load_dmrs_csv(CSV_IN)

if not rows:
    raise SystemExit(f"No rows found in {CSV_IN}")

users = sorted(set(r["user"] for r in rows))
if users != [1, 2, 3, 4]:
    raise SystemExit(f"Expected users [1,2,3,4], found {users}")

# Determine displayed subcarrier labels from CSV.
sc_locals = sorted(set(r["sc_local"] for r in rows))
if not all(0 <= s <= 11 for s in sc_locals):
    raise SystemExit(f"Unexpected local subcarrier values: {sc_locals}")

# Infer global subcarrier labels for all 12 columns.
global_by_local = {}
for r in rows:
    global_by_local[r["sc_local"]] = r["sc_global"]

if len(global_by_local) < 12:
    # Fill missing local subcarriers assuming contiguous global indexing.
    min_local = min(global_by_local)
    min_global = global_by_local[min_local]
    for k in range(12):
        global_by_local.setdefault(k, min_global + (k - min_local))

xlabels = [str(global_by_local[k]) for k in range(12)]

# Create per-user masks and value maps.
masks = {}
texts = {}
ports = {}
for u in [1, 2, 3, 4]:
    masks[u] = np.zeros((14, 12), dtype=int)
    texts[u] = [["" for _ in range(12)] for _ in range(14)]
    ports[u] = "[]"

for r in rows:
    u = r["user"]
    s = r["ofdm_symbol"]
    k = r["sc_local"]

    if not (0 <= s < 14 and 0 <= k < 12):
        continue

    masks[u][s, k] = 1
    texts[u][s][k] = r["formatted"]
    ports[u] = r["dmrs_port_set"]

# Save data used by this renderer.
with open(CSV_USED, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "user",
            "dmrs_port_set",
            "ofdm_symbol",
            "subcarrier_local_0_to_11",
            "subcarrier_global",
            "real",
            "imag",
            "formatted_for_plot",
            "extraction_mode",
        ],
    )
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "user": r["user"],
            "dmrs_port_set": r["dmrs_port_set"],
            "ofdm_symbol": r["ofdm_symbol"],
            "subcarrier_local_0_to_11": r["sc_local"],
            "subcarrier_global": r["sc_global"],
            "real": r["real"],
            "imag": r["imag"],
            "formatted_for_plot": r["formatted"].replace("\n", " "),
            "extraction_mode": r["extraction_mode"],
        })

# Plot.
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 500,
})

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(FIG_W, FIG_H),
    constrained_layout=False,
)

axes = axes.ravel()

# 0=data, 1=DMRS
cmap = ListedColormap(["white", "#bfbfbf"])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

for idx, u in enumerate([1, 2, 3, 4]):
    ax = axes[idx]
    mask = masks[u]

    x = np.arange(13)
    y = np.arange(15)

    ax.pcolormesh(
        x,
        y,
        mask,
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidth=0.65,
        shading="flat",
    )

    ax.set_aspect("equal")
    ax.set_ylim(14, 0)
    ax.set_xlim(0, 12)

    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(xlabels)

    ax.set_yticks(np.arange(14) + 0.5)
    ax.set_yticklabels([str(i) for i in range(14)])

    ax.set_title(f"User {u}   DMRS port set={ports[u]}", pad=8)

    if idx in [2, 3]:
        ax.set_xlabel("Subcarrier index")
    else:
        ax.set_xlabel("")

    if idx in [0, 2]:
        ax.set_ylabel("OFDM symbol")
    else:
        ax.set_ylabel("")

    # Put values only inside DMRS REs.
    for sym in range(14):
        for sc in range(12):
            if mask[sym, sc] == 1:
                ax.text(
                    sc + 0.5,
                    sym + 0.5,
                    texts[u][sym][sc],
                    ha="center",
                    va="center",
                    fontsize=VALUE_FONTSIZE,
                    color="black",
                    linespacing=0.72,
                    clip_on=True,
                )

legend_handles = [
    Patch(facecolor="white", edgecolor="black", label="Data RE"),
    Patch(facecolor="#bfbfbf", edgecolor="black", label="DMRS RE"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.015),
    fontsize=10,
)

fig.suptitle(
    "4-user PUSCH DMRS structure over one PRB "
    "(12 subcarriers × 14 OFDM symbols)",
    y=0.985,
    fontsize=13,
    fontweight="bold",
)

fig.tight_layout(rect=[0.03, 0.065, 0.97, 0.955])

fig.savefig(PNG_MAIN, bbox_inches="tight", pad_inches=0.03)
fig.savefig(PDF_MAIN, bbox_inches="tight", pad_inches=0.03)
fig.savefig(PNG_2X2, bbox_inches="tight", pad_inches=0.03)
fig.savefig(PDF_2X2, bbox_inches="tight", pad_inches=0.03)

plt.close(fig)

# Report.
extraction_modes = sorted(set(str(r.get("extraction_mode", "")) for r in rows))
dmrs_counts = {u: int(masks[u].sum()) for u in [1, 2, 3, 4]}

report_lines = []
report_lines.append("2x2 large-cell 4-user DMRS grid render report")
report_lines.append("=" * 80)
report_lines.append(f"root: {ROOT}")
report_lines.append(f"input_csv: {CSV_IN}")
report_lines.append(f"output_png: {PNG_MAIN}")
report_lines.append(f"output_pdf: {PDF_MAIN}")
report_lines.append(f"output_png_2x2: {PNG_2X2}")
report_lines.append(f"output_pdf_2x2: {PDF_2X2}")
report_lines.append(f"values_used_csv: {CSV_USED}")
report_lines.append(f"figure_size: {FIG_W} x {FIG_H}")
report_lines.append(f"value_fontsize: {VALUE_FONTSIZE}")
report_lines.append(f"extraction_modes: {extraction_modes}")
report_lines.append("")
report_lines.append("DMRS RE count per displayed 12-subcarrier grid:")
for u in [1, 2, 3, 4]:
    report_lines.append(f"  user {u}: {dmrs_counts[u]} DMRS REs, port set={ports[u]}")
report_lines.append("")
report_lines.append("Rendering choices:")
report_lines.append("  data REs are white")
report_lines.append("  DMRS REs are grey")
report_lines.append("  DMRS values are compact two-line real/imaginary values inside grey REs")
report_lines.append("  subplots are arranged as 2x2 for larger RE cells")

REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print("============================================================")
print("CORRECTED 2x2 DMRS FIGURE GENERATED")
print("============================================================")
print(f"PNG main:       {PNG_MAIN}")
print(f"PDF main:       {PDF_MAIN}")
print(f"PNG 2x2 copy:   {PNG_2X2}")
print(f"PDF 2x2 copy:   {PDF_2X2}")
print(f"CSV used:       {CSV_USED}")
print(f"Report:         {REPORT}")
print()
print("DMRS counts per user:")
for u in [1, 2, 3, 4]:
    print(f"  User {u}: {dmrs_counts[u]} DMRS REs, port set={ports[u]}")
PY

echo
echo "============================================================"
echo "Generated files"
echo "============================================================"
ls -lh \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc.png" \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc.pdf" \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc_2x2.png" \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc_2x2.pdf" \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc_2x2_values_used.csv" \
  "${PLOT_DIR}/dmrs_grid_4users_14sym_12sc_2x2_report.txt"
