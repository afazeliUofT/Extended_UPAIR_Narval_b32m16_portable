#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"

# Display one representative 12-subcarrier PRB.
# Default SC_START=0 means subcarriers 0..11 of the full BWP.
SC_START="${SC_START:-0}"
SC_LEN="${SC_LEN:-12}"

# If 1, fail if actual Sionna/UPAIR DMRS extraction fails.
# This avoids accidentally producing a fake/approximate DMRS figure.
REQUIRE_ACTUAL="${REQUIRE_ACTUAL:-1}"

cd "${ROOT}"

if [[ -d "${VENV_PATH}" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "[WARN] VENV_PATH not found: ${VENV_PATH}"
  echo "[WARN] Continuing with current Python environment."
fi

mkdir -p TWC_plots_comprehensive/comprehensive_plots

export ROOT SC_START SC_LEN REQUIRE_ACTUAL
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""

python - <<'PY'
from __future__ import annotations

from pathlib import Path
import os
import sys
import traceback
import json
import math

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(os.environ["ROOT"])
SC_START = int(os.environ["SC_START"])
SC_LEN = int(os.environ["SC_LEN"])
REQUIRE_ACTUAL = os.environ.get("REQUIRE_ACTUAL", "1").strip().lower() in {"1", "true", "yes", "y"}

OUTDIR = ROOT / "TWC_plots_comprehensive" / "comprehensive_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

FIG_PNG = OUTDIR / "dmrs_grid_4users_14sym_12sc.png"
FIG_PDF = OUTDIR / "dmrs_grid_4users_14sym_12sc.pdf"
CSV_OUT = OUTDIR / "dmrs_grid_4users_14sym_12sc_values.csv"
REPORT_OUT = OUTDIR / "dmrs_grid_4users_14sym_12sc_report.txt"

# Make local package importable.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

PREFERRED_CONFIGS = [
    ROOT / "configs" / "twc_comprehensive_mu32_base_narval_stageB_locked_bleronly_compiled_mb16.yaml",
    ROOT / "configs" / "twc_comprehensive_mu32_base_narval_stageB_locked_u2_bleronly_compiled_mb16.yaml",
    ROOT / "configs" / "twc_comprehensive_mu32_base.yaml",
    ROOT / "configs" / "twc_comprehensive_mu32_eval_safe.yaml",
]

CONFIG_PATH = next((p for p in PREFERRED_CONFIGS if p.exists()), None)
if CONFIG_PATH is None:
    raise SystemExit(
        "No suitable config file found. Checked:\n"
        + "\n".join(str(p) for p in PREFERRED_CONFIGS)
    )


def cfmt(z: complex) -> str:
    """Compact complex-value text for DMRS REs."""
    z = complex(z)
    r = z.real
    i = z.imag

    def f(x):
        # Small, readable values for cells.
        if abs(x) < 5e-3:
            return "0"
        if abs(x - 1.0) < 5e-3:
            return "1"
        if abs(x + 1.0) < 5e-3:
            return "-1"
        return f"{x:+.2f}"

    if abs(i) < 5e-3:
        return f"{f(r)}"
    if abs(r) < 5e-3:
        return f"{f(i)}j"
    return f"{f(r)}\n{f(i)}j"


def to_numpy(x):
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def squeeze_to_symbol_subcarrier_grid(arr, user_index: int = 0) -> np.ndarray:
    """
    Return a complex-valued grid with shape [num_symbols, num_subcarriers].
    Works for common Sionna PUSCHConfig.dmrs_grid or pilot-pattern layouts.
    """
    a = to_numpy(arr)
    a = np.asarray(a)

    # Remove trivial singleton dimensions first.
    a = np.squeeze(a)

    if a.ndim == 2:
        if a.shape[0] == 14:
            return a.astype(np.complex64)
        if a.shape[1] == 14:
            return a.T.astype(np.complex64)
        raise ValueError(f"2D array does not contain an OFDM-symbol axis of length 14: shape={a.shape}")

    # For higher-rank arrays, choose the first axis of length 14 as symbol axis
    # and the largest remaining axis as subcarrier axis. Other axes are indexed at 0.
    shape = list(a.shape)
    sym_axes = [i for i, s in enumerate(shape) if s == 14]
    if not sym_axes:
        raise ValueError(f"Cannot find OFDM-symbol axis length 14 in shape={a.shape}")

    sym_axis = sym_axes[0]
    candidate_sc_axes = [i for i, s in enumerate(shape) if i != sym_axis and s >= SC_START + SC_LEN]
    if not candidate_sc_axes:
        candidate_sc_axes = [i for i, s in enumerate(shape) if i != sym_axis and s >= SC_LEN]
    if not candidate_sc_axes:
        raise ValueError(f"Cannot find subcarrier axis in shape={a.shape}")

    sc_axis = max(candidate_sc_axes, key=lambda i: shape[i])

    # Move symbol/subcarrier axes to front, then select index 0 over remaining dims.
    b = np.moveaxis(a, [sym_axis, sc_axis], [0, 1])
    while b.ndim > 2:
        b = b[..., 0]

    return b.astype(np.complex64)


def extract_from_pusch_config_grid(pusch_configs):
    """
    Preferred extraction path:
    PUSCHConfig.dmrs_grid gives the actual mapped DMRS grid per user/port.
    """
    out = []
    for u, pc in enumerate(pusch_configs):
        if not hasattr(pc, "dmrs_grid"):
            raise AttributeError(f"PUSCHConfig for user {u+1} has no dmrs_grid attribute.")
        grid = squeeze_to_symbol_subcarrier_grid(getattr(pc, "dmrs_grid"), user_index=u)
        out.append(grid)
    return out


def extract_from_resource_grid_pilot_pattern(tx):
    """
    Fallback extraction from resource_grid.pilot_pattern if dmrs_grid is unavailable.
    This still uses actual Sionna pilot masks/pilot values when available.
    """
    from upair5g.builders import get_resource_grid

    rg = get_resource_grid(tx)
    pp = getattr(rg, "pilot_pattern", None)
    if pp is None:
        pp = getattr(rg, "_pilot_pattern", None)
    if pp is None:
        raise AttributeError("resource_grid has no pilot_pattern/_pilot_pattern.")

    mask = getattr(pp, "mask", None)
    pilots = getattr(pp, "pilots", None)
    if mask is None or pilots is None:
        raise AttributeError("pilot_pattern does not expose mask and pilots.")

    mask = to_numpy(mask)
    pilots = to_numpy(pilots)

    # Expected often: mask [num_tx, num_streams, num_symbols, num_subcarriers]
    # pilots [num_tx, num_streams, num_pilots]
    mask = np.asarray(mask)
    pilots = np.asarray(pilots)

    if mask.ndim < 4:
        raise ValueError(f"pilot mask rank too small: shape={mask.shape}")

    grids = []
    for u in range(4):
        # Try common shape [U, streams, T, F].
        try:
            m = np.squeeze(mask[u, 0])
        except Exception:
            m = np.squeeze(mask)

        if m.ndim != 2:
            m = squeeze_to_symbol_subcarrier_grid(m)

        if m.shape[0] != 14 and m.shape[1] == 14:
            m = m.T

        if m.shape[0] != 14:
            raise ValueError(f"Could not reduce pilot mask to [14,Nsc] for user {u+1}; got {m.shape}")

        grid = np.zeros_like(m, dtype=np.complex64)
        idx = np.argwhere(m.astype(bool))

        try:
            p = np.ravel(pilots[u, 0])
        except Exception:
            p = np.ravel(pilots)

        n = min(len(idx), len(p))
        for k in range(n):
            s, f = idx[k]
            grid[s, f] = p[k]

        grids.append(grid)

    return grids


def fallback_synthetic_dmrs():
    """
    Emergency fallback. Not used unless REQUIRE_ACTUAL=0.
    It follows the configured Type-A symbol and creates deterministic QPSK values.
    """
    import yaml

    cfg = yaml.safe_load(open(CONFIG_PATH, "r", encoding="utf-8"))
    dmrs_cfg = cfg.get("multiuser", {}).get("dmrs", cfg.get("pusch", {}).get("dmrs", {}))
    sym = int(dmrs_cfg.get("type_a_position", 2))
    config_type = int(dmrs_cfg.get("config_type", 2))
    if config_type == 2:
        scs = [0, 1, 4, 5, 8, 9]
    else:
        scs = [0, 2, 4, 6, 8, 10]

    phases = [1+1j, 1-1j, -1+1j, -1-1j]
    grids = []
    for u in range(4):
        g = np.zeros((14, max(12, SC_START+SC_LEN)), dtype=np.complex64)
        for j, sc in enumerate(scs):
            g[sym, sc] = phases[(u+j) % 4] / np.sqrt(2)
        grids.append(g)
    return grids


def main():
    import yaml

    from upair5g.config import load_config
    from upair5g.builders import build_pusch_transmitter

    cfg = load_config(CONFIG_PATH)

    # Force 4 users for visualization.
    cfg.setdefault("multiuser", {})
    cfg["multiuser"]["enabled"] = True
    cfg["multiuser"]["max_num_users"] = max(4, int(cfg["multiuser"].get("max_num_users", 4)))
    cfg["multiuser"]["fixed_num_users"] = 4
    cfg["multiuser"]["eval_num_users"] = [4]

    # Ensure one full 14-symbol slot.
    cfg.setdefault("pusch", {})
    cfg["pusch"]["symbol_allocation"] = [0, 14]

    # Keep existing n_size_bwp. We only display 12 subcarriers from it.
    # The base config generally uses n_size_bwp=12 RBs = 144 subcarriers.

    extraction_mode = "actual_pusch_config_dmrs_grid"
    error_messages = []

    try:
        tx, _ = build_pusch_transmitter(cfg, num_users=4)
        pusch_configs = getattr(tx, "_upair_pusch_configs", None)
        if pusch_configs is None:
            pusch_configs = getattr(tx, "pusch_configs", None)
        if pusch_configs is None:
            raise AttributeError("Could not locate PUSCH configs on transmitter.")
        pusch_configs = list(pusch_configs)

        # Preferred: exact PUSCHConfig.dmrs_grid values.
        grids = extract_from_pusch_config_grid(pusch_configs)

        port_sets = []
        for pc in pusch_configs:
            dmrs = getattr(pc, "dmrs", None)
            ps = getattr(dmrs, "dmrs_port_set", None) if dmrs is not None else None
            if ps is None:
                ps = getattr(dmrs, "_dmrs_port_set", None) if dmrs is not None else None
            try:
                ps = list(ps)
            except Exception:
                ps = []
            port_sets.append(ps)

    except Exception as e1:
        error_messages.append("PUSCHConfig.dmrs_grid extraction failed:\n" + traceback.format_exc())
        try:
            tx, _ = build_pusch_transmitter(cfg, num_users=4)
            grids = extract_from_resource_grid_pilot_pattern(tx)
            pusch_configs = getattr(tx, "_upair_pusch_configs", [None]*4)
            port_sets = []
            for pc in list(pusch_configs)[:4]:
                dmrs = getattr(pc, "dmrs", None) if pc is not None else None
                ps = getattr(dmrs, "dmrs_port_set", None) if dmrs is not None else None
                try:
                    ps = list(ps)
                except Exception:
                    ps = []
                port_sets.append(ps)
            extraction_mode = "actual_resource_grid_pilot_pattern"
        except Exception as e2:
            error_messages.append("resource_grid.pilot_pattern extraction failed:\n" + traceback.format_exc())
            if REQUIRE_ACTUAL:
                REPORT_OUT.write_text(
                    "DMRS grid extraction failed and REQUIRE_ACTUAL=1.\n\n"
                    + "\n\n".join(error_messages),
                    encoding="utf-8",
                )
                raise SystemExit(
                    f"Actual DMRS extraction failed. See report:\n  {REPORT_OUT}\n"
                    "To allow a clearly marked synthetic fallback, rerun with REQUIRE_ACTUAL=0."
                )
            grids = fallback_synthetic_dmrs()
            port_sets = [[0], [1], [2], [3]]
            extraction_mode = "synthetic_fallback_not_actual"

    # Validate and crop.
    cropped = []
    dmrs_rows = []

    for u, g in enumerate(grids[:4]):
        g = np.asarray(g)
        if g.ndim != 2:
            raise ValueError(f"User {u+1} grid is not 2D after extraction: shape={g.shape}")
        if g.shape[0] != 14 and g.shape[1] == 14:
            g = g.T
        if g.shape[0] != 14:
            raise ValueError(f"User {u+1} grid does not have 14 OFDM symbols: shape={g.shape}")
        if g.shape[1] < SC_START + SC_LEN:
            raise ValueError(
                f"User {u+1} grid has only {g.shape[1]} subcarriers; "
                f"cannot display SC_START={SC_START}, SC_LEN={SC_LEN}."
            )

        c = g[:, SC_START:SC_START+SC_LEN].astype(np.complex64)
        cropped.append(c)

        nz = np.argwhere(np.abs(c) > 1e-8)
        for sym, sc_local in nz:
            z = complex(c[sym, sc_local])
            dmrs_rows.append({
                "user": u + 1,
                "dmrs_port_set": port_sets[u] if u < len(port_sets) else [],
                "ofdm_symbol": int(sym),
                "subcarrier_local_0_to_11": int(sc_local),
                "subcarrier_global": int(SC_START + sc_local),
                "real": float(z.real),
                "imag": float(z.imag),
                "abs": float(abs(z)),
                "phase_rad": float(np.angle(z)),
                "formatted": cfmt(z).replace("\n", ""),
                "extraction_mode": extraction_mode,
            })

    # Write CSV manually to avoid pandas dependency if not present.
    import csv
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        fields = [
            "user",
            "dmrs_port_set",
            "ofdm_symbol",
            "subcarrier_local_0_to_11",
            "subcarrier_global",
            "real",
            "imag",
            "abs",
            "phase_rad",
            "formatted",
            "extraction_mode",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dmrs_rows)

    # Figure.
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 500,
    })

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(7.16, 7.2),
        sharex=True,
        constrained_layout=False,
    )

    # 0=data, 1=DMRS.
    cmap = ListedColormap(["white", "0.72"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    for u, ax in enumerate(axes):
        g = cropped[u]
        mask = (np.abs(g) > 1e-8).astype(int)

        # Use pcolormesh so each RE has crisp grid borders.
        x = np.arange(SC_LEN + 1)
        y = np.arange(14 + 1)
        ax.pcolormesh(
            x,
            y,
            mask,
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidth=0.45,
            shading="flat",
        )

        ax.set_aspect("equal")
        ax.set_ylim(14, 0)
        ax.set_yticks(np.arange(14) + 0.5)
        ax.set_yticklabels([str(i) for i in range(14)])
        ax.set_ylabel("OFDM\nsymbol")

        ps = port_sets[u] if u < len(port_sets) else []
        ax.set_title(f"User {u+1}   DMRS port set={ps}")

        # Annotate DMRS values only.
        for sym, sc in np.argwhere(mask == 1):
            txt = cfmt(g[sym, sc])
            ax.text(
                sc + 0.5,
                sym + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=6.3,
                color="black",
                linespacing=0.82,
            )

        # Light labels for data/DMRS are handled by legend; data cells remain blank white.

    axes[-1].set_xticks(np.arange(SC_LEN) + 0.5)
    axes[-1].set_xticklabels([str(SC_START + i) for i in range(SC_LEN)])
    axes[-1].set_xlabel("Subcarrier index")

    fig.suptitle(
        "4-user PUSCH DMRS structure over one PRB "
        f"({SC_LEN} subcarriers × 14 OFDM symbols)",
        y=0.995,
        fontsize=10.5,
        fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="Data RE"),
        Patch(facecolor="0.72", edgecolor="black", label="DMRS RE"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0.02, 0.045, 0.98, 0.975])
    fig.savefig(FIG_PNG, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    report = []
    report.append("4-user DMRS grid visualization report")
    report.append("=" * 80)
    report.append(f"root: {ROOT}")
    report.append(f"config: {CONFIG_PATH}")
    report.append(f"extraction_mode: {extraction_mode}")
    report.append(f"require_actual: {REQUIRE_ACTUAL}")
    report.append(f"displayed_subcarriers: {SC_START}..{SC_START + SC_LEN - 1}")
    report.append(f"num_users: 4")
    report.append(f"rows_in_csv: {len(dmrs_rows)}")
    report.append("")
    report.append("Config summary:")
    report.append(f"  multiuser.dmrs: {cfg.get('multiuser', {}).get('dmrs', {})}")
    report.append(f"  pusch.dmrs: {cfg.get('pusch', {}).get('dmrs', {})}")
    report.append(f"  pusch.n_size_bwp: {cfg.get('pusch', {}).get('n_size_bwp')}")
    report.append(f"  pusch.symbol_allocation: {cfg.get('pusch', {}).get('symbol_allocation')}")
    report.append("")
    report.append("Outputs:")
    report.append(f"  png: {FIG_PNG}")
    report.append(f"  pdf: {FIG_PDF}")
    report.append(f"  csv: {CSV_OUT}")
    if error_messages:
        report.append("")
        report.append("Extraction fallback diagnostics:")
        report.extend(error_messages)

    REPORT_OUT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("============================================================")
    print("DMRS GRID FIGURE GENERATED")
    print("============================================================")
    print(f"Config used:       {CONFIG_PATH}")
    print(f"Extraction mode:   {extraction_mode}")
    print(f"PNG:               {FIG_PNG}")
    print(f"PDF:               {FIG_PDF}")
    print(f"CSV:               {CSV_OUT}")
    print(f"Report:            {REPORT_OUT}")
    print()
    print("First DMRS values:")
    for row in dmrs_rows[:24]:
        print(
            f"  user={row['user']} port={row['dmrs_port_set']} "
            f"sym={row['ofdm_symbol']} sc={row['subcarrier_global']} "
            f"value={row['formatted']}"
        )


if __name__ == "__main__":
    main()
PY
