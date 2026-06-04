#!/usr/bin/env python3
from __future__ import annotations

"""Probe the installed Narval/Sionna environment for a correct RF impairment path.

This script is intentionally diagnostic-only. It does not modify files and does
not start training. Its output is what we need before implementing a realistic
CFO/phase-noise backend in the UPAIR pipeline.
"""

import argparse
import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _try_import(name: str) -> Any | None:
    try:
        mod = importlib.import_module(name)
        print(f"[OK] import {name}")
        return mod
    except Exception as exc:
        print(f"[MISS] import {name}: {type(exc).__name__}: {exc}")
        return None


def _find_attr(module_names: list[str], attr: str) -> Any | None:
    for module_name in module_names:
        mod = _try_import(module_name)
        if mod is None:
            continue
        if hasattr(mod, attr):
            obj = getattr(mod, attr)
            print(f"[OK] {attr} found in {module_name}: {obj}")
            try:
                print(f"     signature: {inspect.signature(obj)}")
            except Exception:
                pass
            return obj
        print(f"[MISS] {module_name}.{attr}")
    return None


def _shape(x: Any) -> str:
    try:
        return str(tuple(x.shape.as_list()))
    except Exception:
        try:
            return str(tuple(x.shape))
        except Exception:
            return repr(type(x))


def _rank(x: Any) -> Any:
    try:
        return x.shape.rank
    except Exception:
        try:
            return len(x.shape)
        except Exception:
            return None


def _call_variants(obj: Any, *args: Any) -> Any:
    attempts = [
        lambda: obj(*args),
        lambda: obj(list(args)),
        lambda: obj(tuple(args)),
    ]
    last = None
    for fn in attempts:
        try:
            return fn()
        except Exception as exc:
            last = exc
    raise RuntimeError("all call variants failed") from last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/twc_comprehensive_mu32_base.yaml")
    parser.add_argument("--num-users", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--deep", action="store_true", help="Try tiny tensor forwards where possible.")
    args = parser.parse_args()

    _print_header("Python / TensorFlow / Sionna versions")
    print(f"python: {sys.version}")
    tf = _try_import("tensorflow")
    if tf is not None:
        print(f"tensorflow: {getattr(tf, '__version__', 'unknown')}")
        try:
            print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
        except Exception as exc:
            print(f"GPU query failed: {exc}")
    sionna = _try_import("sionna")
    if sionna is not None:
        print(f"sionna: {getattr(sionna, '__version__', 'unknown')}")

    _print_header("Sionna classes needed for a true time-domain RF path")
    needed = {
        "PUSCHTransmitter": ["sionna.phy.nr", "sionna.nr"],
        "PUSCHReceiver": ["sionna.phy.nr", "sionna.nr"],
        "OFDMModulator": ["sionna.phy.ofdm", "sionna.ofdm"],
        "OFDMDemodulator": ["sionna.phy.ofdm", "sionna.ofdm"],
        "TimeChannel": ["sionna.phy.channel", "sionna.channel"],
        "OFDMChannel": ["sionna.phy.channel", "sionna.channel"],
        "ApplyTimeChannel": ["sionna.phy.channel", "sionna.channel"],
        "GenerateTimeChannel": ["sionna.phy.channel", "sionna.channel"],
        "time_lag_discrete_time_channel": ["sionna.phy.channel", "sionna.channel"],
        "cir_to_time_channel": ["sionna.phy.channel", "sionna.channel"],
        "subcarrier_frequencies": ["sionna.phy.channel", "sionna.channel", "sionna.phy.ofdm", "sionna.ofdm"],
    }
    found = {}
    for attr, mods in needed.items():
        found[attr] = _find_attr(mods, attr) is not None

    _print_header("UPAIR config/transmitter/resource-grid probe")
    try:
        from upair5g.config import load_config
        from upair5g.builders import build_pusch_transmitter, build_channel, get_resource_grid
        from upair5g.utils import call_transmitter, call_channel, ebno_db_to_no

        cfg_path = PROJECT_ROOT / args.config
        cfg = load_config(cfg_path)
        cfg.setdefault("multiuser", {})["fixed_num_users"] = int(args.num_users)
        tx, _ = build_pusch_transmitter(cfg, num_users=int(args.num_users))
        rg = get_resource_grid(tx)
        print(f"config: {cfg_path}")
        print(f"requested num_users: {args.num_users}")
        print(f"tx type: {type(tx)}")
        print(f"resource_grid type: {type(rg)}")
        attrs = [
            "fft_size", "num_ofdm_symbols", "cyclic_prefix_length", "num_effective_subcarriers",
            "subcarrier_spacing", "num_tx", "num_streams_per_tx", "dc_null", "num_guard_carriers",
        ]
        for attr in attrs:
            value = getattr(rg, attr, None)
            print(f"rg.{attr}: {value!r}")
        carrier = getattr(getattr(tx, "_upair_pusch_configs", [None])[0], "carrier", None)
        if carrier is not None:
            for attr in ["subcarrier_spacing", "cyclic_prefix", "cyclic_prefix_length", "slot_duration", "num_symbols_per_slot"]:
                print(f"carrier.{attr}: {getattr(carrier, attr, None)!r}")

        x, bits = call_transmitter(tx, int(args.batch_size))
        print(f"freq-domain PUSCH x shape: {_shape(x)}, rank={_rank(x)}, dtype={getattr(x, 'dtype', None)}")
        print(f"bits shape: {_shape(bits) if bits is not None else None}")
        no = ebno_db_to_no(0.0, tx=tx, resource_grid=rg)
        channel = build_channel(cfg, tx)
        y, h = call_channel(channel, x, no)
        print(f"current OFDMChannel y shape: {_shape(y)}, rank={_rank(y)}, dtype={getattr(y, 'dtype', None)}")
        print(f"current OFDMChannel h shape: {_shape(h)}, rank={_rank(h)}, dtype={getattr(h, 'dtype', None)}")

        if args.deep:
            _print_header("Deep probe: OFDMModulator/OFDMDemodulator and time-domain feasibility")
            OFDMModulator = _find_attr(["sionna.phy.ofdm", "sionna.ofdm"], "OFDMModulator")
            OFDMDemodulator = _find_attr(["sionna.phy.ofdm", "sionna.ofdm"], "OFDMDemodulator")
            if OFDMModulator is not None:
                cp = getattr(rg, "cyclic_prefix_length", 0)
                try:
                    mod = OFDMModulator(cyclic_prefix_length=cp, precision=cfg.get("system", {}).get("precision", "single"))
                except Exception:
                    mod = OFDMModulator(cyclic_prefix_length=cp)
                x_time = _call_variants(mod, x)
                print(f"OFDMModulator output shape: {_shape(x_time)}, rank={_rank(x_time)}, dtype={getattr(x_time, 'dtype', None)}")
                # Apply a tiny CFO manually to confirm sample-axis convention.
                sample_rate = None
                try:
                    fft_size = int(getattr(rg, "fft_size"))
                    scs_hz = float(cfg["pusch"]["subcarrier_spacing_khz"]) * 1e3
                    sample_rate = fft_size * scs_hz
                    n = tf.cast(tf.range(tf.shape(x_time)[-1]), tf.float32)
                    cfo_hz = 100.0
                    ph = tf.exp(tf.complex(tf.zeros_like(n), 2.0 * 3.141592653589793 * cfo_hz * n / sample_rate))
                    y_cfo = x_time * tf.cast(ph, x_time.dtype)
                    print(f"manual CFO smoke output shape: {_shape(y_cfo)}, sample_rate_est_hz={sample_rate}")
                except Exception as exc:
                    print(f"manual CFO smoke failed: {type(exc).__name__}: {exc}")
                if OFDMDemodulator is not None:
                    try:
                        fft_size = int(getattr(rg, "fft_size"))
                        demod = OFDMDemodulator(fft_size=fft_size, l_min=0, cyclic_prefix_length=cp, precision=cfg.get("system", {}).get("precision", "single"))
                    except Exception:
                        demod = OFDMDemodulator(int(getattr(rg, "fft_size")), 0, cp)
                    x_round = _call_variants(demod, x_time)
                    print(f"OFDMDemodulator roundtrip output shape: {_shape(x_round)}, rank={_rank(x_round)}, dtype={getattr(x_round, 'dtype', None)}")

            # Try PUSCHTransmitter output_domain='time' by temporarily using builder-like creation.
            print("\nTrying native PUSCHTransmitter(output_domain='time')...")
            try:
                from upair5g.builders import build_pusch_config
                PUSCHTransmitter = _find_attr(["sionna.phy.nr", "sionna.nr"], "PUSCHTransmitter")
                pc = build_pusch_config(cfg)
                try:
                    tx_time = PUSCHTransmitter(pusch_configs=[pc], output_domain="time", return_bits=True, precision=cfg.get("system", {}).get("precision", "single"))
                except Exception:
                    tx_time = PUSCHTransmitter([pc], output_domain="time", return_bits=True)
                xt_out = _call_variants(tx_time, int(args.batch_size))
                if isinstance(xt_out, (tuple, list)):
                    for i, item in enumerate(xt_out):
                        print(f"time transmitter output[{i}] shape={_shape(item)}, dtype={getattr(item, 'dtype', None)}")
                else:
                    print(f"time transmitter output shape={_shape(xt_out)}, dtype={getattr(xt_out, 'dtype', None)}")
            except Exception as exc:
                print(f"native PUSCHTransmitter(output_domain='time') failed: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)

    except Exception as exc:
        print(f"[FATAL] UPAIR probe failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    _print_header("Summary JSON")
    print(json.dumps({"sionna_symbols_found": found}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
