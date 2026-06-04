#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tensorflow as tf

from upair5g.config import load_config, set_cfg
from upair5g.builders import build_channel, build_pusch_transmitter, get_resource_grid
from upair5g.impairments import apply_rf_impairments_to_transmit_grid_if_enabled
from upair5g.utils import call_channel, call_transmitter, ebno_db_to_no


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_rf_mixed_u3_1dmrs.yaml")
    set_cfg(cfg, "system.batch_size_train", 2)
    set_cfg(cfg, "system.batch_size_eval", 2)
    tx, _ = build_pusch_transmitter(cfg, num_users=3)
    x, bits = call_transmitter(tx, 2)
    print("x clean:", x.shape, x.dtype)
    x_imp, meta = apply_rf_impairments_to_transmit_grid_if_enabled(x, tx, cfg, training=True)
    print("x impaired:", x_imp.shape, x_imp.dtype, meta)
    tf.debugging.assert_equal(tf.shape(x_imp), tf.shape(x))
    no = ebno_db_to_no(0.0, tx=tx, resource_grid=get_resource_grid(tx))
    channel = build_channel(cfg, tx)
    y, h = call_channel(channel, x_imp, no)
    print("y:", y.shape, y.dtype)
    print("h:", h.shape, h.dtype)
    print("bits:", None if bits is None else bits.shape)
    print("RF mixed smoke path passed.")


if __name__ == "__main__":
    main()
