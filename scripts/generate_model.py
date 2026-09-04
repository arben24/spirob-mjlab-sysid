#!/usr/bin/env python3
"""Generate the SpiRob MuJoCo model from its four spiral parameters.

The body is a logarithmic spiral. Four numbers fix it completely:

    L_target          length of the central axis           [m]
    base_d            diameter at the base                 [m]
    tip_d             diameter at the tip                  [m]
    Delta_theta_deg   discretisation step per segment      [deg]

``SpiralCalculator`` solves for the growth parameter ``b`` by bisection so the
central-axis length comes out at ``L_target``, then discretises the spiral into
segments. Every segment becomes a MuJoCo body with a hinge joint; two tendons
run along the flanks through site rings and are driven by force actuators with
``ctrlrange=[-50, 0]`` (pull only -- positive control values are no-ops).

Usage::

    uv run scripts/generate_model.py                      # defaults -> models/
    uv run scripts/generate_model.py --L 0.30 --base-d 0.05 --tip-d 0.01
    uv run scripts/generate_model.py --out /tmp/my_spirob.xml --auto-format

``--auto-format`` round-trips the XML through MuJoCo's own writer: nicely
formatted, but every comment is stripped.

Outputs: models/spirob_13seg.xml (or --out)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import spirob.generator as sg
from spirob.paths import MODELS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--L", "--l-target", dest="L_target", type=float, default=0.44,
                    help="Central-axis length in m (default: 0.44)")
    ap.add_argument("--base-d", type=float, default=0.10, help="Base diameter in m (default: 0.10)")
    ap.add_argument("--tip-d", type=float, default=0.03, help="Tip diameter in m (default: 0.03)")
    ap.add_argument("--delta-theta", type=float, default=30.0,
                    help="Discretisation step per segment in degrees (default: 30)")
    ap.add_argument("--name", default="Spirob", help="MuJoCo model name")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "spirob_13seg.xml",
                    help="Output XML path (default: models/spirob_13seg.xml)")
    ap.add_argument("--auto-format", action="store_true",
                    help="Reformat via MuJoCo's writer (drops all comments)")
    args = ap.parse_args()

    geometry = sg.SpiralCalculator(
        L_target=args.L_target,
        base_d=args.base_d,
        tip_d=args.tip_d,
        Delta_theta_deg=args.delta_theta,
    ).compute_geometry()

    print(f"Inputs: L_target={args.L_target} m, base_d={args.base_d} m, "
          f"tip_d={args.tip_d} m, dTheta={args.delta_theta:.1f} deg")
    print(geometry.summary())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    saved = sg.generate_and_save_xml(
        filepath=args.out,
        L_target=args.L_target,
        base_d=args.base_d,
        tip_d=args.tip_d,
        Delta_theta_deg=args.delta_theta,
        model_name=args.name,
        auto_format=args.auto_format,
    )
    print(f"\nXML written to: {saved.resolve()}")


if __name__ == "__main__":
    main()
