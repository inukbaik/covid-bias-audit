"""Command-line interface: ``python -m covid_bias run --analysis all --output reports``."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from covid_bias.analysis import county, education, original, risk
from covid_bias.config import Config

ANALYSES = {
    "education": education.run,
    "county": county.run,
    "risk": risk.run,
    "original": original.run,
}


def run_analyses(names: Sequence[str], cfg: Config, out: Path) -> dict:
    """Run the named analyses, write ``metrics.json`` and return the results."""
    results = {}
    for name in names:
        logging.getLogger(__name__).info("running analysis: %s", name)
        results[name] = ANALYSES[name](cfg, out)
    path = out / "metrics.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(results)
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, default=float) + "\n")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="covid-bias", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one or all analyses")
    run.add_argument("--analysis", choices=[*ANALYSES, "all"], default="all")
    run.add_argument("--output", type=Path, default=None, help="output directory (default reports)")
    run.add_argument("--fast", action="store_true", help="fewer repeats/bootstraps (for testing)")
    run.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = Config()
    if args.fast:
        cfg = cfg.fast_mode()
    out = args.output or cfg.reports_dir
    names = list(ANALYSES) if args.analysis == "all" else [args.analysis]
    results = run_analyses(names, cfg, out)
    print(f"Wrote results for {', '.join(results)} to {out}")  # noqa: T201
    return 0
