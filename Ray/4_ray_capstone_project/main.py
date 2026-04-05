# Ray capstone scaffold starter: TLC-backed per-zone recommendations under skew.
# Runs: prepare TLC replay assets -> initialize per-zone actors -> compare blocking and async execution.

from __future__ import annotations

import argparse
from pathlib import Path

import ray


@ray.remote
class ZoneActor:
    def __init__(self, zone_id: int, zone_data_path: str):
        self.zone_id = zone_id
        self.zone_data_path = zone_data_path
        self.last_tick_id = None
        self.last_decision = None

    def next_snapshot(self, tick_id: int):
        # TODO: Advance actor-owned replay state and return the minimal snapshot
        # needed by the scoring task for this zone and tick.
        pass

    def write_decision(self, tick_id: int, decision: str, used_fallback: bool = False):
        # TODO: Make this write idempotent by (zone_id, tick_id).
        pass


@ray.remote
def score_zone(snapshot: dict) -> dict:
    # TODO: Return a pure per-zone decision like:
    # {"zone_id": ..., "tick_id": ..., "decision": "NEED" | "OK", "task_latency_s": ...}
    pass


def prepare_assets(reference_parquet: Path, replay_parquet: Path, output_dir: Path) -> None:
    # TODO: Validate adjacent months, select active zones, aggregate replay ticks,
    # build reference baselines, and write prepared assets.
    pass


def run_blocking(prepared_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    # TODO: Collect all zone snapshots for a tick, launch all scoring tasks,
    # wait for every result, and write final artifacts.
    pass


def run_async(prepared_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    # TODO: Keep zone work bounded in flight, use ray.wait() inside the driver loop,
    # finalize ticks under a partial-readiness policy, and write final artifacts.
    pass


def run_stress(prepared_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    # TODO: Reuse the async path with harsher skew settings.
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capstone starter for TLC-backed per-zone recommendations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--reference-parquet", type=Path, required=True)
    prepare.add_argument("--replay-parquet", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.set_defaults(handler=handle_prepare)

    run = subparsers.add_parser("run")
    run.add_argument("--prepared-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--mode", choices=("blocking", "async", "stress"), required=True)
    run.add_argument("--max-inflight-zones", type=int, default=4)
    run.add_argument("--tick-timeout-s", type=float, default=2.0)
    run.add_argument("--completion-fraction", type=float, default=0.75)
    run.add_argument("--slow-zone-fraction", type=float, default=0.25)
    run.add_argument("--slow-zone-sleep-s", type=float, default=1.0)
    run.add_argument("--fallback-policy", default="previous_else_ok")
    run.add_argument("--ray-address", default=None)
    run.set_defaults(handler=handle_run)

    return parser


def handle_prepare(args: argparse.Namespace) -> None:
    prepare_assets(args.reference_parquet, args.replay_parquet, args.output_dir)


def handle_run(args: argparse.Namespace) -> None:
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    if args.mode == "blocking":
        run_blocking(args.prepared_dir, args.output_dir, args)
    elif args.mode == "async":
        run_async(args.prepared_dir, args.output_dir, args)
    else:
        run_stress(args.prepared_dir, args.output_dir, args)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
