"""Profile the last DDP training steps on GPU and export one trace per rank."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the last DDP training steps and export traces."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="How many subprocess workers each rank uses for DataLoader batch loading.",
    )
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet34", "resnet50"),
        default="resnet18",
    )
    parser.add_argument("--dataset-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="4_ddp_on_cloud_gpus/traces")
    parser.add_argument("--trace-name", type=str, default="gpu_baseline")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run the profiler script with gloo on CPU for local validation.",
    )
    return parser.parse_args()


# region Reporting Helpers
# Small reporting helpers keep the end-of-run summary readable.
def print_section(title: str, *lines: str) -> str:
    """Print one labeled section and return it for saving to the trace log."""
    body = "\n".join([title, *[f"  {line}" for line in lines], ""])
    print(body, flush=True)
    return body


def gather_summaries_on_rank_zero(
    local_summary: dict[str, Any],
    rank: int,
    world_size: int,
) -> list[dict[str, Any]] | None:
    """Gather one small summary row from every rank onto rank 0 only."""
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_summary, object_gather_list=gathered, dst=0)
    return gathered
# endregion


def run_step(
    ddp_model: DDP,
    optimizer: torch.optim.Optimizer,
    data_iter: Any,
    device: torch.device,
) -> tuple[float, int]:
    with record_function("train_step"):
        with record_function("next_batch"):
            images, targets = next(data_iter)
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)

        with record_function("forward"):
            logits = ddp_model(images)

        with record_function("loss"):
            loss = F.cross_entropy(logits, targets)

        with record_function("backward"):
            # DDP registers autograd hooks up front, so gradient all-reduce starts during backward.
            loss.backward()

        with record_function("optimizer_step"):
            optimizer.step()

    return float(loss.item()), int(images.size(0))


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be at least 0.")

    required_vars = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required_vars if name not in os.environ]
    if missing:
        raise SystemExit(
            f"Missing distributed environment variables: {', '.join(missing)}. "
            "Launch Unit 4 with torchrun."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if world_size < 1:
        raise SystemExit(f"Invalid WORLD_SIZE={world_size}. Launch Unit 4 with torchrun.")

    if args.cpu:
        backend = "gloo"
        device = torch.device("cpu")
        device_name = "CPU"
    else:
        if not torch.cuda.is_available():
            raise SystemExit("Default mode requires a CUDA-enabled runtime.")

        device_count = torch.cuda.device_count()
        if device_count < local_world_size:
            raise SystemExit(
                "torchrun launched more local processes than visible GPUs. "
                f"LOCAL_WORLD_SIZE={local_world_size}, LOCAL_RANK={local_rank}, "
                f"visible_gpus={device_count}. Reduce --nproc_per_node or check "
                "CUDA_VISIBLE_DEVICES."
            )

        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device("cuda", local_rank)
        device_name = torch.cuda.get_device_name(local_rank)

    dist.init_process_group(backend=backend)
    try:

        dataset = datasets.FakeData(
            size=args.dataset_size,
            image_size=(3, 224, 224),
            num_classes=1000,
            transform=transforms.ToTensor(),
        )
        # DistributedSampler partitions the dataset indices of one logical dataset so each
        # rank builds batches from a different shard.
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        # DataLoader then turns this rank's assigned indices into local batches.
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=not args.cpu,
            drop_last=True,
        )
        # This script does not loop over epochs, so we set one fixed epoch here as a reminder:
        # real multi-epoch training must call set_epoch(...) at each epoch boundary or
        # DistributedSampler would reuse the same shuffle order every epoch.
        sampler.set_epoch(0)

        if args.model == "resnet18":
            model = models.resnet18(weights=None)
        elif args.model == "resnet34":
            model = models.resnet34(weights=None)
        else:
            model = models.resnet50(weights=None)
        model = model.to(device)

        if device.type == "cuda":
            ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            ddp_model = DDP(model)

        # Per-parameter optimizer state lives on the parameter device.
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9)

        data_iter = iter(loader)
        warmup_steps = max(args.steps - 2, 0)
        profiled_steps = min(args.steps, 2)
        final_loss = 0.0
        profiled_images = 0

        for _ in range(warmup_steps):
            final_loss, _ = run_step(
                ddp_model=ddp_model,
                optimizer=optimizer,
                data_iter=data_iter,
                device=device,
            )

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        if device.type == "cuda":
            # CUDA launches are asynchronous, so fence off any prior work before starting
            # the wall-clock throughput timer for the profiled window.
            torch.cuda.synchronize(device)
        profiled_start = time.perf_counter()
        with profile(activities=activities, record_shapes=False) as prof:
            for _ in range(profiled_steps):
                final_loss, batch_size = run_step(
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    data_iter=data_iter,
                    device=device,
                )
                profiled_images += batch_size
        if device.type == "cuda":
            # Wait for the profiled window's queued CUDA work to finish before stopping
            # the wall-clock throughput timer.
            torch.cuda.synchronize(device)
        profiled_seconds = time.perf_counter() - profiled_start

        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{args.trace_name}_rank{rank}.json"
        prof.export_chrome_trace(str(trace_path))

        gathered_rows = gather_summaries_on_rank_zero(
            {
                "rank": rank,
                "local_rank": local_rank,
                "device": str(device),
                "device_name": device_name,
                "profiled_images": profiled_images,
                "profiled_seconds": profiled_seconds,
                "profiled_steps": profiled_steps,
                "loss": final_loss,
            },
            rank=rank,
            world_size=world_size,
        )

        if rank == 0: #Rank 0 logging
            rows_by_rank = {
                int(row["rank"]): row
                for row in gathered_rows or []
            }
            log_blocks: list[str] = []

            log_blocks.append(
                print_section(
                    "config",
                    f"world_size={world_size}",
                    f"model={args.model}",
                    f"batch_size={args.batch_size}",
                    f"num_workers={args.num_workers}",
                    f"dataset_size={args.dataset_size}",
                    f"steps={args.steps}",
                    f"mode={'cpu' if args.cpu else 'gpu'}",
                    f"backend={backend}",
                    f"trace_name={args.trace_name}",
                    f"warmup_steps={warmup_steps}",
                    f"profiled_steps={profiled_steps}",
                )
            )

            mapping_lines = [
                (
                    f"rank {row_rank}: local_rank={row['local_rank']} "
                    f"device={row['device']} ({row['device_name']})"
                )
                for row_rank, row in sorted(rows_by_rank.items())
            ]
            log_blocks.append(print_section("gpu mapping", *mapping_lines))

            log_blocks.append(
                print_section(
                    "trace labels to look for",
                    "train_step",
                    "next_batch",
                    "forward",
                    "loss",
                    "backward",
                    "optimizer_step",
                )
            )

            summary_lines = []
            max_profiled_seconds = 0.0
            total_profiled_images = 0
            for row_rank, row in sorted(rows_by_rank.items()):
                row_seconds = float(row["profiled_seconds"])
                row_images = int(row["profiled_images"])
                total_profiled_images += row_images
                max_profiled_seconds = max(max_profiled_seconds, row_seconds)
                avg_step_time = row_seconds / max(int(row["profiled_steps"]), 1)
                summary_lines.append(
                    (
                        f"rank {row_rank}: profiled_steps={row['profiled_steps']}, "
                        f"profiled_images={row_images}, avg_profiled_step_time={avg_step_time:.4f}s, "
                        f"loss={float(row['loss']):.4f}"
                    )
                )
            log_blocks.append(print_section("profiled step summary", *summary_lines))

            if max_profiled_seconds <= 0:
                global_images_per_second = "n/a"
            else:
                global_images_per_second = (
                    f"{total_profiled_images / max_profiled_seconds:.1f} images/s"
                )
            log_blocks.append(
                print_section(
                    "throughput summary",
                    f"profiled_window_images={total_profiled_images}",
                    f"profiled_window_seconds={max_profiled_seconds:.4f}",
                    f"estimated_global_images_per_second={global_images_per_second}",
                    (
                        "compare this number across the tuning ladder together with the GPU trace"
                        if device.type == "cuda"
                        else "use this CPU-mode throughput only as a local validation check"
                    ),
                )
            )

            trace_lines = [
                str(trace_dir / f"{args.trace_name}_rank{row_rank}.json")
                for row_rank in sorted(rows_by_rank)
            ]
            log_blocks.append(print_section("trace files", *trace_lines))

            stdout_log_path = trace_dir / f"{args.trace_name}_stdout.log"
            log_blocks.append(print_section("saved log", str(stdout_log_path)))
            stdout_log_path.write_text("".join(log_blocks), encoding="utf-8")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
