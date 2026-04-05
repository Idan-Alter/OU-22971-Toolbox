"""Profile the manual data-parallel image step and export CPU trace files.

Rank 0 picks one global minibatch of dataset indices and scatters one local
shard to each rank. Every rank indexes directly into a ``torchvision``
``FakeData`` dataset, computes forward and backward locally, and then
synchronizes gradients by hand with ``dist.all_reduce`` before the optimizer
step.

This profiler version keeps that same Unit 2 training step logic, but wraps it
with the PyTorch profiler.

Examples
--------
Capture a baseline trace:

    torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name baseline

Capture a waiting-heavy trace:

    torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name slow_rank --slow-rank 0 --sleep-before-sync 1.0

Stress communication more than compute:

    torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name extra_sync --extra-sync-mb 256

What to look for:
- trace JSON files appear under ``3_profiler_cpu_traces/traces``
"""

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision import datasets, transforms


IMAGE_SIZE = (3, 64, 64)
NUM_CLASSES = 10
MAX_CONV_BLOCKS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the toy manual data-parallel image step and export CPU traces."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--conv-blocks", type=int, default=5)
    parser.add_argument("--dataset-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--extra-sync-mb", type=float, default=0.0)
    parser.add_argument("--slow-rank", type=int, default=-1)
    parser.add_argument("--sleep-before-sync", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace-dir", type=str, default="3_profiler_cpu_traces/traces")
    parser.add_argument("--trace-name", type=str, default="baseline")
    parser.add_argument("--profile-memory", action="store_true")
    return parser.parse_args()


# region Reporting Helpers
# Small reporting helpers keep the end-of-run summary readable.
def gather_summaries_on_rank_zero(
    local_values: list[float],
    rank: int,
    world_size: int,
) -> list[list[float]] | None:
    """Gather one small summary vector from every rank onto rank 0 only."""
    local_tensor = torch.tensor(local_values, dtype=torch.float64)
    if rank == 0:
        gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    else:
        gathered = None
    dist.gather(local_tensor, gather_list=gathered, dst=0)
    if rank == 0:
        return [row.tolist() for row in gathered]
    return None


def print_section(title: str, *lines: str) -> None:
    """Print one labeled block so the console output reads like a report."""
    body = "\n".join([title, *[f"  {line}" for line in lines], ""])
    print(body, flush=True)


# endregion


# region Same DL logic as in Unit 2
class TinyConvNet(nn.Module):
    """A small convnet whose width and depth are easy to scale for the demo."""

    def __init__(self, base_channels: int, conv_blocks: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = IMAGE_SIZE[0]

        for block_idx in range(conv_blocks):
            out_channels = base_channels * (2 ** block_idx)
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def build_fake_image_dataset(dataset_size: int, seed: int) -> datasets.FakeData:
    """Build one deterministic fake image dataset per rank.

    ``datasets.FakeData`` is lazy, so it does not materialize every sample up
    front. Instead, it generates each image/label pair deterministically when
    that index is requested.
    """
    return datasets.FakeData(
        size=dataset_size,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        transform=transforms.ToTensor(),
        random_offset=seed,
    )


def prepare_local_batch(
    dataset: datasets.FakeData,
    local_batch_size: int,
    rank: int,
    world_size: int,
    index_generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Let rank 0 sample a global minibatch, shard it, and materialize one local batch."""
    with record_function("plan_batch_indices"):
        local_indices = torch.empty(local_batch_size, dtype=torch.int64)

        if rank == 0:
            global_batch_size = local_batch_size * world_size
            # Teaching shortcut: sample with replacement via randint so the manual
            # data-parallel demo stays simple and avoids the extra full-dataset
            # randperm cost that could skew lightweight profiling.
            global_indices = torch.randint(
                high=len(dataset),
                size=(global_batch_size,),
                generator=index_generator,
            )
            scatter_list = list(global_indices.chunk(world_size))
        else:
            scatter_list = None
    #NEW in Unit 3:
    with record_function("scatter_batch_indices"):
        dist.scatter(local_indices, scatter_list=scatter_list, src=0)
    #NEW in Unit 3:
    with record_function("index_batch"):
        images: list[torch.Tensor] = []
        targets: list[int] = []
        for dataset_index in local_indices.tolist():
            image, target = dataset[int(dataset_index)]
            images.append(image)
            targets.append(target)
        return torch.stack(images, dim=0), torch.tensor(targets, dtype=torch.long)


def manual_gradient_sync(model: nn.Module, world_size: int) -> None:
    """Average gradients across ranks so every rank holds the same grads before the optimizer step."""
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size


# endregion


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    extra_sync_tensor: torch.Tensor | None,
    args: argparse.Namespace,
    dataset: datasets.FakeData,
    rank: int,
    world_size: int,
    index_generator: torch.Generator | None,
) -> float:
    #NEW in Unit 3:
    with record_function("train_step"):
        with record_function("next_batch"):
            images, targets = prepare_local_batch(
                dataset=dataset,
                local_batch_size=args.batch_size,
                rank=rank,
                world_size=world_size,
                index_generator=index_generator,
            )

        optimizer.zero_grad(set_to_none=True)

        with record_function("forward"):
            logits = model(images)

        with record_function("loss"):
            loss = F.cross_entropy(logits, targets)

        with record_function("backward"):
            loss.backward()

        with record_function("sleep_before_sync"):
            if args.slow_rank == rank and args.sleep_before_sync > 0:
                time.sleep(args.sleep_before_sync)

        with record_function("grad_sync"):
            manual_gradient_sync(model, world_size)

        if extra_sync_tensor is not None:
            with record_function("extra_sync"):
                dist.all_reduce(extra_sync_tensor, op=dist.ReduceOp.SUM)  # In-place: mutates the buffer.

        with record_function("optimizer_step"):
            optimizer.step()

    return float(loss.item())


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.base_channels < 1:
        raise SystemExit("--base-channels must be at least 1.")
    if args.conv_blocks < 1:
        raise SystemExit("--conv-blocks must be at least 1.")
    if args.conv_blocks > MAX_CONV_BLOCKS:
        raise SystemExit(
            f"--conv-blocks must be at most {MAX_CONV_BLOCKS} for {IMAGE_SIZE[1]}x{IMAGE_SIZE[2]} inputs."
        )
    if args.dataset_size < 1:
        raise SystemExit("--dataset-size must be at least 1.")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1.")

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        torch.manual_seed(args.seed)

        dataset = build_fake_image_dataset(
            dataset_size=args.dataset_size,
            seed=args.seed,
        )
        index_generator = torch.Generator().manual_seed(args.seed) if rank == 0 else None
        model = TinyConvNet(args.base_channels, args.conv_blocks)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

        if args.extra_sync_mb > 0:
            numel = max(1, int(args.extra_sync_mb * 1024 * 1024 / 4))
            extra_sync_tensor = torch.ones(numel, dtype=torch.float32)
        else:
            extra_sync_tensor = None

        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{args.trace_name}_rank{rank}.json"
        warmup_steps = max(args.steps - 2, 0)
        profiled_steps = min(args.steps, 2)
        loss_value = 0.0

        for _ in range(warmup_steps):
            loss_value = train_step(
                model=model,
                optimizer=optimizer,
                extra_sync_tensor=extra_sync_tensor,
                args=args,
                dataset=dataset,
                rank=rank,
                world_size=world_size,
                index_generator=index_generator,
            )
        #NEW in Unit 3:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=args.profile_memory,
        ) as prof:
            for _ in range(profiled_steps):
                loss_value = train_step(
                    model=model,
                    optimizer=optimizer,
                    extra_sync_tensor=extra_sync_tensor,
                    args=args,
                    dataset=dataset,
                    rank=rank,
                    world_size=world_size,
                    index_generator=index_generator,
                )
        #NEW in Unit 3:
        prof.export_chrome_trace(str(trace_path))

        summary_rows = gather_summaries_on_rank_zero(
            [loss_value],
            rank=rank,
            world_size=world_size,
        )

        if rank == 0:
            print_section(
                "config",
                f"world_size={world_size}",
                f"steps={args.steps}",
                f"batch_size={args.batch_size}",
                f"dataset_size={args.dataset_size}",
                f"image_size={IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x{IMAGE_SIZE[2]}",
                f"num_classes={NUM_CLASSES}",
                f"base_channels={args.base_channels}",
                f"conv_blocks={args.conv_blocks}",
                f"extra_sync_mb={args.extra_sync_mb:.2f}",
                f"slow_rank={args.slow_rank}",
                f"sleep_before_sync={args.sleep_before_sync:.2f}s",
                f"trace_name={args.trace_name}",
                f"manual_warmup_steps={warmup_steps}",
                f"profiled_steps={profiled_steps}",
                "record_shapes=True",
                f"profile_memory={args.profile_memory}",
            )

            print_section(
                "trace labels to look for",
                "train_step",
                "next_batch",
                "index_batch",
                "plan_batch_indices",
                "scatter_batch_indices",
                "forward",
                "loss",
                "backward",
                "sleep_before_sync",
                "grad_sync",
                "extra_sync",
                "optimizer_step",
            )

            summary_lines = []
            for row_rank, row in enumerate(summary_rows):
                summary_lines.append(f"rank {row_rank}: final_loss={row[0]:.6f}")
            print_section("loss summary", *summary_lines)

            trace_lines = [
                str(trace_dir / f"{args.trace_name}_rank{trace_rank}.json")
                for trace_rank in range(world_size)
            ]
            print_section("trace files", *trace_lines)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
