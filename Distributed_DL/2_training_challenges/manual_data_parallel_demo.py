"""Run a toy manual data-parallel image classifier and summarize its costs.

Rank 0 picks one global minibatch of dataset indices and scatters one local
shard to each rank. Every rank indexes directly into a ``torchvision``
``FakeData`` dataset of small RGB images, computes forward and backward
locally, and then synchronizes gradients by hand with ``dist.all_reduce``
before the optimizer step.

The printed summaries are meant to help connect distributed training behavior
back to four recurring bottlenecks: compute, memory, communication, and
waiting. The CLI knobs let you exaggerate one pressure point at a time by
growing the batch, widening the convnet, adding extra communication, or making
one rank arrive late to synchronization.

Examples
--------
Run the baseline lesson script:

    torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py

Stress communication more than compute:

    torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --extra-sync-mb 256

Simulate one slow worker before gradient sync:

    torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --slow-rank 0 --sleep-before-sync 1.0

What to look for:
- ``training step flow`` shows the order of local work and synchronization
- ``average step summary`` shows whether both ranks slow down together
- ``memory estimate per rank`` shows which memory categories grow first
"""

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


IMAGE_SIZE = (3, 64, 64)
NUM_CLASSES = 10
MAX_CONV_BLOCKS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual data-parallel image step with simple summaries."
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
    return parser.parse_args()


# region Reporting Helpers
# Small reporting helpers keep the end-of-run summary readable.
def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Estimate optimizer-state memory by summing tensor-backed state entries."""
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def model_parameter_bytes(model: nn.Module) -> int:
    """Estimate how many bytes one model replica uses for parameters."""
    return sum(param.numel() * param.element_size() for param in model.parameters())


def model_gradient_bytes(model: nn.Module) -> int:
    """Estimate how many bytes the current gradients occupy."""
    total = 0
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.numel() * param.grad.element_size()
    return total


def format_bytes(num_bytes: int) -> str:
    """Format a byte count in MiB for the printed summaries."""
    return f"{num_bytes / (1024 ** 2):.2f} MiB"


def input_batch_bytes(batch_size: int, dtype: torch.dtype = torch.float32) -> int:
    """Estimate the bytes of one local image batch."""
    bytes_per_value = torch.tensor([], dtype=dtype).element_size()
    channels, height, width = IMAGE_SIZE
    return batch_size * channels * height * width * bytes_per_value


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


class TinyConvNet(nn.Module):
    """A small convnet whose width and depth are easy to scale for the demo."""

    def __init__(self, base_channels: int, conv_blocks: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = IMAGE_SIZE[0]
        self.block_channels: list[int] = []

        for block_idx in range(conv_blocks):
            out_channels = base_channels * (2 ** block_idx)
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            self.block_channels.append(out_channels)
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

    def activation_bytes_per_step(
        self,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> int:
        """Estimate activation memory from the batch size and convnet shape."""
        bytes_per_value = torch.tensor([], dtype=dtype).element_size()
        height = IMAGE_SIZE[1]
        width = IMAGE_SIZE[2]
        total = 0

        for out_channels in self.block_channels:
            total += batch_size * out_channels * height * width * bytes_per_value
            height = max(height // 2, 1)
            width = max(width // 2, 1)
            total += batch_size * out_channels * height * width * bytes_per_value

        total += batch_size * self.classifier.in_features * bytes_per_value
        total += batch_size * self.classifier.out_features * bytes_per_value
        return total


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
    """Let rank 0 sample global indices, shard them, and materialize one local batch."""
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

    dist.scatter(local_indices, scatter_list=scatter_list, src=0)

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


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    extra_sync_tensor: torch.Tensor | None,
    dataset: datasets.FakeData,
    local_batch_size: int,
    rank: int,
    world_size: int,
    index_generator: torch.Generator | None,
    slow_rank: int,
    sleep_before_sync: float,
) -> tuple[float, float]:
    """Run batch prep, local compute, synchronization, and return timing stats."""
    step_start = time.perf_counter()

    images, targets = prepare_local_batch(
        dataset=dataset,
        local_batch_size=local_batch_size,
        rank=rank,
        world_size=world_size,
        index_generator=index_generator,
    )

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    if slow_rank == rank and sleep_before_sync > 0:
        time.sleep(sleep_before_sync)

    manual_gradient_sync(model, world_size)

    if extra_sync_tensor is not None:
        dist.all_reduce(extra_sync_tensor, op=dist.ReduceOp.SUM)  # In-place: mutates the buffer.

    optimizer.step()

    step_time = time.perf_counter() - step_start
    return step_time, float(loss.item())


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

        step_time_total = 0.0
        loss_total = 0.0

        for _ in range(args.steps):

            step_time, loss_value = train_step(
                model=model,
                optimizer=optimizer,
                extra_sync_tensor=extra_sync_tensor,
                dataset=dataset,
                local_batch_size=args.batch_size,
                rank=rank,
                world_size=world_size,
                index_generator=index_generator,
                slow_rank=args.slow_rank,
                sleep_before_sync=args.sleep_before_sync,
            )
            step_time_total += step_time
            loss_total += loss_value

        avg_step_time = step_time_total / args.steps
        avg_loss = loss_total / args.steps
        summary_rows = gather_summaries_on_rank_zero(
            [avg_step_time, avg_loss],
            rank=rank,
            world_size=world_size,
        )

        if rank == 0:
            parameter_bytes = model_parameter_bytes(model)
            gradient_bytes = model_gradient_bytes(model)
            state_bytes = optimizer_state_bytes(optimizer)
            batch_bytes = input_batch_bytes(args.batch_size)
            activation_bytes_estimate = model.activation_bytes_per_step(args.batch_size)

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
            )

            print_section(
                "training step flow",
                "rank 0 samples global dataset indices with replacement -> scatter local batch indices",
                "each rank indexes FakeData and converts images to tensors -> forward -> loss -> backward",
                "optional sleep before sync -> gradient all_reduce -> optimizer step",
                "optional extra dummy all_reduce after gradient sync",
            )

            summary_lines = []
            for row_rank, row in enumerate(summary_rows):
                summary_lines.append(
                    f"rank {row_rank}: avg_step_time={row[0]:.4f}s, avg_loss={row[1]:.6f}"
                )
            print_section("average step summary", *summary_lines)

            print_section(
                "memory estimate per rank",
                f"parameters={format_bytes(parameter_bytes)}",
                f"gradients={format_bytes(gradient_bytes)}",
                f"optimizer_state={format_bytes(state_bytes)}",
                f"input_batch={format_bytes(batch_bytes)}",
                f"activations_estimate={format_bytes(activation_bytes_estimate)}",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
