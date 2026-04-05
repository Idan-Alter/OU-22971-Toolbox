"""Run the smallest possible DistributedDataParallel training step for Unit 0.

This script is the quick environment check for the rest of the part. Each rank
builds the same tiny MLP, runs one forward pass, one backward pass, and one
optimizer step under ``DistributedDataParallel``, then rank 0 gathers a short
status summary from every process.

On a GPU machine it prefers CUDA with NCCL. Inside the local CPU-only dev
container it falls back to CPU with Gloo, so the same command is still useful
before moving on to the later lessons.

Examples
--------
From ``/workspace`` inside the dev container:

    torchrun --standalone --nproc_per_node=2 0_devcontainer_setup/1_ddp_smoke_test.py

What to look for:
- rank 0 prints one combined summary block
- each rank reports the same ``world_size``
- every rank ends with ``ddp_step=ok``
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > local_rank
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    return rank, local_rank, world_size, device, backend


def format_status_report(statuses):
    lines = ["DDP smoke test summary", "-" * 21]
    for status in sorted(statuses, key=lambda item: item["rank"]):
        lines.extend(
            [
                f"rank {status['rank']}",
                f"  local_rank={status['local_rank']} world_size={status['world_size']}",
                f"  backend={status['backend']} device={status['device']}",
                f"  loss={status['loss']:.6f} ddp_step=ok",
            ]
        )
    return "\n".join(lines)


def main():
    rank, local_rank, world_size, device, backend = setup()
    try:
        torch.manual_seed(22971)

        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        ).to(device)

        if device.type == "cuda":
            ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            ddp_model = DDP(model)

        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

        torch.manual_seed(22971 + rank)
        inputs = torch.randn(16, 8, device=device)
        targets = torch.randn(16, 4, device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = ddp_model(inputs)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        dist.barrier()

        status = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "backend": backend,
            "device": str(device),
            "loss": loss.item(),
        }
        statuses = [None] * world_size
        dist.all_gather_object(statuses, status)

        if rank == 0:
            print(format_status_report(statuses), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
