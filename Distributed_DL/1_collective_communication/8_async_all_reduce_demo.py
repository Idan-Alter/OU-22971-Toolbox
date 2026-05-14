"""Compare synchronous and asynchronous all_reduce under uneven local work."""

import time

import torch
import torch.distributed as dist

from _pretty_print import print_block


TENSOR_SIZE = 8_000_000
BASE_WORK_SECONDS = 0.15
EXTRA_SLOW_RANK = 0
EXTRA_DELAY_SECONDS = 0.10


def fake_local_work(rank: int) -> float:
    delay = BASE_WORK_SECONDS + (EXTRA_DELAY_SECONDS if rank == EXTRA_SLOW_RANK else 0.0)
    time.sleep(delay)
    return delay


def gather_metrics(values: list[float]) -> list[list[float]]:
    local = torch.tensor(values, dtype=torch.float64)
    gathered = (
        [torch.zeros_like(local) for _ in range(dist.get_world_size())]
        if dist.get_rank() == 0
        else None
    )
    dist.gather(local, gather_list=gathered, dst=0)
    if gathered is None:
        return []
    return [row.tolist() for row in gathered]


def print_summary(title: str, rows: list[list[float]], columns: list[str]) -> None:
    lines = []
    for rank, row in enumerate(rows):
        metrics = ", ".join(f"{name}={value:.2f}s" for name, value in zip(columns, row))
        lines.append(f"rank {rank}: {metrics}")
    print_block(title, *lines)


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()

        tensor = torch.full((TENSOR_SIZE,), float(rank + 1), dtype=torch.float32)
        sync_start = time.perf_counter()
        dist.all_reduce(tensor)
        collective = time.perf_counter() - sync_start

        local_work = fake_local_work(rank)
        total = time.perf_counter() - sync_start
        sync_rows = gather_metrics([collective, local_work, total])
        if rank == 0:
            print_summary(
                "sync all_reduce",
                sync_rows,
                ["collective", "local_work", "total"],
            )

        dist.barrier()

        tensor = torch.full((TENSOR_SIZE,), float(rank + 1), dtype=torch.float32)
        async_start = time.perf_counter()
        work = dist.all_reduce(tensor, async_op=True)
        launch = time.perf_counter() - async_start

        local_work = fake_local_work(rank)
        wait_start = time.perf_counter()
        work.wait()
        wait = time.perf_counter() - wait_start
        total = time.perf_counter() - async_start

        async_rows = gather_metrics([launch, local_work, wait, total])
        if rank == 0:
            print_summary(
                "async all_reduce",
                async_rows,
                ["launch", "local_work", "wait", "total"],
            )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
