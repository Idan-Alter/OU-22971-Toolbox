"""Demonstrate scattering one tensor shard from rank 0 to each rank."""

import torch
import torch.distributed as dist

from _pretty_print import print_block


def main():
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()

        if rank == 0:
            scatter_list = [
                torch.tensor([10], dtype=torch.int64),
                torch.tensor([20], dtype=torch.int64),
                torch.tensor([30], dtype=torch.int64),
                torch.tensor([40], dtype=torch.int64),
            ]
        else:
            scatter_list = None

        tensor = torch.zeros(1, dtype=torch.int64)
        dist.scatter(tensor, scatter_list=scatter_list, src=0)
        print_block(f"rank {rank}", f"after scatter: {tensor.tolist()}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
