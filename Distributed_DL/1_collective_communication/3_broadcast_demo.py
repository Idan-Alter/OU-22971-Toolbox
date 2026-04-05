"""Show how broadcast copies one rank's tensor to every other rank."""

import torch
import torch.distributed as dist

from _pretty_print import print_block


def main():
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()

        tensor = torch.tensor([42 if rank == 0 else -1], dtype=torch.int64)
        print_block(f"rank {rank}", f"before broadcast: {tensor.tolist()}")
        dist.broadcast(tensor, src=0)
        print_block(f"rank {rank}", f"after broadcast: {tensor.tolist()}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
