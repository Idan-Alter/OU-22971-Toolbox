"""Compare reduce and all_reduce with a tiny integer tensor."""

import torch
import torch.distributed as dist

from _pretty_print import print_block


def main():
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()

        tensor = torch.tensor([rank + 1], dtype=torch.int64)
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            print_block(f"rank {rank}", f"after reduce: {tensor.tolist()}")
        else:
            print_block(
                f"rank {rank}",
                "reduce complete",
                f"local tensor is not guaranteed: {tensor.tolist()}",
            )

        tensor = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print_block(f"rank {rank}", f"after all_reduce: {tensor.tolist()}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
