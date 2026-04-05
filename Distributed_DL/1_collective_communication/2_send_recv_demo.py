"""Demonstrate point-to-point send and receive between two ranks."""

import torch
import torch.distributed as dist

from _pretty_print import print_block


def main():
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()

        tensor = torch.zeros(1)

        if rank == 0:
            tensor += 1
            dist.send(tensor=tensor, dst=1)
            print_block(
                f"rank {rank}",
                f"sent {tensor.tolist()}",
                "destination=rank 1",
            )
        elif rank == 1:
            dist.recv(tensor=tensor, src=0)
            print_block(
                f"rank {rank}",
                f"received {tensor.tolist()}",
                "source=rank 0",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
