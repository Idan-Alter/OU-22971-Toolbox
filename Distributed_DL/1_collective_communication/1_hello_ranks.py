"""Print each rank's identity inside a simple distributed job."""

import os

import torch
import torch.distributed as dist

from _pretty_print import print_block


def main():
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        tensor = torch.tensor([rank], dtype=torch.int64)
        print_block(
            f"rank {rank}",
            f"local_rank={local_rank}",
            f"world_size={world_size}",
            f"local_tensor={tensor.tolist()}",
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
