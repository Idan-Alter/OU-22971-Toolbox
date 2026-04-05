"""Contrast gather on rank 0 with all_gather on every rank."""

import torch
import torch.distributed as dist

from _pretty_print import print_block


def main():
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        tensor = torch.tensor([rank], dtype=torch.int64)

        if rank == 0:
            gather_list = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
        else:
            gather_list = None
        dist.gather(tensor, gather_list=gather_list, dst=0)
        if rank == 0:
            gathered = [item.tolist() for item in gather_list]
            print_block(f"rank {rank}", f"after gather: {gathered}")

        tensor_list = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        gathered = [item.tolist() for item in tensor_list]
        print_block(f"rank {rank}", f"after all_gather: {gathered}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
