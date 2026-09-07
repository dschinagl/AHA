import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return (not self.is_distributed) or (self.rank == 0)

    @property
    def device(self) -> torch.device:
        if self.is_distributed:
            return torch.device("cuda", self.local_rank)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def init() -> "DistributedContext":
        is_dist = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
        if not is_dist:
            return DistributedContext(is_distributed=False, rank=0, local_rank=0, world_size=1)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        return DistributedContext(
            is_distributed=True,
            rank=dist.get_rank(),
            local_rank=local_rank,
            world_size=dist.get_world_size(),
        )

    def destroy(self) -> None:
        if self.is_distributed:
            dist.destroy_process_group()

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.is_distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / self.world_size
        return tensor

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.is_distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor
