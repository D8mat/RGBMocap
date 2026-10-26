"""Learning-rate schedulers local to HTD-Refine optimization."""

from __future__ import annotations

from bisect import bisect_right

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler.LRScheduler):
    """Multi-step LR schedule with linear warmup."""

    def __init__(self, optimizer, milestones, warmup=0, gamma=0.1, last_epoch=-1, verbose="deprecated"):
        if warmup >= milestones[0]:
            raise ValueError(
                f"warmup must be smaller than first milestone, got warmup={warmup}, milestones={milestones}"
            )
        self.milestones = milestones
        self.warmup = warmup
        self.gamma = gamma
        super().__init__(optimizer, last_epoch=last_epoch)  # PyTorch 2.7 removed verbose.

    def get_lr(self):
        if self.last_epoch < self.warmup:
            factor = (self.last_epoch + 1) / (self.warmup + 1)
            return [base_lr * factor for base_lr in self.base_lrs]
        factor = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]


__all__ = ["WarmupMultiStepLR"]
