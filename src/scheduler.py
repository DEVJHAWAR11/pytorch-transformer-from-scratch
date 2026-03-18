import torch


class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _compute_lr(self):
        step = self.current_step
        d    = self.d_model
        warm = self.warmup_steps
        return (d ** -0.5) * min(step ** -0.5, step * warm ** -1.5)
