from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def learning_rate_schedule(t: int, ln_max: float, ln_min: float, warmup_steps: int, cosine_annealing_steps: int) -> float:
    if t < warmup_steps:
        return t * ln_max / warmup_steps
    elif t <= cosine_annealing_steps:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (t - warmup_steps) / (cosine_annealing_steps - warmup_steps)))
        return ln_min + (ln_max - ln_min) * cosine_decay
    else:
        return ln_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                # Initialize m and v if they don't exist.
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]

                # Update biased first moment estimate.
                m.mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])
                # Update biased second raw moment estimate.
                v.mul_(self.betas[1]).addcmul_(grad, grad, value=1 - self.betas[1])

                # Compute bias-corrected first moment estimate.
                m_hat = m / (1 - self.betas[0] ** (t + 1))
                # Compute bias-corrected second raw moment estimate.
                v_hat = v / (1 - self.betas[1] ** (t + 1))

                # Update parameters with weight decay.
                p.data -= lr * (m_hat / (torch.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)

                state["t"] = t + 1

        return loss