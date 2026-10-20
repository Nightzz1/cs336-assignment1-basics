"""Element-wise activations, softmax, and cross-entropy loss."""

import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # implement the cross entropy loss function here from scratch
    # logits: (batch_size, vocab_size): logits[i][j] is the unnormalized logit of jth class for the ith example
    # targets: (batch_size,): targets[i] is the true class for the ith example
    # return: (batch_size,): the average cross-entropy loss across examples.

    # Subtract the largest element for numerical stability.
    # Cancel out log and exp whenever possible.
    # Handle any additional batch dimensions and return the average across the batch. We assume batch-like dimensions always come first, before the vocabulary size dimension.

    # Compute the log softmax of the logits
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logits
    log_probs = stable_logits - torch.log(torch.sum(torch.exp(stable_logits), dim=-1, keepdim=True))
    return -log_probs[torch.arange(log_probs.size(0)), targets].mean()
