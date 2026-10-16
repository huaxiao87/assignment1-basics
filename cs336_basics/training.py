import torch
import cs336_basics.transformer as transformer
from jaxtyping import Float, Int
from torch import Tensor
import math
from typing import Optional, Callable, Iterable
import numpy.typing as npt
import numpy as np
import os
def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    # Subtract the max logit from each logit
    logits = inputs - inputs.max(dim=-1, keepdim=True).values
    # Get the logits for the targets
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Log sum exp trick
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))
    # Compute the cross-entropy loss, averaged over the batch
    return (log_sum_exp - target_logits).mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
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
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e1)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                state["t"] += 1
                bias_correction1 = 1 - betas[0] ** state["t"] # 1- beta1^t
                bias_correction2 = 1 - betas[1] ** state["t"] # 1- beta2^t
                lr_corrected = lr * math.sqrt(bias_correction2) / bias_correction1 # Corrected learning rate
                if weight_decay != 0:
                    p.data -= weight_decay * lr * p.data # Apply weight decay
                m.mul_(betas[0]).add_(p.grad, alpha=1 - betas[0]) # Update the running average of the gradient
                v.mul_(betas[1]).addcmul_(p.grad, p.grad, value=1 - betas[1]) # Update the running average of the squared gradient
                p.data -= lr_corrected * m / (v.sqrt() + eps) # Update the parameters
        return loss

def lr_cosine_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if t < warmup_iters:
        return t * lr_max / warmup_iters
    elif t <= cosine_cycle_iters:
        return lr_min + 1/2 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    else:
        return lr_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    l2_norm = 0.0
    # materialize the generator into a list
    parameters_list = list(parameters)
    for p in parameters_list:
        if p.grad is not None:
            l2_norm += p.grad.data.norm()**2
    l2_norm = math.sqrt(l2_norm)
    if l2_norm > max_l2_norm:
        for p in parameters_list:
            if p.grad is not None:
                p.grad.data.div_(l2_norm / max_l2_norm + 1e-6)

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Generate the start indices of the training sequences any 1≤i≤n−m gives a valid training sequence
    start_indices = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))
    # Generate consecutive offsets of size (context_length)
    offsets = np.arange(context_length)
    # Combine the start indices and offsets to get the training sequences
    training_sequences = start_indices.reshape(-1, 1) + offsets.reshape(1, -1)
    # Combine the start indices and offsets to get the target sequences
    target_sequences = training_sequences + 1
    # Convert the training sequences and target sequences to tensors
    training_sequences = torch.tensor(dataset[training_sequences], device=device)
    target_sequences = torch.tensor(dataset[target_sequences], device=device)
    return training_sequences, target_sequences

def save_checkpoint(model, optimizer, iteration, out: str):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration, # To resume learning rate scheduling
    }, out)

def load_checkpoint(source: str, model, optimizer):
    checkpoint = torch.load(source)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]