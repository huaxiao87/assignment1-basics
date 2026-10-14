import torch
import cs336_basics.transformer as transformer
from jaxtyping import Float, Int
from torch import Tensor
import math
from typing import Optional, Callable

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