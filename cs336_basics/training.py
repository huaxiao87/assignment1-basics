import torch
import cs336_basics.transformer as transformer
from jaxtyping import Float, Int
from torch import Tensor

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    # Subtract the max logit from each logit
    logits = inputs - inputs.max(dim=-1, keepdim=True).values
    # Get the logits for the targets
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Log sum exp trick
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))
    # Compute the cross-entropy loss, averaged over the batch
    return (log_sum_exp - target_logits).mean()

