import torch.nn as nn
import torch
import torch.nn.functional as F

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
       
    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)).log_softmax(-1), q.view(-1, q.size(-1)).log_softmax(-1)
        m = (0.5 * (p + q))
        return 0.5 * (self.kl(m, p) + self.kl(m, q))
    

def total_variation_distance_from_logits(logits_p, logits_q):
    """
    Calculate the Total Variation Distance between two probability distributions.

    Args:
        logits_p (torch.Tensor): The logits (raw network outputs) for the first distribution.
        logits_q (torch.Tensor): The logits (raw network outputs) for the second distribution.

    Returns:
        torch.Tensor: The Total Variation Distance between the probability distributions.
    """
    # Convert logits to probabilities using the softmax function
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)

    # Calculate the Total Variation Distance
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return 0.5 * torch.sum(torch.abs(cdf_p - cdf_q))