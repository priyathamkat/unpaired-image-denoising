import torch
import torch.nn.functional as F
import torch.distributions as D

cuda_device = torch.device("cuda:0")

prior = D.normal.Normal(0, 1)

mse_loss = F.mse_loss

def nll_loss(zs, log_det):
    batch_size = zs[0].shape[0]
    nll = torch.zeros(batch_size).to(cuda_device)
    for z in zs:
        nll = nll - prior.log_prob(z).flatten(1).sum(-1)
    nll = nll - log_det
    return nll.mean()
