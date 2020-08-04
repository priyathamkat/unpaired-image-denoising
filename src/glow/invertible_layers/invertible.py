import torch
import torch.nn as nn


class InvertibleModule(nn.Module):
    @torch.no_grad()
    def invert(self, y, log_det):
        raise NotImplementedError


class InvertibleSequential(nn.Sequential):
    def forward(self, x, log_det):
        for module in self._modules.values():
            x, log_det = module(x, log_det)
        return x, log_det

    @torch.no_grad()
    def invert(self, y, log_det):
        for module in reversed(self._modules.values()):
            y, log_det = module.invert(y, log_det)
        return y, log_det


def hook(module, input, output):
    if output[0].abs().max() / input[0].abs().max() > 10:
        print("s: ", module.s.max())
        for name, param in module.named_parameters():
            print(name, param.data.abs().max())
        print("input: ", input[0].abs().max())
        print("output: ", output[0].abs().max())
