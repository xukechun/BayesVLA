import torch
import torch.nn.functional as F
from torch import nn, Tensor


# class RMSNorm(nn.Module):
#     def __init__(self, dim, eps=1e-6, unit_offset=False):
#         super().__init__()
#         self.unit_offset = unit_offset
#         self.scale = dim ** 0.5
#         self.eps = eps

#         self.g = nn.Parameter(torch.zeros(dim))
#         nn.init.constant_(self.g, 1. - float(unit_offset))

#     def forward(self, x):
#         gamma = self.g + float(self.unit_offset)
#         return F.normalize(x, dim=-1, eps=self.eps) * self.scale * gamma

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)
    
    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        # to float32 first for numeric issues
        output = self._norm(x.float()).type_as(x)
        return (output * self.weight) if self.elementwise_affine else output
