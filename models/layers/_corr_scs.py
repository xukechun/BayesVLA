import torch
from torch import nn, Tensor
from spatial_correlation_sampler import SpatialCorrelationSampler


class CorrKernel(nn.Module):
    def __init__(self, max_disp, dila_patch=1):
        super().__init__()
        self.max_disp = max_disp
        self.patch_size = max_disp * 2 // dila_patch + 1
        self.kernel = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.patch_size,
            stride=1,
            padding=0,
            dilation=1,
            dilation_patch=dila_patch
        )

        oy, ox = torch.meshgrid(
            torch.arange(0, self.patch_size) * dila_patch, 
            torch.arange(0, self.patch_size) * dila_patch, 
            indexing="ij"
        )

        self.register_buffer("oy", oy.flatten().long(), persistent=False)  # int64
        self.register_buffer("ox", ox.flatten().long(), persistent=False)  # int64
    
    @property
    def out_channels(self):
        return self.patch_size ** 2
    
    def valid_mask(self, fmap1_hw) -> Tensor:
        H, W = fmap1_hw
        D = self.max_disp
        cy, cx = torch.meshgrid(
            torch.arange(H).to(self.oy),
            torch.arange(W).to(self.ox),
            indexing="ij"
        )
        x = self.ox[:, None, None] + cx[None, :, :]  # (L, H, W), L = (2R/S+1)^2
        y = self.oy[:, None, None] + cy[None, :, :]  # (L, H, W), L = (2R/S+1)^2
        mask = (x >= D) & (x < W + D) & (y >= D) & (y < H + D)
        return mask  # (L, H, W), L = (2R/S+1)^2
    
    def forward(self, fmap0: Tensor, fmap1: Tensor):
        B, _, H, W = fmap0.size()
        corr: Tensor = self.kernel(fmap0, fmap1)
        corr = corr.view(B, -1, H, W)
        return corr

