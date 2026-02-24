import os
import torch
import inspect
import taichi as ti
from torch import nn, Tensor
import torch.nn.functional as F


@ti.data_oriented
class _CorrKernel(object):

    PARALLELIZE = min(os.cpu_count(), 64)
    FWD_BLOCK_DIM = 64
    BWD_BLOCK_DIM = 32
    USE_SHARED_ARRAY = True

    def __init__(self):
        self.dtype = ti.float32

    def config_shared_array_dtype(self, dtype):
        self.dtype = dtype

    @ti.kernel
    def forward(
        self,
        ox: ti.types.ndarray(dtype=ti.i32, ndim=1),  # type: ignore comment
        oy: ti.types.ndarray(dtype=ti.i32, ndim=1),  # type: ignore comment
        fmap0: ti.types.ndarray(ndim=4),             # type: ignore comment
        fmap1_pad: ti.types.ndarray(ndim=4),         # type: ignore comment
        corr: ti.types.ndarray(ndim=4),              # type: ignore comment
    ):
        """
        Arguments:
        - ox: (L,), offset x
        - oy: (L,), offset y
        - fmap0: (B, H, W, C)
        - fmap1_pad: (B, Hp, Wp, C)
        - corr: (B, L, H, W)
        """
        L = ox.shape[0]
        B, H, W, C = fmap0.shape

        ti.loop_config(
            block_dim=self.FWD_BLOCK_DIM,
            parallelize=self.PARALLELIZE
        )

        for b, i0, j0, l in ti.ndrange(B, H, W, L):
            i1 = i0 + oy[l]
            j1 = j0 + ox[l]
            dot_sum = fmap0[b, i0, j0, 0] * 0
            for c in range(C):
                dot_sum += fmap0[b, i0, j0, c] * fmap1_pad[b, i1, j1, c]
            corr[b, l, i0, j0] = dot_sum

    @ti.kernel
    def forward_shared(
        self,
        ox: ti.types.ndarray(dtype=ti.i32, ndim=1),  # type: ignore comment
        oy: ti.types.ndarray(dtype=ti.i32, ndim=1),  # type: ignore comment
        fmap0: ti.types.ndarray(ndim=4),             # type: ignore comment
        fmap1_pad: ti.types.ndarray(ndim=4),         # type: ignore comment
        corr: ti.types.ndarray(ndim=4),              # type: ignore comment
    ):
        """
        Arguments:
        - ox: (L,), offset x
        - oy: (L,), offset y
        - fmap0: (B, H, W, C)
        - fmap1_pad: (B, Hp, Wp, C)
        - corr: (B, L, H, W)
        """
        L = ox.shape[0]
        B, H, W, C = fmap0.shape

        ti.loop_config(block_dim=self.FWD_BLOCK_DIM)
        num_c_partitions = C//self.FWD_BLOCK_DIM + (C%self.FWD_BLOCK_DIM > 0)

        for b, i0, j0, thread_idx in ti.ndrange(B, H, W, self.FWD_BLOCK_DIM):
            prod_sum = ti.simt.block.SharedArray((self.FWD_BLOCK_DIM,), self.dtype)

            for l in range(L):
                oyl = oy[l]; oxl = ox[l]
                prod_sum[thread_idx] = 0

                for c_part_id in range(num_c_partitions):
                    c = c_part_id * self.FWD_BLOCK_DIM + thread_idx
                    if c < C:
                        prod_sum[thread_idx] += fmap0[b, i0, j0, c] * fmap1_pad[b, i0+oyl, j0+oxl, c]
                
                ti.simt.block.sync()
                if thread_idx == 0:
                    reduce_sum = ti.cast(0, self.dtype)
                    for tids in range(self.FWD_BLOCK_DIM):
                        reduce_sum += prod_sum[tids]
                    corr[b, l, i0, j0] = reduce_sum
                ti.simt.block.sync()

    @ti.kernel
    def backward_grad1(
        self,
        ox: ti.types.ndarray(dtype=ti.i32, ndim=1),  # type: ignore comment
        oy: ti.types.ndarray(dtype=ti.i32, ndim=1),  # type: ignore comment
        fmap1_pad: ti.types.ndarray(ndim=4),         # type: ignore comment
        grad_corr: ti.types.ndarray(ndim=4),         # type: ignore comment
        grad_fmap0: ti.types.ndarray(ndim=4),        # type: ignore comment
    ):
        """
        Arguments:
        - ox: (L,)
        - oy: (L,)
        - fmap1_pad: (B, C, Hp, Wp)
        - grad_corr: (B, L, H, W)
        - grad_fmap0: (B, C, H, W)
        """
        L = ox.shape[0]
        B, C, H, W = grad_fmap0.shape

        ti.loop_config(
            block_dim=self.BWD_BLOCK_DIM,
            parallelize=self.PARALLELIZE
        )

        for b, c, l in ti.ndrange(B, C, L):
            oyl = oy[l]; oxl = ox[l]
            for i0, j0 in ti.ndrange(H, W):
                ti.atomic_add(
                    grad_fmap0[b, c, i0, j0],
                    grad_corr[b, l, i0, j0] * fmap1_pad[b, c, i0+oyl, j0+oxl]
                )

    @ti.kernel
    def backward_grad2(
        self,
        ox: ti.types.ndarray(dtype=ti.i32, ndim=1),  # type: ignore comment
        oy: ti.types.ndarray(dtype=ti.i32, ndim=1),  # type: ignore comment
        fmap0: ti.types.ndarray(ndim=4),             # type: ignore comment
        grad_corr: ti.types.ndarray(ndim=4),         # type: ignore comment
        grad_fmap1_pad: ti.types.ndarray(ndim=4),    # type: ignore comment
    ):
        """
        Arguments:
        - ox: (L,)
        - oy: (L,)
        - fmap0: (B, C, H, W)
        - grad_corr: (B, L, H, W)
        - grad_fmap1_pad: (B, C, Hp, Wp)
        """
        L = ox.shape[0]
        B, C, H, W = fmap0.shape

        ti.loop_config(
            block_dim=self.BWD_BLOCK_DIM,
            parallelize=self.PARALLELIZE
        )

        for b, c, l in ti.ndrange(B, C, L):
            oyl = oy[l]; oxl = ox[l]
            for i0, j0 in ti.ndrange(H, W):
                ti.atomic_add(
                    grad_fmap1_pad[b, c, i0+oyl, j0+oxl],
                    grad_corr[b, l, i0, j0] * fmap0[b, c, i0, j0]
                )


dtype_mapping = {
    torch.float16: ti.float16,
    torch.float32: ti.float32,
    torch.float64: ti.float64,
    torch.uint8: ti.uint8,
    torch.int8: ti.int8,
    torch.int16: ti.int16,
    torch.int32: ti.int32,
    torch.int64: ti.int64
}

class _CorrFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        fmap0: Tensor, 
        fmap1_pad: Tensor, 
        ox: Tensor, 
        oy: Tensor, 
        kernel: _CorrKernel
    ):
        """
        Arguments:
        - fmap0: (B, C, H, W)
        - fmap1_pad: (B, C, Hp, Wp)
        - ox: (L,), offset x
        - oy: (L,), offset y
        """
        ctx.save_for_backward(fmap0, fmap1_pad, ox, oy)
        ctx.kernel = kernel

        L = ox.size(0)
        B, _, H, W = fmap0.size()

        fmap0_rearrange = fmap0.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        fmap1_pad_rearange = fmap1_pad.permute(0, 2, 3, 1).contiguous()  # (B, Hp, Wp, C)

        corr = fmap0.new_zeros((B, L, H, W), requires_grad=False)  # (B, L, H, W)
        if fmap0.device != torch.device("cpu") and kernel.USE_SHARED_ARRAY:
            # enable simd shared array if specified and runs on cuda
            # config is effective only at the first run
            kernel.config_shared_array_dtype(dtype=dtype_mapping[fmap0.dtype])
            kernel.forward_shared(ox, oy, fmap0_rearrange, fmap1_pad_rearange, corr)
        else:
            kernel.forward(ox, oy, fmap0_rearrange, fmap1_pad_rearange, corr)

        return corr

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        grad_out = grad_out.contiguous()  # (B, L, H, W)
        fmap0, fmap1_pad, ox, oy = ctx.saved_tensors
        kernel: _CorrKernel = ctx.kernel

        if ctx.needs_input_grad[0]:
            grad_fmap0 = torch.zeros_like(fmap0, requires_grad=False)
            kernel.backward_grad1(ox, oy, fmap1_pad, grad_out, grad_fmap0)
        else:
            grad_fmap0 = None
        
        if ctx.needs_input_grad[1]:
            grad_fmap1_pad = torch.zeros_like(fmap1_pad, requires_grad=False)
            kernel.backward_grad2(ox, oy, fmap0, grad_out, grad_fmap1_pad)
        else:
            grad_fmap1_pad = None
        
        return grad_fmap0, grad_fmap1_pad, None, None, None


class CorrKernel(nn.Module):
    def __init__(self, max_disp, dila_patch=1):
        """
        Arguments:
        - max_disp: maximum displacement
        - dila_patch: dilation on patch
        """
        super(CorrKernel, self).__init__()
        self.max_disp = max_disp
        self.dila_patch = dila_patch

        patch_size = max_disp * 2 // dila_patch + 1
        pad_l = pad_t = pad_r = pad_b = max_disp

        self.patch_size = patch_size
        self.pad_size = (pad_l, pad_r, pad_t, pad_b)

        meshgrid_need_index = "indexing" in inspect.getfullargspec(torch.meshgrid).kwonlyargs
        self.meshgrid_kwargs = {"indexing": "ij"} if meshgrid_need_index else {}
        oy, ox = torch.meshgrid(
            torch.arange(0, patch_size) * dila_patch, 
            torch.arange(0, patch_size) * dila_patch, 
            **self.meshgrid_kwargs
        )

        self.register_buffer("oy", oy.flatten().int(), persistent=False)  # int32
        self.register_buffer("ox", ox.flatten().int(), persistent=False)  # int32
        self.kernel = _CorrKernel()

    @property
    def out_channels(self):
        return self.patch_size ** 2

    def valid_mask(self, fmap1_hw) -> Tensor:
        H, W = fmap1_hw
        D = self.max_disp
        cy, cx = torch.meshgrid(
            torch.arange(H).to(self.oy),
            torch.arange(W).to(self.ox),
            **self.meshgrid_kwargs
        )
        x = self.ox[:, None, None] + cx[None, :, :]  # (L, H, W), L = (2R/S+1)^2
        y = self.oy[:, None, None] + cy[None, :, :]  # (L, H, W), L = (2R/S+1)^2
        mask = (x >= D) & (x < W + D) & (y >= D) & (y < H + D)
        return mask  # (L, H, W), L = (2R/S+1)^2

    def forward(self, fmap0: Tensor, fmap1: Tensor) -> Tensor:
        """
        Arguments:
        - fmap0: (B, C, H, W)
        - fmap1: (B, C, H, W)
        
        Returns:
        - corr: (B, L, H, W), L = (2*R//S + 1)^2
        """
        fmap1_pad = F.pad(fmap1, self.pad_size, "constant", 0)
        corr = _CorrFunction.apply(fmap0, fmap1_pad, self.ox, self.oy, self.kernel)
        return corr


if __name__ == "__main__":
    ti.init(ti.cpu)

    B, C, H, W = 1, 1, 4, 4
    fmap0 = torch.ones(B, C, H, W)
    fmap1 = torch.ones(B, C, H, W)
    kernel = CorrKernel(3, 2)

    corr = kernel(fmap0, fmap1)
    mask = kernel.valid_mask((H, W))
    P = kernel.patch_size

    for i in range(H):
        for j in range(W):
            print("---------- i = {}, j = {}".format(i, j))
            print(corr[0, :, i, j].reshape(P, P))
            print(mask[:, i, j].reshape(P, P).float())
