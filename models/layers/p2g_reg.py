import torch
import taichi as ti
from torch import nn, Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx, once_differentiable


@ti.data_oriented
class P2GKernel(object):
    FWD_BLOCK_DIM = 32
    BWD_BLOCK_DIM = 32

    def __init__(self, grid_num, dx, ndim, dtype=ti.f32):
        self.dx = dx
        self.ndim = ndim
        self.dtype = dtype
        self.grid_num = grid_num
        self.neighbor = ((-1, 2),) * ndim
    
    def config_dtype(self, dtype):
        self.dtype = dtype
    
    @ti.kernel
    def p2g_forward_kernel(
        self,
        pos: ti.types.ndarray(ndim=3),              # (B, N_points, ndim) # type: ignore
        prob: ti.types.ndarray(ndim=2),             # (B, N_points)       # type: ignore
        weight_out: ti.types.ndarray(ndim=2),       # (B, grid_num^ndim)  # type: ignore
        weight_prob_out: ti.types.ndarray(ndim=2),  # (B, grid_num^ndim)  # type: ignore
    ):
        B, Np, _ = pos.shape
        ti.loop_config(block_dim=self.FWD_BLOCK_DIM)
        for b, i in ti.ndrange(B, Np):
            Xp = ti.Vector([pos[b, i, d] for d in 
                            ti.static(range(self.ndim))]) / self.dx
            base = ti.cast(Xp, ti.i32)
            fx = Xp - base
            w = [0.5 * (1 - fx)**2, 0.75 - (fx - 0.5)**2, 0.5 * fx**2]

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.ndim)):
                    weight *= w[offset[d] + 1][d]
                
                target_offset = base + offset
                target_in_box = ti.i32(1)
                for d in ti.static(range(self.ndim)):
                    if target_offset[d] < 0 or target_offset[d] >= self.grid_num:
                        target_in_box = 0
                
                if target_in_box > 0:
                    I = target_offset[0]
                    for d in ti.static(range(1, self.ndim)):
                        I = I * self.grid_num + target_offset[d]
                    weight_out[b, I] += weight
                    weight_prob_out[b, I] += weight * prob[b, i]
    
    @ti.kernel
    def p2g_backward_kernel(
        self,
        pos: ti.types.ndarray(ndim=3),                   # (B, N_points, ndim) # type: ignore
        grad_prob: ti.types.ndarray(ndim=2),             # (B, N_points)       # type: ignore
        grad_weight_prob_out: ti.types.ndarray(ndim=2),  # (B, grid_num^ndim)  # type: ignore
    ):
        B, Np, _ = pos.shape
        ti.loop_config(block_dim=self.BWD_BLOCK_DIM)
        for b, i in ti.ndrange(B, Np):
            Xp = ti.Vector([pos[b, i, d] for d in 
                            ti.static(range(self.ndim))]) / self.dx
            base = ti.cast(Xp, ti.i32)
            fx = Xp - base
            w = [0.5 * (1 - fx)**2, 0.75 - (fx - 0.5)**2, 0.5 * fx**2]

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.ndim)):
                    weight *= w[offset[d] + 1][d]
                
                target_offset = base + offset
                target_in_box = ti.i32(1)
                for d in ti.static(range(self.ndim)):
                    if target_offset[d] < 0 or target_offset[d] >= self.grid_num:
                        target_in_box = 0
                
                if target_in_box > 0:
                    I = target_offset[0]
                    for d in ti.static(range(1, self.ndim)):
                        I = I * self.grid_num + target_offset[d]
                    grad_prob[b, i] += weight * grad_weight_prob_out[b, I]


class P2GImpl(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, pos: Tensor, prob: Tensor, p2g_kernel: P2GKernel):
        """
        - pos: (..., N, 3)
        - prob: (..., N)
        """
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(pos)
        ctx.kernel = p2g_kernel

        N, D = pos.shape[-2:]
        aux_shapes = prob.shape[:-1]

        pos = pos.view(-1, N, D)
        prob = prob.view(-1, N)
        
        out_chann = p2g_kernel.grid_num ** p2g_kernel.ndim
        weight_out = torch.zeros((pos.shape[0], out_chann), 
                                 dtype=pos.dtype, device=pos.device)
        weight_prob_out = torch.zeros_like(weight_out)
        p2g_kernel.p2g_forward_kernel(pos, prob, weight_out, weight_prob_out)
        weight_out = weight_out.view(*aux_shapes, out_chann)
        weight_prob_out = weight_prob_out.view(*aux_shapes, out_chann)
        ctx.mark_non_differentiable(weight_out)
        return weight_out, weight_prob_out
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_weight_out, grad_weight_prob_out: Tensor):
        """
        - grad_weight_out: None
        - grad_weight_prob_out: (..., out_chann)
        """
        if grad_weight_prob_out is not None:
            pos, = ctx.saved_tensors
            p2g_kernel: P2GKernel = ctx.kernel

            N, D = pos.shape[-2:]
            pos: Tensor = pos.view(-1, N, D)
            aux_shapes = grad_weight_prob_out.shape[:-1]
            out_chann = p2g_kernel.grid_num ** p2g_kernel.ndim
            assert out_chann == grad_weight_prob_out.size(-1)

            grad_weight_prob_out = grad_weight_prob_out.contiguous()
            grad_weight_prob_out = grad_weight_prob_out.view(-1, out_chann)

            grad_prob = torch.zeros((pos.shape[0], N),
                                    dtype=pos.dtype, device=pos.device)
            p2g_kernel.p2g_backward_kernel(pos, grad_prob, grad_weight_prob_out)
            grad_prob = grad_prob.view(*aux_shapes, N)
        else:
            grad_prob = None
        return None, grad_prob, None


class P2G(nn.Module):
    def __init__(self, grid_num: int, dx: float, ndim: int):
        super().__init__()
        self.dx = dx
        self.ndim = ndim
        self.grid_num = grid_num
        self.kernel = P2GKernel(grid_num, dx, ndim)
        self.dtype_configed = False
    
    @property
    def out_channels(self):
        return self.grid_num ** self.ndim
    
    def forward(self, pos: Tensor, prob: Tensor, clip_minmax: bool = True) -> Tensor:
        """
        Arguments:
        - pos: (B, ..., Npoint, 3)
        - prob: (B, ..., Npoint)

        Returns:
        - prob: (B, ..., Ncoord)
        """
        if not self.dtype_configed:
            if pos.dtype == torch.float64:
                self.kernel.config_dtype(ti.f64)
            else:
                self.kernel.config_dtype(ti.f32)
            self.dtype_configed = True
        
        if clip_minmax:
            pos = torch.clip(pos, 1e-5, self.dx * self.grid_num - 1e-5)

        weight_out, weight_prob_out = P2GImpl.apply(
            pos.contiguous(), prob.contiguous(), self.kernel)
        prob = weight_prob_out / (weight_out + 1e-7)
        return prob



def test_forward():

    ndim = 2
    device = "cuda:0"
    # device = "cpu"
    # dtype = torch.float32
    dtype = torch.float64

    ti.init(ti.cpu if device.startswith("cpu") else ti.cuda)

    B = 4
    N = 1000
    grid_num = 15
    dx = 0.05

    pos = torch.rand(B, N, ndim, dtype=dtype).to(device) * (grid_num * dx)
    prob = torch.rand(B, N, dtype=dtype).to(device).requires_grad_(True)
    # print((pos / dx).long())

    Xp = pos / dx
    base = Xp.long()
    fx = Xp - base
    print("Xp", Xp)
    print("fx", fx)
    print("prob", prob)

    # weight_out = torch.zeros(B, grid_num**ndim).to(device)
    # weight_prob_out = torch.zeros(B, grid_num**ndim).to(device)

    # p2g = P2GKernel(grid_num, dx)
    # p2g.p2g_forward_kernel(pos, prob, weight_out, weight_prob_out)

    # print(weight_out)
    # print(weight_prob_out)

    p2g = P2G(grid_num, dx, ndim)
    prob_out = p2g(pos, prob)

    print(prob_out)
    prob_out.sum().backward()
    print(prob.grad)

    f_wrap = lambda x: p2g(pos, x)
    out = torch.autograd.gradcheck(f_wrap, inputs=(prob,), eps=1e-5)
    print(out)


def test_speed():
    import time

    ndim = 3
    device = "cuda:0"
    dtype = torch.float32

    ti.init(ti.cpu if device.startswith("cpu") else ti.cuda)

    B = 40 * 256
    N = 256 * 2
    grid_num = 15
    dx = 0.05

    pos = torch.rand(1, B, 1, N, ndim, dtype=dtype).to(device) * (grid_num * dx)
    prob = torch.rand(1, B, 1, N, dtype=dtype).to(device).requires_grad_(True)
    # prob = torch.rand(B, N, dtype=dtype).to(device).requires_grad_(False)

    p2g = P2G(grid_num, dx, ndim)
    prob_out = p2g(pos, prob)

    t0 = time.perf_counter()
    for _ in range(1000):
        prob_out = p2g(pos, prob)
        print(prob_out.sum())
        t1 = time.perf_counter(); print(t1 - t0, flush=True); t0 = t1
    
    prob_out.sum().backward()
    print(prob.grad)
    print(prob_out.shape)



if __name__ == "__main__":
    test_forward()
    # test_speed()

