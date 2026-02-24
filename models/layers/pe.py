import math
import torch
from torch import nn, Tensor


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, temperature=1e4):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x: Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.temperature) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RoPE(nn.Module):
    def __init__(self, feature_dim, position_dim, num_heads=1, temperature=1e4):
        super().__init__()
        assert feature_dim % (position_dim * num_heads) == 0
        self.head_dim = feature_dim // num_heads
        self.feature_dim = feature_dim
        self.position_dim = position_dim
        self.num_heads = num_heads
        self.temperature = temperature
    
    @staticmethod
    def embed_rotary(x: Tensor, pe: Tensor):
        """
        - x: (batch_size, num_heads, length, head_dim)
        - pe: (batch_size, length, head_dim, 2)
        """
        assert x.shape[-2:] == pe.shape[-3:-1]
        if x.dim() == 4: pe = pe.unsqueeze(1)  # (batch_size, 1, length, head_dim, 2)
        cos, sin = pe.type_as(x).unbind(-1)
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x = x * cos + x2 * sin
        return x
    
    @staticmethod
    def embed_rotary_xformer(x: Tensor, pe: Tensor):
        """
        - x: (batch_size, length, num_heads, head_dim)
        - pe: (batch_size, length, head_dim, 2)
        """
        assert x.shape[1] == pe.shape[1]
        assert x.shape[-1] == pe.shape[-2]
        if x.dim() == 4: pe = pe.unsqueeze(2)  # (batch_size, length, 1, head_dim, 2)
        cos, sin = pe.type_as(x).unbind(-1)  # each of (B, L, 1, H)
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x = x * cos + x2 * sin
        return x

    @torch.no_grad()
    def forward(self, x: Tensor):
        """
        Arguments:
        - x: (..., position_dim)

        Returns:
        - pe: (..., head_dim, 2)
        """
        # B, L, PD = x.size()
        PD = x.shape[-1]
        assert PD == self.position_dim

        HD = self.feature_dim // self.num_heads
        div_term = torch.exp(
            torch.arange(0, HD // PD, 2, dtype=torch.float, device=x.device)
            * (-math.log(self.temperature) / (HD // PD))
        )  # (hd//pd/2,)

        x = x.unsqueeze(-1)  # (..., pd, 1)
        sinx = torch.sin(x * div_term)  # (..., pd, hd//pd/2)
        cosx = torch.cos(x * div_term)  # (..., pd, hd//pd/2)

        sinx = sinx.repeat_interleave(2, dim=-1)  # (..., pd, hd//pd)
        cosx = cosx.repeat_interleave(2, dim=-1)  # (..., pd, hd//pd)

        sinx = sinx.flatten(-2)  # (..., pd, hd//pd) -> (..., hd)
        cosx = cosx.flatten(-2)  # (..., pd, hd//pd) -> (..., hd)
        position_code = torch.stack([cosx, sinx], dim=-1)  # (..., hd, 2)

        if position_code.requires_grad:
            position_code = position_code.detach()
        return position_code


def test_corr():
    rope = RoPE(192, 3)
    to_q = nn.Linear(192, 192, bias=False)
    to_k = nn.Linear(192, 192, bias=False)

    x = torch.randn(1, 6, 192)
    y = torch.randn(1, 4, 192)

    x_pos = torch.randn(1, 6, 3)
    y_pos = torch.randn(1, 4, 3)
    offset = torch.randn(3)

    x_pe0, y_pe0 = rope(x_pos), rope(y_pos)
    x_pe1, y_pe1 = rope(x_pos + offset), rope(y_pos + offset)

    q0, k0 = to_q(x), to_k(y)
    q0 = rope.embed_rotary(q0, *x_pe0.unbind(-1))
    k0 = rope.embed_rotary(k0, *y_pe0.unbind(-1))
    corr0 = torch.bmm(q0, k0.transpose(-1, -2))

    q1, k1 = to_q(x), to_k(y)
    q1 = rope.embed_rotary(q1, *x_pe1.unbind(-1))
    k1 = rope.embed_rotary(k1, *y_pe1.unbind(-1))
    corr1 = torch.bmm(q1, k1.transpose(-1, -2))

    print(corr0)
    print(corr1)

    
    


if __name__ == "__main__":
    # test_equal()
    test_corr()

