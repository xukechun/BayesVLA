import torch
from torch import nn, Tensor
from einops import rearrange
import torch.nn.functional as F
from enum import IntFlag, auto
from .pe import RoPE
from .norms import RMSNorm

try:
    import xformers.ops as xops
    xformers_available = True
except Exception as e:
    xformers_available = False

xformers_available = False  # manually disable xformers, use sdpa


class ProjOpt(IntFlag):
    QK = auto()  # use 1 linear layer, to_qk, note v is not projected
    QKV = auto()  # use 1 linear layer: to_qkv
    Q_KV = auto()  # use 2 linear layers: to_q, to_kv
    Q_K_V = auto()  # use 3 linear layers: to_q, to_k, to_v
    DEFAULT = Q_K_V


class MySimpleMHA(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        dropout=0., 
        kdim=None, 
        vdim=None, 
        proj_opt=ProjOpt.DEFAULT, 
        flash=True,
        qk_norm=False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.dropout = dropout
        self.flash = flash
        self.qk_norm = qk_norm

        self.proj_opt = proj_opt
        if proj_opt == ProjOpt.QKV:
            self.to_qkv = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        elif proj_opt == ProjOpt.Q_KV:
            self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.to_kv = nn.Linear(self.kdim, 2*embed_dim, bias=False)
        elif proj_opt == ProjOpt.Q_K_V:
            self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.to_k = nn.Linear(self.kdim, embed_dim, bias=False)
            self.to_v = nn.Linear(self.vdim, embed_dim, bias=False)
        elif proj_opt == ProjOpt.QK:
            self.to_qk = nn.Linear(embed_dim, 2*embed_dim, bias=False)

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)    

        self.to_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        # nn.MultiheadAttention use xavier_uniform initialization
        if self.proj_opt == ProjOpt.QKV:
            nn.init.xavier_uniform_(self.to_qkv.weight)
        elif self.proj_opt == ProjOpt.Q_KV:
            nn.init.xavier_uniform_(self.to_q.weight)
            nn.init.xavier_uniform_(self.to_kv.weight)
        elif self.proj_opt == ProjOpt.Q_K_V:
            nn.init.xavier_uniform_(self.to_q.weight)
            nn.init.xavier_uniform_(self.to_k.weight)
            nn.init.xavier_uniform_(self.to_v.weight)
        elif self.proj_opt == ProjOpt.QK:
            nn.init.xavier_uniform_(self.to_qk.weight)

    @classmethod
    def check_expand_attn_mask(cls, attn_mask: Tensor, B_H_Lx_Lc):
        """
        Arguments:
        - attn_mask: (B, Lc) or (B, Lx, Lc) or (B, H, Lx, Lc)
        - B_H_Lx_Lc: tuple of ints, (B, H, Lx, Lc)

        Returns:
        - attn_mask: (B, H, Lx, Lc)
        """
        if attn_mask is not None:
            B, H, Lx, Lc = B_H_Lx_Lc
            attn_dim = attn_mask.dim()
            assert attn_dim in (2, 3, 4)
            if attn_dim == 2:
                assert attn_mask.shape == (B, Lc), \
                    "attn_mask shape = {}, (B, Lc) = ({}, {})".format(attn_mask.shape, B, Lc)
                attn_mask = attn_mask[:, None, None, :].expand(B, H, Lx, Lc)
            elif attn_dim == 3:
                assert attn_mask.shape == (B, Lx, Lc), \
                    "attn_mask shape = {}, (B, Lx, Lc) = ({}, {}, {})".format(attn_mask.shape, B, Lx, Lc)
                attn_mask = attn_mask[:, None, :, :].expand(B, H, Lx, Lc)
            elif attn_dim == 4:
                assert attn_mask.shape == (B, H, Lx, Lc), \
                    "attn_mask shape = {}, (B, H, Lx, Lc) = ({}, {}, {})".format(attn_mask.shape, B, H, Lx, Lc)
        return attn_mask

    def project_naive(self, x: Tensor, c: Tensor):
        if self.proj_opt == ProjOpt.QKV:
            assert x is c, "id(x) = {:x}, id(c) = {:x}".format(id(x), id(c))
            qkv = rearrange(self.to_qkv(x), "b l (n h c) -> b h n l c", n=3, h=self.num_heads)
            q, k, v = torch.unbind(qkv, dim=2)
        elif self.proj_opt == ProjOpt.Q_KV:
            q = rearrange(self.to_q(x), "b l (h c) -> b h l c", h=self.num_heads)
            kv = rearrange(self.to_kv(c), "b l (n h c) -> b h n l c", n=2, h=self.num_heads)
            k, v = torch.unbind(kv, dim=2)
        elif self.proj_opt == ProjOpt.Q_K_V:
            q = rearrange(self.to_q(x), "b l (h c) -> b h l c", h=self.num_heads)
            k = rearrange(self.to_k(c), "b l (h c) -> b h l c", h=self.num_heads)
            v = rearrange(self.to_v(c), "b l (h c) -> b h l c", h=self.num_heads)
        elif self.proj_opt == ProjOpt.QK:
            qk = rearrange(self.to_qk(x), "b l (n h c) -> b h n l c", n=2, h=self.num_heads)
            q, k = torch.unbind(qk, dim=2)
            v = rearrange(c, "b l (h c) -> b h l c", h=self.num_heads)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        return q, k, v
    
    def project_xformer(self, x: Tensor, c: Tensor):
        if self.proj_opt == ProjOpt.QKV:
            assert x is c, "id(x) = {:x}, id(c) = {:x}".format(id(x), id(c))
            qkv = rearrange(self.to_qkv(x), "b l (n h c) -> b l n h c", n=3, h=self.num_heads)
            q, k, v = torch.unbind(qkv, dim=2)  # each of (B, L, H, C)
        elif self.proj_opt == ProjOpt.Q_KV:
            q = rearrange(self.to_q(x), "b l (h c) -> b l h c", h=self.num_heads)
            kv = rearrange(self.to_kv(c), "b l (n h c) -> b l n h c", n=2, h=self.num_heads)
            k, v = torch.unbind(kv, dim=2)  # each of (B, L, H, C)
        elif self.proj_opt == ProjOpt.Q_K_V:
            q = rearrange(self.to_q(x), "b l (h c) -> b l h c", h=self.num_heads)
            k = rearrange(self.to_k(c), "b l (h c) -> b l h c", h=self.num_heads)
            v = rearrange(self.to_v(c), "b l (h c) -> b l h c", h=self.num_heads)
        elif self.proj_opt == ProjOpt.QK:
            qk = rearrange(self.to_qk(x), "b l (n h c) -> b l n h c", n=2, h=self.num_heads)
            q, k = torch.unbind(qk, dim=2)
            v = rearrange(c, "b l (h c) -> b l h c", h=self.num_heads)
        return q, k, v

    def forward(
        self, 
        x: Tensor, 
        c: Tensor, 
        x_pe: Tensor = None, 
        c_pe: Tensor = None, 
        attn_mask: Tensor = None,
        return_attn_weights = False,
        average_attn_weights = True,
    ):
        """
        Arguments:
        - x: (B, Lx, C)
        - c: (B, Lc, C), context
        - x_pe: (B, Lx, HD, 2)
        - c_pe: (B, Lc, HD, 2)
        - attn_mask: (B, Lc) or (B, Lx, Lc) or (B, NH, Lx, Lc)
            * if boolean: then attn_bias at attn_mask will be -inf
            * if float/double: then attn_bias will be attn_mask
        - return_attn_weights: bool
        - average_attn_weights: bool

        Returns:
        - out: (B, Lx, C)
        - (Optional) attn_weight: (B, Lx, Lc) if average_attn_weights 
                             else (B, H, Lx, Lc)
        """
        use_xformer = self.flash and xformers_available and (not return_attn_weights) and x.is_cuda
        use_sdpa = self.flash and (not return_attn_weights)

        if use_xformer:
            return self.forward_xformer(x, c, x_pe, c_pe, attn_mask)
        elif use_sdpa:
            return self.forward_sdpa(x, c, x_pe, c_pe, attn_mask)
        else:
            return self.forward_naive(x, c, x_pe, c_pe, attn_mask, return_attn_weights, average_attn_weights)
    
    def forward_naive(
        self, 
        x: Tensor, 
        c: Tensor, 
        x_pe: Tensor = None, 
        c_pe: Tensor = None, 
        attn_mask: Tensor = None,
        return_attn_weights = False,
        average_attn_weights = True,
    ):
        q, k, v = self.project_naive(x, c)
        if x_pe is not None: q = RoPE.embed_rotary(q, x_pe)
        if c_pe is not None: k = RoPE.embed_rotary(k, c_pe)

        B, Lx, _ = x.shape
        B, Lc, _ = c.shape
        H = self.num_heads
        attn_mask = self.check_expand_attn_mask(attn_mask, (B, H, Lx, Lc))

        dk = q.shape[-1]
        corr: Tensor = (q @ k.transpose(-1, -2)) / (dk ** 0.5)  # (B, H, Lx, Lc)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                corr.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                corr = corr + attn_mask
        corr = corr - corr.amax(dim=-1, keepdim=True).detach()  # (B, H, Lx, Lc)
        attn_weight = torch.softmax(corr, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, self.training)
        out = attn_weight @ v  # (B, H, Lx, HD)
        out = out.masked_fill(out.isnan(), 0.)  # nan occurs if one row is fully masked out
        out = rearrange(out, "b h l c -> b l (h c)", h=self.num_heads)
        out: Tensor = self.to_out(out)

        if return_attn_weights and average_attn_weights:
            if self.num_heads == 1:
                attn_weight = attn_weight.squeeze(1)
            else:
                attn_weight = attn_weight.mean(1)  # (B, Lx, Lc)
        
        if return_attn_weights:
            return (out, attn_weight), attn_mask
        else:
            return out, attn_mask
    
    def forward_sdpa(
        self, 
        x: Tensor, 
        c: Tensor, 
        x_pe: Tensor = None, 
        c_pe: Tensor = None, 
        attn_mask: Tensor = None
    ):
        q, k, v = self.project_naive(x, c)
        if x_pe is not None: q = RoPE.embed_rotary(q, x_pe)
        if c_pe is not None: k = RoPE.embed_rotary(k, c_pe)

        B, Lx, _ = x.shape
        B, Lc, _ = c.shape
        H = self.num_heads
        attn_mask = self.check_expand_attn_mask(attn_mask, (B, H, Lx, Lc))
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = attn_mask,
            dropout_p = self.dropout if self.training else 0., 
            is_causal = False
        )
        out = out.masked_fill(out.isnan(), 0.)  # nan occurs if one row is fully masked out
        out = rearrange(out, "b h l c -> b l (h c)", h=self.num_heads)
        out: Tensor = self.to_out(out)
        return out, attn_mask
    
    def forward_xformer(
        self,
        x: Tensor,
        c: Tensor,
        x_pe: Tensor = None, 
        c_pe: Tensor = None, 
        attn_mask: Tensor = None
    ):
        q, k, v = self.project_xformer(x, c)
        if x_pe is not None: q = RoPE.embed_rotary_xformer(q, x_pe)
        if c_pe is not None: k = RoPE.embed_rotary_xformer(k, c_pe)

        B, Lx, _ = x.shape
        B, Lc, _ = c.shape
        H = self.num_heads
        
        if attn_mask is None:
            attn_bias = None
        else:
            attn_mask = self.check_expand_attn_mask(attn_mask, (B, H, Lx, Lc))
            
            if attn_mask.dtype == torch.bool:
                # attn_bias = q.new_zeros((B, H, Lx, Lc)).masked_fill_(~attn_mask, float("-inf"))
                attn_bias = attn_mask.type_as(q).log_()
            else:
                attn_bias = attn_mask
            
            attn_stride = attn_bias.stride()[-2]
            # s = 8 if (attn_bias.dtype == torch.bfloat16) else 4
            s = 16 // attn_bias.dtype.itemsize
            if attn_stride % s != 0:
                padding = s - (attn_stride % s)
                attn_bias = F.pad(attn_bias, (0, padding), "constant", float("-inf")).contiguous()
                attn_bias = attn_bias[:, :, :, :Lc]
        
        out = xops.memory_efficient_attention(
            q, k, v, 
            attn_bias = attn_bias, 
            p = self.dropout if self.training else 0., 
        )  # (B, L, H, C)
        out = out.masked_fill(out.isnan(), 0.)  # nan occurs if one row is fully masked out
        out = rearrange(out, "b l h c -> b l (h c)")
        out: Tensor = self.to_out(out)
        return out, attn_bias


def test_mask():
    pos_dim = 3
    embed_dim = 192
    num_attn_heads = 4
    rope = RoPE(embed_dim, pos_dim, num_attn_heads).cuda()
    # attn = MySimpleMHA(embed_dim, num_attn_heads, flash=False)
    attn = MySimpleMHA(embed_dim, num_attn_heads, flash=True, proj_opt=ProjOpt.Q_KV).cuda()

    x = torch.randn(1, 6, embed_dim).cuda()
    y = torch.randn(1, 4, embed_dim).cuda()

    x_pos = torch.randn(1, 6, pos_dim).cuda()
    y_pos = torch.randn(1, 4, pos_dim).cuda()

    attn_mask = torch.ones(1, 6, 4).bool().cuda()
    attn_mask[0, -1, :] = False

    x_pe0, y_pe0 = rope(x_pos), rope(y_pos)
    out0 = attn(x, y, x_pe=x_pe0, c_pe=y_pe0, attn_mask=attn_mask)

    attn.flash = False
    out1 = attn(x, y, x_pe=x_pe0, c_pe=y_pe0, attn_mask=attn_mask)
    print(out0[0, :, :4])
    print(out1[0, :, :4])


def chunk_test():
    n, h, l, c = 3, 4, 10, 1
    
    x = torch.randn(l, n*h*c)
    qkv0 = rearrange(x, "l (n h c) -> l n h c", n=n, h=h)
    q0, k0, v0 = qkv0.unbind(1)  # (L, H, C)
    
    qkv1 = rearrange(x, "l (h c) -> h l c", h=h)
    q1, k1, v1 = qkv1.chunk(3, dim=-1)  # (H, L, C)
    
    # print((q0 == q1.transpose(0, 1)).all())
    print(q0.squeeze())
    print(q1.transpose(0, 1).squeeze())


if __name__ == "__main__":
    # test_mha()
    test_mask()
    
    # chunk_test()

