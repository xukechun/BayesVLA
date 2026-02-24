import torch
from typing import List, Dict
from torch import nn, Tensor
from einops import rearrange, repeat
from .mha import MySimpleMHA, ProjOpt
from timm.models.vision_transformer import RmsNorm


class AdaLN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x: Tensor, t: Tensor):
        """
        Arguments:
        - x: (B, L, C)
        - t: (B, C) or (B, L, C)
        """
        assert x.dim() == 3
        assert t.dim() in (2, 3)
        if t.dim() == 2:
            t = t.unsqueeze(1)
        assert t.shape[1] == 1 or t.shape[1] == x.shape[1]
        scale, shift = torch.chunk(self.modulation(t), 2, dim=-1)
        x = x * (1 + scale) + shift
        return x


class AdaLN_Zero(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x: Tensor, t: Tensor):
        """
        Arguments:
        - x: (B, L, C)
        - t: (B, C) or (B, L, C)
        """
        assert x.dim() == 3
        assert t.dim() in (2, 3)
        if t.dim() == 2:
            t = t.unsqueeze(1)
        assert t.shape[1] == 1 or t.shape[1] == x.shape[1]
        scale, shift, gate = torch.chunk(self.modulation(t), 3, dim=-1)
        x = x * (1 + scale) + shift
        return x, gate


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, use_adaln=False):
        super().__init__()
        self.norm = RmsNorm(embed_dim, affine=not use_adaln)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln = AdaLN_Zero(embed_dim)

    def forward(self, x: Tensor, film: Tensor = None):
        y = self.norm(x)
        if self.use_adaln:
            y, gate = self.adaln(y, film)

        y = self.ff(y)
        if self.use_adaln:
            y = y * gate
        return x + y


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_adaln=False, 
                 proj_opt=ProjOpt.Q_KV):
        super().__init__()
        self.norm = RmsNorm(embed_dim, affine=not use_adaln)
        self.attn = MySimpleMHA(
            embed_dim, num_heads, dropout=dropout, proj_opt=proj_opt
        )
        self.dropout = nn.Dropout(dropout)

        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln = AdaLN_Zero(embed_dim)

    def forward(
        self, 
        query, 
        value, 
        query_pe=None, 
        value_pe=None, 
        value_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        x = self.norm(query)
        if self.use_adaln:
            x, gate = self.adaln(x, film)

        attn_output, value_mask = self.attn(
            x=x,
            c=value,
            x_pe=query_pe,
            c_pe=value_pe,
            attn_mask=value_mask,
            return_attn_weights=return_attn_weights,
            average_attn_weights=average_attn_weights
        )
        if return_attn_weights:
            attn_output, attn_weight = attn_output

        attn_output = self.dropout(attn_output)  # drop projection
        if self.use_adaln:
            attn_output = attn_output * gate
        output = query + attn_output
        return ((output, attn_weight) if return_attn_weights else output), value_mask


class SelfAttentionLayer(CrossAttentionLayer):
    def __init__(self, embed_dim, num_heads, dropout=0, use_adaln=False, 
                 proj_opt=ProjOpt.QKV):
        super().__init__(embed_dim, num_heads, dropout, use_adaln, proj_opt)

    def forward(
        self, 
        query, 
        query_pe=None, 
        query_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        x = self.norm(query)
        if self.use_adaln:
            x, gate = self.adaln(x, film)
        
        attn_output, query_mask = self.attn(
            x=x,
            c=x,
            x_pe=query_pe,
            c_pe=query_pe,
            attn_mask=query_mask,
            return_attn_weights=return_attn_weights,
            average_attn_weights=average_attn_weights
        )
        if return_attn_weights:
            attn_output, attn_weight = attn_output
        
        attn_output = self.dropout(attn_output)  # drop projection
        if self.use_adaln:
            attn_output = attn_output * gate
        output = query + attn_output
        return ((output, attn_weight) if return_attn_weights else output), query_mask


class FinalNorm(nn.Module):
    def __init__(self, embed_dim, use_adaln=False):
        super().__init__()
        self.norm = RmsNorm(embed_dim, affine=not use_adaln)
        self.use_adaln = use_adaln

        if use_adaln:
            self.adaln = AdaLN(embed_dim)
    
    def forward(self, x: Tensor, film: Tensor = None):
        y = self.norm(x)
        if self.use_adaln:
            y = self.adaln(x, film)
        return y


class FFWCrossAttentionLayers(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.0, use_adaln=False,
                 proj_opt=ProjOpt.Q_KV):
        super().__init__()
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_layers.append(
                CrossAttentionLayer(embed_dim, num_heads, dropout, use_adaln, proj_opt))
            self.ffn_layers.append(
                FFN(embed_dim, 2*embed_dim, dropout, use_adaln))
    
    def forward(
        self, 
        query, 
        value, 
        query_pe=None, 
        value_pe=None, 
        value_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        """
        - query, value: (B, Lq or Lv, C)
        - query_pe, value_pe: (B, Lq or Lv, C, 2)
        - value_mask: (B, Lv)
        - film: (B, C)
        """
        outputs = []
        for i in range(self.num_layers):
            query, value_mask = self.attn_layers[i](
                query, value, query_pe, value_pe, value_mask, film,
                return_attn_weights, average_attn_weights)
            if return_attn_weights:
                query, attn_weight = query
            query = self.ffn_layers[i](query, film)
            outputs.append((query, attn_weight) if return_attn_weights else query)
        return outputs


class FFWSelfAttentionLayers(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0, use_adaln=False, 
                 proj_opt=ProjOpt.QKV):
        super().__init__()
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_layers.append(
                SelfAttentionLayer(embed_dim, num_heads, dropout, use_adaln, proj_opt))
            self.ffn_layers.append(
                FFN(embed_dim, 2*embed_dim, dropout, use_adaln))

    def forward(
        self, 
        query, 
        query_pe=None, 
        query_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        """
        - query, value: (B, Lq or Lv, C)
        - query_pe, value_pe: (B, Lq or Lv, C, 2)
        - value_mask: (B, Lv)
        - film: (B, C)
        """
        outputs = []
        for i in range(self.num_layers):
            query, query_mask = self.attn_layers[i](
                query, query_pe, query_mask, film,
                return_attn_weights, average_attn_weights)
            if return_attn_weights:
                query, attn_weight = query
            query = self.ffn_layers[i](query, film)
            outputs.append((query, attn_weight) if return_attn_weights else query)
        return outputs


class FFWXAttentionLayers(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_types: str, dropout=0.0, use_adaln=False):
        """
        - layer_types: 'scfscf' means ['self-attn', 'cross-attn', 'ffn'] x 2 on single side
        """
        super().__init__()
        self.layer_types = layer_types.lower().strip()
        self.layers = nn.ModuleList()

        for char in layer_types:
            if char == "s":
                self.layers.append(
                    SelfAttentionLayer(embed_dim, num_heads, dropout, use_adaln))
            elif char == "c":
                self.layers.append(
                    CrossAttentionLayer(embed_dim, num_heads, dropout, use_adaln))
            elif char == "f":
                self.layers.append(
                    FFN(embed_dim, 2*embed_dim, dropout, use_adaln))
            else:
                raise ValueError("{} not recognized, available char are ['c', 's', 'f']"
                                 .format(char))

    def forward(
        self, 
        query, 
        value, 
        query_pe=None, 
        value_pe=None, 
        query_mask=None, 
        value_mask=None, 
        film=None,
    ):
        for char, layer in zip(self.layer_types, self.layers):
            if char == "s":  # self attention
                query, query_mask = layer(query, query_pe, query_mask, film)
            elif char == "c":  # cross attention
                query, value_mask = layer(query, value, query_pe, value_pe, value_mask, film)
            elif char == "f":
                query = layer(query, film)
        return query


class ParallelAttentionLayers(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_types: str, dropout=0.0, use_adaln=False):
        """
        - layer_types: 'scfscf' means ['self-attn', 'cross-attn', 'ffn'] x 2 on both sides
        """
        super().__init__()
        self.layer_types = layer_types.lower().strip()
        self.layers = nn.ModuleList()

        for char in layer_types:
            if char == "s":
                self.layers.append(
                    SelfAttentionLayer(embed_dim, num_heads, dropout, use_adaln))
            elif char == "c":
                self.layers.append(
                    CrossAttentionLayer(embed_dim, num_heads, dropout, use_adaln))
            elif char == "f":
                self.layers.append(
                    FFN(embed_dim, 2*embed_dim, dropout, use_adaln))
            else:
                raise ValueError("{} not recognized, available char are ['c', 's', 'f']"
                                 .format(char))

    @classmethod
    def self_attn_forward(
        cls, 
        attn: SelfAttentionLayer, 
        x: Tensor, pe, mask, film,
        return_attn_weights, average_attn_weights,
        mask_cached = None
    ):
        T = x.shape[1]
        x = rearrange(x, "b t l c -> (b t) l c", t=T)
        if pe is not None:
            pe = rearrange(pe, "b t l c d -> (b t) l c d", t=T)
        if mask_cached is not None:
            mask = mask_cached
        elif mask is not None:
            # (B, T, L) or (B, T, L, L) or (B, T, H, L, L)
            mask = rearrange(mask, "b t ... -> (b t) ...", t=T)
        if film is not None:
            film = repeat(film, "b c -> (b t) c", t=T)
        
        x, mask_cached = attn(query=x, query_pe=pe, query_mask=mask, film=film,
                              return_attn_weights=return_attn_weights, 
                              average_attn_weights=average_attn_weights)
        if return_attn_weights:
            x, a = x
            a = rearrange(a, "(b t) ... l1 l2 -> b t ... l1 l2", t=T)
        x = rearrange(x, "(b t) l c -> b t l c", t=T)
        return ((x, a) if return_attn_weights else x), mask_cached

    @classmethod
    def cross_attn_forward(
        cls, 
        attn: CrossAttentionLayer, 
        x: Tensor, c: Tensor, x_pe, c_pe, mask, film,
        return_attn_weights, average_attn_weights,
        mask_cached = None
    ):
        Tx = x.shape[1]
        Tc = c.shape[1]
        x = rearrange(x, "b t l c -> b (t l) c", t=Tx)
        c = rearrange(c, "b t l c -> b (t l) c", t=Tc)
        if x_pe is not None:
            x_pe = rearrange(x_pe, "b t l c d -> b (t l) c d", t=Tx)
        if c_pe is not None:
            c_pe = rearrange(c_pe, "b t l c d -> b (t l) c d", t=Tc)
        if mask_cached is not None:
            mask = mask_cached
        
        x, mask_cached = attn(query=x, value=c, 
                              query_pe=x_pe, value_pe=c_pe, 
                              value_mask=mask, film=film,
                              return_attn_weights=return_attn_weights, 
                              average_attn_weights=average_attn_weights)
        if return_attn_weights:
            x, a = x
            a = rearrange(a, "b ... (tx lx) (tc lc) -> b ... tx lx tc lc", tx=Tx, tc=Tc)
        x = rearrange(x, "b (t l) c -> b t l c", t=Tx)
        return ((x, a) if return_attn_weights else x), mask_cached

    @classmethod
    def ffn_forward(cls, ffn: FFN, x: Tensor, film: Tensor):
        T = x.shape[1]
        x = rearrange(x, "b t l c -> b (t l) c", t=T)
        x = ffn(x, film)
        x = rearrange(x, "b (t l) c -> b t l c", t=T)
        return x

    def forward(
        self, 
        query: Tensor, 
        value: Tensor,
        query_pe: Tensor = None, 
        value_pe: Tensor = None, 
        qq_mask: Tensor = None, 
        vv_mask: Tensor = None, 
        qv_mask: Tensor = None, 
        vq_mask: Tensor = None, 
        film: Tensor = None, 
        return_attn_weights=False,
        average_attn_weights=True
    ):
        """
        Arguments:
        - query: (B, Tq, Lq, C)
        - value: (B, Tv, Lv, C)
        - query_pe: (B, Tq, Lq, HD, 2)
        - value_pe: (B, Tv, Lv, HD, 2)
        - qq_mask: (B, Tq, Lq) or (B, Tq, Lq, Lq) or (B, Tq, NH, Lq, Lq)
        - vv_mask: (B, Tv, Lv) or (B, Tv, Lv, Lv) or (B, Tv, NH, Lv, Lv)
        - qv_mask: (B, Tv*Lv) or (B, Tq*Lq, Tv*Lv) or (B, NH, Tq*Lq, Tv*Lv)
        - vq_mask: (B, Tq*Lq) or (B, Tv*Lv, Tq*Lq) or (B, NH, Tv*Lv, Tq*Lq)
        - film: (B, C)

        Returns:
        - query: (B, Tq, Lq, C)
        - value: (B, Tv, Lv, C)
        - (Optional) attns: stores qq/vv/qv/vq attentions of all self/cross attention layers
            * qq: (B, Tq, (H,) Lq, Lq)
            * vv: (B, Tv, (H,) Lv, Lv)
            * qv: (B, (H,) Tq, Lq, Tv, Lv)
            * vq: (B, (H,) Tv, Lv, Tq, Lq)
            Note: if `average_attn_weights` is True, then `(H,)` is omitted
        """
        unsqueeze_q = query.dim() == 3
        unsqueeze_v = value.dim() == 3

        if unsqueeze_q:
            query = query.unsqueeze(1)
            if query_pe is not None:
                query_pe = query_pe.unsqueeze(1)
            if qq_mask is not None:
                qq_mask = qq_mask.unsqueeze(1)

        if unsqueeze_v:
            value = value.unsqueeze(1)
            if value_pe is not None:
                value_pe = value_pe.unsqueeze(1)
            if vv_mask is not None:
                vv_mask = vv_mask.unsqueeze(1)

        if return_attn_weights:
            attns: Dict[str, List[Tensor]] = {
                "qq": [], "vv": [], "qv": [], "vq": []
            }
        
        qq_mask_cached = None
        vv_mask_cached = None
        qv_mask_cached = None
        vq_mask_cached = None

        for char, layer in zip(self.layer_types, self.layers):
            if char == "s":
                # print("self-attention called")
                query, qq_mask_cached = self.self_attn_forward(
                    layer, query, query_pe, qq_mask, film,
                    return_attn_weights, average_attn_weights, 
                    qq_mask_cached
                )
                value, vv_mask_cached = self.self_attn_forward(
                    layer, value, value_pe, vv_mask, film,
                    return_attn_weights, average_attn_weights, 
                    vv_mask_cached
                )
                if return_attn_weights:
                    query, qq_attn = query; attns["qq"].append(qq_attn)
                    value, vv_attn = value; attns["vv"].append(vv_attn)
            
            elif char == "c":
                # print("cross-attention called")
                query_new, qv_mask_cached = self.cross_attn_forward(
                    layer, query, value, query_pe, value_pe, qv_mask, film,
                    return_attn_weights, average_attn_weights, 
                    qv_mask_cached
                )
                value_new, vq_mask_cached = self.cross_attn_forward(
                    layer, value, query, value_pe, query_pe, vq_mask, film,
                    return_attn_weights, average_attn_weights, 
                    vq_mask_cached
                )
                query = query_new
                value = value_new

                if return_attn_weights:
                    query, qv_attn = query; attns["qv"].append(qv_attn)
                    value, vq_attn = value; attns["vq"].append(vq_attn)
            
            elif char == "f":
                # print("ffn called")
                query = self.ffn_forward(layer, query, film)
                value = self.ffn_forward(layer, value, film)
            
            # print("layer {}, nan in query: {}, nan in value: {}"
            #       .format(char, query.isnan().any(), value.isnan().any()))
        
        if unsqueeze_q:
            query = query.squeeze(1)
            if return_attn_weights:
                for qq_attn in attns["qq"]:
                    qq_attn.squeeze_(1)  # (B, T, (H,) L, L) -> (B, (H), L, L)
                for qv_attn in attns["qv"]:
                    qv_attn.squeeze_(-4)  # (B, (H), Tq, Lq, Tv, Lv) -> (B, (H), Lq, Tv, Lv)
        if unsqueeze_v:
            value = value.squeeze(1)
            if return_attn_weights:
                for vv_attn in attns["vv"]:
                    vv_attn.squeeze_(1)  # (B, T, (H,) L, L) -> (B, (H,) L, L)
                for vq_attn in attns["vq"]:
                    vq_attn.squeeze_(-2)  # (B, (H), (Tq,) Lq, Tv, Lv) -> (B, (H,) (Tq,) Lq, Lv)
        return (query, value, attns) if return_attn_weights else (query, value)


if __name__ == "__main__":
    from .pe import RoPE

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    B, C = 4, 192

    To, Lo = 3, 255
    Td, Ld = 7, 257

    rope = RoPE(C, 3, 4)
    model = ParallelAttentionLayers(C, 4, "csf" * 2, use_adaln=True)

    x = torch.randn(B, To, Lo, C)
    y = torch.randn(B, Td, Ld, C)
    film = torch.randn(B, C)
    
    x_pos = torch.randn(B, To, Lo, 3)
    y_pos = torch.randn(B, Td, Ld, 3)

    x_pe = rope(x_pos)
    y_pe = rope(y_pos)
    print(x_pe.shape, y_pe.shape)

    xx_mask = torch.ones(B, To, Lo).bool()
    yy_mask = torch.ones(B, Td, Ld).bool()
    xy_mask = torch.zeros(B, To*Lo, Td*Ld).float()
    yx_mask = torch.zeros(B, Td*Ld, To*Lo).float()

    x, y = model(
        query=x,
        value=y,
        query_pe=x_pe,
        value_pe=y_pe,
        qq_mask=xx_mask,
        vv_mask=yy_mask,
        qv_mask=xy_mask,
        vq_mask=yx_mask,
        film=film
    )

    print(x)
    print(y)
    print(x.shape, y.shape)



