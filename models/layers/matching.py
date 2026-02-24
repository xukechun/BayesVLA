import torch
from torch import nn, Tensor
from einops import rearrange
from torch_geometric.utils import softmax
from torch_geometric.nn import radius, SumAggregation
from .p2g_irreg import P2G


class CoarseMatching(nn.Module):
    def __init__(self, grid_nums=[12], voxel_sizes=[0.12], ndim=3):
        super().__init__()
        self.offsets = [grid_num * voxel_size / 2.0 for grid_num, voxel_size 
                        in zip(grid_nums, voxel_sizes)]
        self.p2gs = [P2G(grid_num, voxel_size, ndim=ndim) for grid_num, voxel_size
                     in  zip(grid_nums, voxel_sizes)]

    @property
    def out_channels(self):
        return tuple([p2g.out_channels for p2g in self.p2gs])

    def match_distribution(self, prob: Tensor, pos: Tensor):
        """
        Arguments:
        - prob: (B, L0, L1)
        - pos: (B, L0, L1, 3)

        Returns:
        - probs: [(B, L0, N_coord)]
        """
        B, L0, L1 = prob.shape
        batch = torch.arange(B*L0, device=prob.device).repeat_interleave(L1)

        probs = []
        for p2g, offset in zip(self.p2gs, self.offsets):
            pos_ = pos.view(B*L0*L1, 3) + offset
            prob_ = prob.view(B*L0*L1)
            prob_out = p2g(pos_, prob_, batch, B*L0)
            probs.append(rearrange(prob_out, "(b l) c -> b l c", b=B, l=L0))
        return probs

    def forward(
        self, 
        x0: Tensor, x1: Tensor, 
        pos0: Tensor, pos1: Tensor, 
        global_match_score: Tensor = None,
    ):
        """Find correspondence for x0 in x1

        Arguments: 
        - x0: (B, L0, C)
        - x1: (B, L1, C)
        - pos0: (B, L0, 3)
        - pos1: (B, L1, 3)

        Returns:
        - flow: (B, L0, 3)
        - flow_distributions: [(B, L0, N_coord)]
        """
        corr = torch.bmm(x0, x1.transpose(-1, -2)) / float(x0.shape[-1]**0.5)
        
        score_0_match_1 = torch.softmax(corr - corr.amax(dim=-1, keepdim=True).detach(), dim=-1)
        score_1_match_0 = torch.softmax(corr - corr.amax(dim=-2, keepdim=True))
        
        correlation = correlation - torch.amax(correlation.detach(), dim=-1, keepdim=True)
        prob = torch.softmax(correlation, dim=-1)  # (B, L0, L1)
        if global_match_score is not None:
            prob = prob * global_match_score
            prob = prob / (prob.sum(dim=-1, keepdim=True) + 1e-8)
        correspondence = torch.bmm(prob, pos1)  # (B, L0, 3)
        flow = correspondence - pos0  # (B, L0, 3)

        delta_pos = pos1[:, None, :, :] - pos0[:, :, None, :]  # (B, L0, L1, 3)
        flow_distributions = self.match_distribution(prob, delta_pos)  # [(B, L0, N_coord)]
        return flow, flow_distributions


class FineMatching(nn.Module):
    def __init__(self, grid_num=8, voxel_size=0.04, ndim=3, max_neighbors=128):
        super().__init__()
        self.aggr = SumAggregation()
        self.p2g = P2G(grid_num, voxel_size, ndim=ndim)
        self.offset = grid_num * voxel_size / 2.0
        self.radius = (grid_num + 1) * voxel_size / 2.0
        self.max_neighbors = max_neighbors
    
    @property
    def out_channel(self):
        return self.p2g.out_channels
    
    def forward(
        self, 
        x0: Tensor, x1: Tensor, 
        pos0: Tensor, pos1: Tensor, 
        pad_mask: Tensor = None
    ):
        """
        Arguments: 
        - x0: (B, L0, C)
        - x1: (B, L1, C)
        - pos0: (B, L0, 3)
        - pos1: (B, L1, 3)
        - pad_mask: (B, L1)

        Returns:
        - flow: (B, T0, L0, 3)
        - flow_distribution: (B, T0, L0, N_coord)
        """
        B, L0, C = x0.shape
        B, L1, C = x1.shape

        x0 = rearrange(x0, "b l c -> (b t l) c")
        x1 = rearrange(x1, "b l c -> (b t l) c")
        pos0 = rearrange(pos0, "b l c -> (b l) c")
        pos1 = rearrange(pos1, "b l c -> (b l) c")

        batch0 = torch.arange(B, device=x0.device).repeat_interleave(L0)
        batch1 = torch.arange(B, device=x1.device).repeat_interleave(L1)

        edge_index = radius(pos1, pos0, self.radius, batch1, batch0,
                            max_num_neighbors=self.max_neighbors).flip(0)
        e1, e0 = edge_index.unbind(0)
        # e0 ranges from [0, B*L0 - 1]
        # e1 ranges from [0, B*L1 - 1]

        # x0[e0] -> (E, C); x1[e1] -> (E, C); max(E) = B*L0*self.max_neighbors
        correlation = (x0[e0] * x1[e1]).sum(dim=-1) / (C**0.5)
        if pad_mask is not None:
            pad_mask = pad_mask.flatten()  # (B, L1) -> (B*L1,)
            correlation = torch.masked_fill(
                correlation,  # (E,)
                pad_mask[e1],  # (E,)
                float('-inf')
            )

        prob = softmax(correlation, e0, num_nodes=B*L0)  # (E,)
        prob[prob.isnan()] = 0.0   # some src nodes cannot find neighbors, resulting nan
        dpos = pos1[e1] - pos0[e0]  # (E, 3)
        flow_distribution = self.p2g(
            pos=dpos+self.offset, 
            prob=prob, 
            batch_indices=e0, 
            batch_size=B*L0
        )  # (B*L0, num_grid**3)
        flow_distribution = rearrange(flow_distribution, "(b l) c -> b l c", 
                                      b=B, l=L0)
        flow = prob.unsqueeze(-1) * dpos  # (E, 3)
        flow = self.aggr(flow, e0, dim_size=B*L0)  # (B*L0, 3)
        flow = rearrange(flow, "(b l) c -> b l c", b=B, l=L0)

        return flow, flow_distribution

