import torch
import torch.nn.functional as F

from torch import nn, Tensor
from einops import rearrange
from typing import Optional, Tuple, Dict

from .layers.utils import simple_mlp
from .layers.pe import SinusoidalPosEmb, RoPE
from .layers.attn_dn import FFWXAttentionLayers, NormOrAdaLN, init_xncoder
from .layers.rot_transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


class ActionProb(nn.Module):
    def __init__(
        self, 
        hdim: int, 
        num_heads: int, 
        num_layers: int,
        mlp_layers: int = 2,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()   
        self.act_enc = simple_mlp([self.act_dim, hdim, hdim], activation="gelu")
        self.rope = RoPE(hdim, 2)

        # Note that only cross-attention is used because the actions are regarded as independent from one another
        self.vla_attn = FFWXAttentionLayers(
            hdim, num_heads, layer_types="cf"*num_layers, use_adaln=False, dropout=attn_dropout
        )
        
        self.final_norm = NormOrAdaLN(hdim)
        # self.act_logits_head = simple_mlp([hdim, hdim, 1], activation="gelu")

        # modify act_logits_head as multi-layer mlp
        mlp_neurons = [hdim] * mlp_layers + [1]
        self.act_logits_head = simple_mlp(mlp_neurons, activation="gelu", dropout=mlp_dropout)

        init_xncoder(num_layers*2, self.vla_attn)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if m.bias is not None and m.bias.requires_grad:
                    # Do not modify the bias in DINOv2!!!
                    nn.init.zeros_(m.bias)

    @property
    def act_dim(self):
        """dimension of action defined in camera frame"""
        return 9

    def forward(
        self, 
        obs_extrinsics: Tensor, 
        obs_intrinsics: Tensor,

        #### from vlm ####
        prompt_film: Optional[Tensor], 
        prompt_token_img: Optional[Tensor], 
        prompt_token_lang: Optional[Tensor], 
        prompt_mask_img: Optional[Tensor], 
        prompt_mask_lang: Optional[Tensor], 
        ##################

        grasp_poses: Tensor,
        grasp_masks: Optional[Tensor],
        gt_select_grasp: Tensor,

        inference: bool,
        fp16: bool,
    ):
        """
        Args:
            obs_extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            obs_intrinsics: (B, ncam, 3, 3), ^{world}_{camera} K

            prompt_film: (B, hdim), output of vlm model
            prompt_token_img: (B, Nprompt, hdim), output of vlm model
            prompt_token_lang: (B, Nprompt, hdim), output of vlm model
            prompt_mask_img: (B, Nprompt), output of vlm model
            prompt_mask_lang: (B, Nprompt), output of vlm model

            grasp_poses: (B, Ngrasp, 4, 4), ^{world}_{ee} T
            grasp_masks: (B, Ngrasp), whether the grasp pose is valid
            gt_select_grasp: (B,), index of the selected grasp

            inference: if True, do not compute loss
            fp16: if True, use bfloat16
        
        Returns
        -------
        (if inference is True)
            select_grasp_pose (Tensor): (B, 4, 4), ^{world}_{ee} T
        (else)
            loss (Tensor): scalar tensor
            metrics (Dict[str, Tensor]): metrics for logging
        """
        latest_cam_poses = obs_extrinsics[:, -1]  # (B, Ncam, 4, 4)
        current_cam_pose = latest_cam_poses[:, 0]

        grasp_poses_cam = ee2cam(current_cam_pose, grasp_poses)  # (B, Ngrasp, 9)
        
        # project grasp positions to pixel coordinates
        grasp_pos = grasp_poses_cam[..., :3] # (B, Ngrasp, 3)
        cx = obs_intrinsics[:, 0, 0, 2]
        cy = obs_intrinsics[:, 0, 1, 2]
        fx = obs_intrinsics[:, 0, 0, 0]
        fy = obs_intrinsics[:, 0, 1, 1]
        grasp_u = fx[:, None] * grasp_pos[..., 0] / grasp_pos[..., 2] + cx[:, None]
        grasp_v = fy[:, None] * grasp_pos[..., 1] / grasp_pos[..., 2] + cy[:, None]
        grasp_uv = torch.stack([grasp_u, grasp_v], dim=-1)  # (B, Ngrasp, 2)

        cxy = torch.stack([cx, cy], axis=-1).to(grasp_uv.device)  # (B, 2)
        fxy = torch.stack([fx, fy], axis=-1).to(grasp_uv.device)  # (B, 2)
        grasp_norm_uv = (grasp_uv - cxy[:, None]) / fxy[:, None]  # (B, Ngrasp, 2)

        grasp_feats = self.act_enc(grasp_poses_cam) # (B, Ngrasp, hdim)
        grasp_pe = self.rope(grasp_norm_uv)
        grasp_feats = self.rope.embed_rotary(grasp_feats, grasp_pe)

        if prompt_film is not None:
            film = prompt_film
        else:
            film = None

        if prompt_mask_lang is None:
            prompt_mask_lang = torch.ones_like(prompt_token_lang[..., 0]) # (B, Nprompt)
        if prompt_mask_img is None:
            prompt_mask_img = torch.ones_like(prompt_token_img[..., 0]) # (B, Nprompt)

        with torch.autocast(
            grasp_feats.device.type, 
            torch.bfloat16 if fp16 else torch.float32
        ):
            grasp_feats = self.vla_attn(
                query=grasp_feats, 
                value=torch.cat([prompt_token_img, prompt_token_lang], dim=1),
                value_mask=torch.cat([prompt_mask_img, prompt_mask_lang], dim=-1).bool(), 
                film=film
            )
            # grasp_feats = self.final_norm(grasp_feats)
            grasp_logits = self.act_logits_head(grasp_feats).squeeze(-1)
            if grasp_masks is not None:
                grasp_logits = grasp_logits.masked_fill(~grasp_masks, float('-inf'))

        if inference:
            pred_select_grasp = grasp_logits.argmax(dim=-1)
            select_grasp_pose = grasp_poses[0][pred_select_grasp.item()] # (4, 4)
            print("grasp_logits", grasp_logits)
            return select_grasp_pose

        # loss calculation
        loss = F.cross_entropy(grasp_logits, gt_select_grasp)
        metrics = {
            "total_loss": loss.item(),
        }
        return loss, metrics


def ee2cam(cur_wcT: Tensor, ee_states: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        ee_states (Tensor): (B, T, 4, 4), ^{world} T _{ee}
    """
    cur_wcT = cur_wcT.float()
    ee_states = ee_states.float()
    ee_states_cam = torch.inverse(cur_wcT[:, None]) @ ee_states

    r = matrix_to_rotation_6d(ee_states_cam[:, :, :3, :3])
    t = ee_states_cam[:, :, :3, 3]
    t3r6 = torch.cat([t, r], dim=-1)

    return t3r6


def space_ee2cam(cur_wcT: Tensor, cur_weT: Tensor, fut_weT: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        fut_weT (Tensor): (B, T, 4, 4), future ee pose in world frame
    
    Returns:
        t3r6 (Tensor): some repr of ^{cam} v _{ee} * dt, shape (B, T, 9)
    """
    e1e2T = torch.inverse(cur_weT[:, None]) @ fut_weT  # (B, T, 4, 4)
    e1e2R = e1e2T[:, :, :3, :3]  # (B, T, 3, 3)
    e1e2t = e1e2T[:, :, :3, 3]  # (B, T, 3)

    ceT = torch.inverse(cur_wcT) @ cur_weT  # (B, 4, 4)
    ceR = ceT[:, :3, :3]  # (B, 3, 3)
    
    r = matrix_to_rotation_6d(ceR[:, None] @ e1e2R @ ceR[:, None].transpose(-1, -2))
    t = (ceR[:, None] @ e1e2t.unsqueeze(-1)).squeeze(-1)
    t3r6 = torch.cat([t, r], dim=-1)
    return t3r6


def space_cam2ee(cur_wcT: Tensor, cur_weT: Tensor, t3r6: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        t3r6 (Tensor): (B, T, 9)
    
    Returns:
        fut_weT (Tensor), future ee pose in world frame, shape (B, T, 4, 4)
    """
    ecT = torch.inverse(cur_weT) @ cur_wcT  # (B, 4, 4)
    ecR = ecT[:, :3, :3]  # (B, 3, 3)
    
    e1e2R = ecR[:, None] @ rotation_6d_to_matrix(t3r6[..., 3:]) @ ecR[:, None].transpose(-1, -2)
    e1e2t = (ecR[:, None] @ t3r6[..., :3].unsqueeze(-1)).squeeze(-1)
    
    e1e2T = e1e2t.new_zeros(*e1e2t.shape[:-1], 4, 4)
    e1e2T[..., :3, :3] = e1e2R
    e1e2T[..., :3, 3] = e1e2t
    e1e2T[..., 3, 3] = 1

    fut_weT = cur_weT[:, None] @ e1e2T
    return fut_weT


def states2action(cur_wcT: Tensor, cur_weT: Tensor, ee_states: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        ee_states (Tensor): (B, T, 4, 4), ^{world} T _{ee}
    
    Returns:
        action (Tensor): (B, T, 9 or 10)
    """
    weT = ee_states.float()
    cur_wcT = cur_wcT.float()
    cur_weT = cur_weT.float()
    t3r6 = space_ee2cam(cur_wcT, cur_weT, weT)
    
    return t3r6


def count_parameters():
    model = ActionProb(
        hdim=256,
        num_heads=4,
        num_layers=4,
    )

    modules = [
        model
    ]

    num_param = 0
    for m in modules:
        for p in m.parameters():
            if not p.requires_grad:
                continue
            
            num_param += p.numel()

    print("[INFO] Total {:.3f}M trainable parameters"
          .format(num_param / 1e6))


if __name__ == "__main__":
    count_parameters()

