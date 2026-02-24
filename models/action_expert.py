import torch
import torch.nn.functional as F

from torch import nn, Tensor
from einops import rearrange
from typing import Optional, Tuple, Dict
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .encoders.clip_enc import Encoder
from .layers.utils import simple_mlp
from .layers.pe import SinusoidalPosEmb
from .layers.attn_dn import FFWXAttentionLayers, NormOrAdaLN, init_xncoder
from .layers.rot_transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from .utils.attn_vis import attn_vis

class ActionPooling(nn.Module):
    def __init__(self, hdim: int, act_dim: int):
        super().__init__()
        self.act_dim = act_dim  # transformed action dim
        self.conv1d = nn.Sequential(
            nn.Conv1d(act_dim - 1, hdim, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(hdim, hdim, kernel_size=3, padding=1), nn.AdaptiveAvgPool1d(1)
        )
        self.out = nn.Sequential(
            nn.ReLU(inplace=True), 
            nn.Linear(hdim, hdim), 
            nn.LayerNorm(hdim), 
        )

    def forward(self, history_action: Tensor):
        """
        Args:
            history_action (Tensor): (B, nhist, act_dim)
        
        Returns:
            film (Tensor): (B, hdim)
        """
        #### aggregate history_global_obs + history_action + prompt_token using self-attention layers
        history_action = rearrange(history_action[..., :self.act_dim-1], "b l c -> b c l")
        aggr: Tensor = self.conv1d(history_action)  # (B, C, 1)
        film: Tensor = self.out(aggr.squeeze(-1))  # (B, C)
        return film  # (B, C)


class ContextEncoder(nn.Module):
    def __init__(self, hdim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.context_attn = FFWXAttentionLayers(
            hdim, num_heads, layer_types="sf"*num_layers, use_adaln=True
        )
        self.enc_cet = simple_mlp([3, hdim, hdim])
        init_xncoder(num_layers*2, self.context_attn)
    
    def forward(
        self,
        context: Tensor, 
        context_mask: Optional[Tensor], 
        cet: Tensor, 
        fp16: bool
    ):
        """
        Args:
            context (Tensor): (B, Ncam, Lc, C)
            context_mask (Tensor): (B, Ncam, Lc) or None
            cet (Tensor): (B, 3)
            fp16 (bool):
        
        Returns
        -------
            context (Tensor): (B, Ncam*Lc, C)
            context_mask (Tensor): (B, Ncam*Lc, C)
        """
        context = rearrange(context, "b n l c -> b (n l) c")
        if context_mask is not None:
            context_mask = rearrange(context_mask, "b n l -> b (n l)")

        film = self.enc_cet(cet)

        with torch.autocast(
            context.device.type, 
            torch.bfloat16 if fp16 else torch.float32
        ):
            context = self.context_attn(
                query=context, 
                query_mask=context_mask, 
                value=context,
                value_mask=context_mask,
                film=film
            )
        
        return context, context_mask, film


class DiffusionHead(nn.Module):
    """
    - Input: image observations and noisy action at `t`
    - Output: noisy action at `t-1`
    """

    def __init__(self, hdim: int, num_heads: int, act_dim: int, num_layers: int, mlp_layers: int = 2):
        super().__init__()
        self.act_dim = act_dim
        self.traj_enc = simple_mlp([act_dim, hdim, hdim])
        self.traj_time_embed = SinusoidalPosEmb(hdim)
        self.denoising_time_embed = nn.Sequential(
            SinusoidalPosEmb(hdim),
            simple_mlp([hdim, hdim, hdim])
        )

        ### traj self attn + traj-context cross attn
        self.traj_context_attn = FFWXAttentionLayers(
            hdim, num_heads, 
            layer_types="scf"*num_layers, 
            use_adaln=True
        )
        init_xncoder(num_layers*2, self.traj_context_attn)

        ### traj-vl cross attn
        self.traj_vl_attn = FFWXAttentionLayers(
            hdim, num_heads, 
            layer_types="scf"*num_layers, 
            use_adaln=True
        )
        init_xncoder(num_layers*2, self.traj_vl_attn)

        ### final mlp
        self.final_norm = NormOrAdaLN(hdim, use_adaln=True)
        # self.act_head = simple_mlp([hdim, hdim, act_dim])

        # modify act_logits_head as multi-layer mlp
        mlp_neurons = [hdim] * mlp_layers + [act_dim]
        self.act_head = simple_mlp(mlp_neurons)

    def forward(
        self, 
        denoise_timestep: Tensor, 
        trajectory: Tensor, 
        context: Tensor,
        context_mask: Tensor, 
        cond: Tensor,
        cond_mask: Tensor,
        film: Tensor, 
        fp16: bool
    ):
        """
        Args:
            denoise_timestep: (B,), denoising time step
            trajectory: (B, Ta, act_dim)
            context: (B, Lo, hdim)
            context_mask: (B, Lo)
            cond: (B, Nprompt, hdim)
            cond_mask: (B, Nprompt)
            film: (B, hdim)
            fp16 (bool): use bfloat16

        Returns:
            action_epsilon: (B, Ta, act_dim)
        """
        
        denoise_time_embed = self.denoising_time_embed(denoise_timestep)  # (B, hdim)
        if film is not None:
            film = denoise_time_embed + film
        else:
            film = denoise_time_embed

        # noisy trajectory add temporal positional embeddings
        B, Ta, _ = trajectory.shape
        traj_feats = self.traj_enc(trajectory[:, :, :self.act_dim])  # (B, Ta, hdim)
        traj_time_pe = self.traj_time_embed(torch.arange(Ta).to(traj_feats))
        traj_feats = traj_feats + traj_time_pe[None].expand(B, -1, -1)  # (B, Ta, hdim)

        with torch.autocast(
            traj_feats.device.type, 
            torch.bfloat16 if fp16 else torch.float32
        ):
            traj_feats = self.traj_context_attn(
                query=traj_feats, 
                value=context,
                value_mask=context_mask, 
                film=film
            )

            traj_feats, attn_weights = self.traj_vl_attn(
                query=traj_feats, 
                value=cond,
                value_mask=cond_mask, 
                film=film,
                # return_hidden_states=True
                return_attn_weights=True
            )   


        traj_feats = self.final_norm(traj_feats, film=film)
        action = self.act_head(traj_feats)
        return action, attn_weights


class ActionExpert(nn.Module):
    def __init__(
        self, 
        hdim: int, 
        num_heads: int, 
        num_context_layers: int,
        num_diffusion_layers: int, 
        num_diffusion_mlp_layers: int = 2,
        diffusion_timesteps: int = 100, 
        inference_timesteps: Optional[int] = None, 
    ):
        super().__init__()
        self.ray_pe = nn.Linear(6, hdim)
        self.scene_encoder = Encoder(hdim)
        self.history_action_pool = ActionPooling(hdim, self.act_dim)
        self.context_encoder = ContextEncoder(hdim, num_heads, num_layers=num_context_layers)
        self.dp_head = DiffusionHead(hdim, num_heads, self.act_dim, num_layers=num_diffusion_layers, mlp_layers=num_diffusion_mlp_layers)

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )

        self.diffusion_timesteps = diffusion_timesteps
        if inference_timesteps is None:
            inference_timesteps = max(diffusion_timesteps//5, 10)
        self.inference_timesteps = inference_timesteps

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if m.bias is not None and m.bias.requires_grad:
                    # Do not modify the bias in DINOv2!!!
                    nn.init.zeros_(m.bias)

    @property
    def act_dim(self):
        """dimension of action defined in camera frame"""
        return 10

    def iterative_denoise(
        self, 
        traj_shape: Tuple[int, int, int],
        fixed_inputs: Dict[str, Tensor],
        initial_noise: Optional[Tensor] = None,
    ):
        """
        Args:
            trajectory_shape: (B, Ta, act_dim)
            fixed_inputs: inputs for diffusion head
            intial_noise: (B, Ta, act_dim) or None

        Returns:
            trajectory: (B, Ta, act_dim)
        """
        if initial_noise is None:
            B, Ta, _ = traj_shape
            device = next(iter(fixed_inputs.values())).device
            initial_noise = torch.randn(B, Ta, self.act_dim, device=device)
        
        self.noise_scheduler.set_timesteps(self.inference_timesteps)
        trajectory = initial_noise
        for t in self.noise_scheduler.timesteps:
            out, attn_weights = self.dp_head(
                t * torch.ones(trajectory.shape[0], device=trajectory.device), 
                trajectory,
                **fixed_inputs
            )

            trajectory = self.noise_scheduler.step(
                out[..., :self.act_dim], t, trajectory[..., :self.act_dim]
            ).prev_sample
        return trajectory, attn_weights

    def forward(
        self, 
        obs_rgbs: Tensor,
        obs_masks: Optional[Tensor], 
        obs_norm_xys: Tensor, 
        obs_extrinsics: Tensor, 

        current_ee_pose: Tensor, 
        history_ee_states: Tensor, 
        gt_future_ee_states: Tensor, 
        inference: bool, 
        fp16: bool,

        #### from vlm ####
        prompt_film: Optional[Tensor], 
        prompt_token_img: Optional[Tensor], 
        prompt_token_lang: Optional[Tensor], 
        prompt_mask_img: Optional[Tensor], 
        prompt_mask_lang: Optional[Tensor], 
        ##################
    ):
        """
        Args:
            obs_rgbs: (B, To, ncam, 3, H, W)
            obs_masks: (B, To, ncam, H, W)
            obs_norm_xys: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
            obs_extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T

            !!! Note current_ee_pose is the grasp pose !!!
            current_ee_pose: (B, 4, 4), ^{world}_{ee} T, 
            history_ee_states: (B, nhist, 4*4+1), in world frame,
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
            gt_future_ee_states: (B, Ta, 4*4+1), ground truth future actions, in world frame
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
                * Note: if `inference` is True, we only derive prediction actions shape from gt_future_ee_states
            inference: if True, returns the predicted trajectory, otherwise returns loss and metrics for logging
            fp16: if True, use bfloat16

            prompt_film: (B, hdim), output of vlm model
            prompt_token_img: (B, Nprompt, hdim), output of vlm model
            prompt_token_lang: (B, Nprompt, hdim), output of vlm model
            prompt_mask_img: (B, Nprompt), output of vlm model
            prompt_mask_lang: (B, Nprompt), output of vlm model

        Returns
        -------
        (if inference is True)
            pred_future_ee_states (Tensor): (B, Ta, 4*4+1)
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
        (else)
            loss (Tensor): scalar tensor
            metrics (Dict[str, Tensor]): metrics for logging
        """
        # get relative camera extrinsic referring to camera 0 at the latest timestep
        cam_extr_ref0 = torch.inverse(obs_extrinsics[:, -1:, 0:1]) @ obs_extrinsics
        obs_info = self.scene_encoder.encode_obs(
            rgb=obs_rgbs, 
            mask=obs_masks, 
            norm_xy=obs_norm_xys, 
            extrinsic=cam_extr_ref0
        )
        # obs_info (Dict[str, Tensor]): image observation from image encoder, 
        # - x:    tensor of shape (B, To, Ncam, L, hdim) patch feature
        # - gx:   tensor of shape (B, To, Ncam, hdim), projected global feature, no mask applied
        # - pe:   tensor of shape (B, To, Ncam, L, 6), ray pe
        # - mask: tensor of shape (B, To, Ncam, L) or None, patch mask
        # - aux:  {aux_user_key: tensor of shape (B, To, Ncam, L, ...)}

        latest_cam_poses = obs_extrinsics[:, -1]  # (B, Ncam, 4, 4)
        current_cam_pose = latest_cam_poses[:, 0]
        history_action = states2action(current_cam_pose, current_ee_pose, history_ee_states)
        
        B, Ta, _ = gt_future_ee_states.shape
        if not inference:
            gt_future_action = states2action(current_cam_pose, current_ee_pose, 
                                             gt_future_ee_states)
        
        # prepare data for film generation
        ceT = torch.inverse(latest_cam_poses) @ current_ee_pose[:, None]  # (B, Ncam, 4, 4)
        cet = ceT[:, :, :3, 3]  # (B, Ncam, 3)

        history_film: Tensor = self.history_action_pool(
            history_action=history_action,  # history in camera 0, shape (B, nhist, act_dim)
        )  # (B, hdim)

        # patch features as current observation context in diffusion, for self-attention
        x: Tensor = obs_info["x"]  # (B, To, Ncam, Lo, hdim)
        pe: Tensor = obs_info["ray_pe"]  # (B, To, Ncam, Lo, 6)
        mask: Tensor = obs_info["mask"]  # (B, To, Ncam, Lo)

        # add ray positional embedding
        context = x[:, -1] + self.ray_pe(pe[:, -1])  # only use the latest frame, (B, Ncam, Lo, hdim)
        if mask is None:
            context_mask = None
        else:
            # only use the latest frame
            context_mask = mask[:, -1]
        
        context, context_mask, context_film = self.context_encoder(
            context=context,
            context_mask=context_mask,
            cet=cet[:, 0],  # first camera, (B, 3)
            fp16=fp16
        )

        if prompt_mask_lang is None:
            prompt_mask_lang = torch.ones_like(prompt_token_lang[..., 0])
        if prompt_mask_img is None:
            prompt_mask_img = torch.ones_like(prompt_token_img[..., 0])

        # prepare inputs for conditional sample
        fixed_inputs = dict(
            context=context,
            context_mask=context_mask,
            cond=torch.cat([prompt_token_img, prompt_token_lang], dim=1),
            cond_mask=torch.cat([prompt_mask_img, prompt_mask_lang], dim=1),
            film=history_film + context_film,
            fp16=fp16
        )

        ###################### Inference ######################
        if inference:
            pred_actions, attn_weights = self.iterative_denoise(
                traj_shape=(B, Ta, self.act_dim),
                fixed_inputs=fixed_inputs,
            )  # (B, Ta, act_dim)
            pred_future_ee_states = action2states(
                current_cam_pose, current_ee_pose, pred_actions
            )  # (B, Ta, 4*4+1)

            # visualize attention weights on obs_rgbs
            img_attn_0 = attn_weights[-1][0][:, :256].sum(dim=0)  # (Lo)
            img_attn_1 = attn_weights[-1][0][:, 256:512].sum(dim=0)  # (Lo)
            lang_attn = attn_weights[-1][0][:, 512:].sum(dim=0)  # (Nprompt)
            print("lang_attn: ", lang_attn)

            # visualize image attention weights
            img_attn_0 = img_attn_0.cpu().numpy()
            img_attn_1 = img_attn_1.cpu().numpy()
            # obs_rgbs: (B, To, ncam, 3, H, W)
            rgb_0 = obs_rgbs[0, -1, 0].cpu().numpy().transpose(1, 2, 0) * 255.0
            attn_vis(img_attn_0, rgb_0, win_name="img_attn_0")
            if obs_rgbs.shape[2] > 1:
                rgb_1 = obs_rgbs[0, -1, 1].cpu().numpy().transpose(1, 2, 0) * 255.0
                attn_vis(img_attn_1, rgb_1, win_name="img_attn_1")

            return pred_future_ee_states

        ###################### Training ######################
        # sample noise
        noise = torch.randn(B, Ta, self.act_dim, 
                            device=gt_future_ee_states.device)

        # sample a random timestep
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            size=(B,), 
            device=noise.device
        )

        # add noise to the clean trajectory
        noisy_trajectory = self.noise_scheduler.add_noise(
            gt_future_action, noise,
            timesteps
        )

        pred, inter_feats = self.dp_head(timesteps, noisy_trajectory, **fixed_inputs)
        target = get_target(gt_future_action, noise, timesteps, self.noise_scheduler)

        # loss calculation
        pos_loss = F.l1_loss(pred[..., 0:3], target[..., 0:3], reduction="mean")
        rot_loss = F.l1_loss(pred[..., 3:9], target[..., 3:9], reduction="mean")
        openness_loss = F.l1_loss(pred[..., 9:10], target[..., 9:10], reduction="mean")

        total_loss = 30 * pos_loss + 10 * rot_loss + 10 * openness_loss

        metrics = {
            "pos_loss": pos_loss.item(),
            "rot_loss": rot_loss.item(),
            "openness_loss": openness_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics


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
        ee_states (Tensor): (B, T, 16 or 17)
    
    Returns:
        action (Tensor): (B, T, 9 or 10)
    """
    B, Ta, C = ee_states.shape
    weT = ee_states[:, :, :16].view(B, Ta, 4, 4)
    t3r6 = space_ee2cam(cur_wcT, cur_weT, weT)
    
    if C == 16:
        return t3r6
    else:
        openness = (ee_states[:, :, -1:] - 0.5) * 2  # normalize gripper openness
        return torch.cat([t3r6, openness], dim=-1)


def action2states(cur_wcT: Tensor, cur_weT: Tensor, action: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        action (Tensor): (B, T, 9 or 10)
    
    Returns:
        ee_states (Tensor): (B, T, 16 or 17)
    """
    B, Ta, C = action.shape
    t3r6 = action[:, :, :9]
    weT = space_cam2ee(cur_wcT, cur_weT, t3r6).view(B, Ta, 16)
    
    if C == 9:
        return weT
    else:
        openness = action[:, :, -1:] / 2 + 0.5  # denormalize gripper openness
        return torch.cat([weT, openness], dim=-1)


def get_target(traj: Tensor, noise: Tensor, timesteps: Tensor, scheduler: DDIMScheduler):
    """returns supervision depending on scheduler type"""
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        target = noise
    if pred_type == "sample":
        target = traj
    if pred_type == "v_prediction":
        target = scheduler.get_velocity(traj, noise, timesteps) 
    return target


def count_parameters():
    model = ActionExpert(
        hdim=256,
        diffusion_timesteps=100,
        num_heads=4,
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

