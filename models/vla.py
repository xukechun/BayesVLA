from typing import Optional
from torch import nn, Tensor

from .action_expert import ActionExpert
from .action_prob import ActionProb
from .vlm import VLM, VLOutput


class PreContactVLA(nn.Module):
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
        self.vlm = VLM(
            hdim=hdim,
            contact_phase="pre",
        )
        self.actor = ActionProb(
            hdim=hdim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )
    
    def forward(
        self, 
        obs_rgbs: Tensor,
        obs_masks: Optional[Tensor], 
        obs_norm_xys: Tensor, 
        obs_extrinsics: Tensor, 
        obs_intrinsics: Tensor,

        prompt_text: Tensor, 

        grasp_poses: Tensor,
        grasp_masks: Optional[Tensor],
        gt_select_grasp: Tensor,

        inference: bool, 
        fp16: bool,
    ):
        """
        Args:
            obs_rgbs: (B, To, ncam, 3, H, W)
            obs_masks: (B, To, ncam, H, W)
            obs_norm_xys: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
            obs_extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            obs_intrinsics: (B, ncam, 3, 3), intrinsic matrix of the camera

            prompt_text: (B, Lang, E) or None, language instruction

            grasp_poses: (B, Ng, 4, 4), ^{world}_{ee} T
            grasp_masks: (B, Ng), whether the grasp pose is valid
            gt_select_grasp: (B, 1)

            inference: if True, returns the predicted trajectory, otherwise returns loss and metrics for logging
            fp16: if True, use bfloat16
        
        Returns
        -------
        (if inference is True)
            pred_future_ee_states (Tensor): (B, 4*4+1)
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
        (else)
            loss (Tensor): scalar tensor
            metrics (Dict[str, Tensor]): metrics for logging
        """

        ### select the latest frame for higher execution frequency of action expert
        ### select the agentview camera
        obs_rgbs = obs_rgbs[:, -1:, 0:1]                # (B, To=1, Ncam=1, 3, H, W)
        if obs_masks is not None:
            obs_masks = obs_masks[:, -1:, 0:1]          # (B, To=1, Ncam=1, H, W)
        obs_norm_xys = obs_norm_xys[:, -1:, 0:1]        # (B, To=1, Ncam=1, 2, H, W)
        obs_extrinsics = obs_extrinsics[:, -1:, 0:1]    # (B, To=1, Ncam=1, 4, 4)
        obs_intrinsics = obs_intrinsics[:, 0:1]         # (B, Ncam=1, 3, 3)

        vlm_out: VLOutput = self.vlm(
            obs_rgbs=obs_rgbs,
            obs_masks=obs_masks,
            obs_norm_xys=obs_norm_xys,
            obs_extrinsics=obs_extrinsics,
            prompt_text=prompt_text,
            fp16=fp16
        )

        return self.actor(
            obs_extrinsics=obs_extrinsics,
            obs_intrinsics=obs_intrinsics,

            grasp_poses=grasp_poses,
            grasp_masks=grasp_masks,
            gt_select_grasp=gt_select_grasp,
            inference=inference,
            fp16=fp16,

            **vlm_out.asdict()
        )

class PostContactVLA(nn.Module):
    def __init__(
        self, 
        hdim: int, 
        num_heads: int,

        num_actor_context_layers: int,
        num_actor_diffusion_layers: int, 
        num_actor_diffusion_mlp_layers: int = 2,

        diffusion_timesteps: int = 100, 
        inference_timesteps: Optional[int] = None, 
    ):
        super().__init__()
        self.vlm = VLM(
            hdim=hdim,
            contact_phase="post",
        )
        self.actor = ActionExpert(
            hdim=hdim,
            num_heads=num_heads,
            num_context_layers=num_actor_context_layers,
            num_diffusion_layers=num_actor_diffusion_layers, 
            num_diffusion_mlp_layers=num_actor_diffusion_mlp_layers,
            diffusion_timesteps=diffusion_timesteps,
            inference_timesteps=inference_timesteps,
        )
    
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

        prompt_text: Optional[Tensor] = None,
    ):
        """
        Args:
            obs_rgbs: (B, To, ncam, 3, H, W)
            obs_masks: (B, To, ncam, H, W)
            obs_norm_xys: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
            obs_extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T

            current_ee_pose: (B, 4, 4), ^{world}_{ee} T
            history_ee_states: (B, nhist, 4*4+1), in world frame,
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
            gt_future_ee_states: (B, Ta, 4*4+1), ground truth future actions, in world frame
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
                * Note: if `inference` is True, we only derive prediction actions shape from gt_future_ee_states
            inference: if True, returns the predicted trajectory, otherwise returns loss and metrics for logging
            fp16: if True, use bfloat16
        
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

        ### select the latest frame for higher execution frequency of action expert
        obs_rgbs = obs_rgbs[:, -1:]             # (B, To=1, ncam, 3, H, W)
        if obs_masks is not None:
            obs_masks = obs_masks[:, -1:]       # (B, To=1, ncam, H, W)
        obs_norm_xys = obs_norm_xys[:, -1:]     # (B, To=1, ncam, 2, H, W)
        obs_extrinsics = obs_extrinsics[:, -1:] # (B, To=1, ncam, 4, 4)

        vlm_out: VLOutput = self.vlm(
            obs_rgbs=obs_rgbs,
            obs_masks=obs_masks,
            obs_norm_xys=obs_norm_xys,
            obs_extrinsics=obs_extrinsics,
            prompt_text=prompt_text,
            fp16=fp16
        )

        return self.actor(
            obs_rgbs=obs_rgbs,
            obs_masks=obs_masks,
            obs_norm_xys=obs_norm_xys,
            obs_extrinsics=obs_extrinsics,

            current_ee_pose=current_ee_pose,
            history_ee_states=history_ee_states,
            gt_future_ee_states=gt_future_ee_states,
            inference=inference,
            fp16=fp16,

            **vlm_out.asdict()
        )


def pre_vla_tiny():
    return PreContactVLA(
        hdim=192,
        num_heads=3,
        num_layers=4,
    )


def pre_vla_small():
    return PreContactVLA(
        hdim=384,
        num_heads=6,
        num_layers=4,
        mlp_layers=2,
        # attn_dropout=0.1,
    )


def pre_vla_base():
    return PreContactVLA(
        hdim=768,
        num_heads=12,
        num_layers=4,
    )


def pre_vla_large():
    return PreContactVLA(
        hdim=1024,
        num_heads=16,
        num_layers=8,
    )

def post_vla_tiny(
    diffusion_timesteps: int = 100, 
    num_actor_diffusion_mlp_layers: int = 2,
    inference_timesteps: Optional[int] = None, 
):
    return PostContactVLA(
        hdim=192,
        num_heads=3,
        num_actor_context_layers=4,
        num_actor_diffusion_layers=4,
        num_actor_diffusion_mlp_layers=num_actor_diffusion_mlp_layers,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps
    )


def post_vla_small(
    diffusion_timesteps: int = 100, 
    num_actor_diffusion_mlp_layers: int = 2,
    inference_timesteps: Optional[int] = None, 
):
    return PostContactVLA(
        hdim=384,
        num_heads=6,
        num_actor_context_layers=4,
        num_actor_diffusion_layers=4,
        num_actor_diffusion_mlp_layers=num_actor_diffusion_mlp_layers,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps
    )


def post_vla_base(
    diffusion_timesteps: int = 100, 
    num_actor_diffusion_mlp_layers: int = 2,
    inference_timesteps: Optional[int] = None, 
):
    return PostContactVLA(
        hdim=768,
        num_heads=12,
        num_actor_context_layers=4,
        num_actor_diffusion_layers=4,
        num_actor_diffusion_mlp_layers=num_actor_diffusion_mlp_layers,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps
    )


def post_vla_large(
    diffusion_timesteps: int = 100, 
    num_actor_diffusion_mlp_layers: int = 4,
    inference_timesteps: Optional[int] = None, 
):
    return PostContactVLA(
        hdim=1024,
        num_heads=16,
        num_actor_context_layers=8,
        num_actor_diffusion_layers=8,
        num_actor_diffusion_mlp_layers=num_actor_diffusion_mlp_layers,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps
    )


def count_parameters(contact_phase: str="pre"):
    if contact_phase == "pre":
        # model = pre_vla_tiny()
        # model = pre_vla_small()
        model = pre_vla_base()
        # model = pre_vla_large()
    elif contact_phase == "post":
        # model = post_vla_tiny()
        # model = post_vla_small()
        model = post_vla_base()
        # model = post_vla_large()
    else:
        raise ValueError(f"Invalid contact phase: {contact_phase}")

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
    count_parameters(contact_phase="pre")
    count_parameters(contact_phase="post")