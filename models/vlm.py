import torch
from einops import rearrange
from torch import nn, Tensor
from typing import Optional, List
from models.encoders.clip_enc import Encoder
from models.layers.pe import RoPE


class VLOutput(object):
    def __init__(
        self,
        prompt_film: Optional[Tensor] = None, 
        prompt_token_img: Optional[Tensor] = None,
        prompt_token_lang: Optional[Tensor] = None,
        prompt_mask_img: Optional[Tensor] = None,
        prompt_mask_lang: Optional[Tensor] = None,
    ):
        self.prompt_film = prompt_film    # (B, hdim)
        self.prompt_token_img = prompt_token_img  # (B, L, hdim)
        self.prompt_token_lang = prompt_token_lang  # (B, L, hdim)
        self.prompt_mask_img = prompt_mask_img    # (B, L)
        self.prompt_mask_lang = prompt_mask_lang    # (B, L)
    
    def asdict(self):
        return vars(self)


class VLM(nn.Module):
    def __init__(
        self, 
        hdim: int,
        random_mask_vl: bool=False,
        contact_phase: str="pre",
    ):
        super().__init__()
        self.clip = Encoder(hdim)
        self.random_mask_vl = random_mask_vl
        self.contact_phase = contact_phase

        if self.contact_phase == "pre":
            self.rope = RoPE(hdim, 2)
        elif self.contact_phase == "post":
            self.ray_pe = nn.Linear(6, hdim)
        else:
            raise ValueError(f"Invalid contact phase: {self.contact_phase}")

    def forward(
        self, 
        obs_rgbs: Tensor,
        obs_masks: Tensor, 
        obs_norm_xys: Tensor, 
        obs_extrinsics: Tensor, 
        prompt_text: List[str], 
        fp16: bool
    ):
        """
        Args:
            obs_rgbs: (B, To, ncam, 3, H, W)
            obs_masks: (B, To, ncam, H, W)
            obs_norm_xys: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
            obs_extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            prompt_text (List[str]): len = B
            fp16 (bool): 
        """
        B, T, N, _, _, _ = obs_rgbs.shape
        clip_output = self.clip(obs_rgbs, prompt_text, obs_masks, obs_norm_xys, obs_extrinsics)

        x_vision: Tensor = clip_output["x"]  # (B, T, N, L, C)
        vision_mask: Tensor = clip_output["mask"]  # (B, T, N, L)
        norm_pe_vision: Tensor = clip_output["norm_xy_pe"]  # (B, T, N, L, 2)
        ray_pe_vision: Tensor = clip_output["ray_pe"] # (B, T, N, L, 2)
        x_text: Tensor = clip_output["x_text"]  # (B, Ltext, C)
        text_mask: Tensor = clip_output["text_mask"]  # (B, Ltext)

        if self.contact_phase == "pre":
            pe_vision = self.rope(norm_pe_vision)
            x_vision = self.rope.embed_rotary(x_vision, pe_vision)
        elif self.contact_phase == "post":
            x_vision = x_vision + self.ray_pe(ray_pe_vision)
        
        x_vision = rearrange(x_vision, "b t n l c -> b (t n l) c") # (B, T*N*L, C)
        if vision_mask is not None:
            vision_mask = rearrange(vision_mask, "b t n l -> b (t n l)") # (B, T*N*L)

        if self.random_mask_vl:
            mask_lang = torch.rand(B, device=x_text.device) < 0.1  # 10% unconditional
            text_mask = mask_lang.logical_not()[:, None].repeat(1, x_text.shape[1])

        return VLOutput(
            prompt_film=None,
            prompt_token_img=x_vision,
            prompt_token_lang=x_text,
            prompt_mask_img=vision_mask,
            prompt_mask_lang=text_mask,
        )

