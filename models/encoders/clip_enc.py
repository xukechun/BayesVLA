import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from einops import rearrange
from torch import nn, Tensor
from torchvision.transforms import v2
from typing import Tuple, List, Optional
from timm.models.vision_transformer import RmsNorm
from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer
from ..layers.rot_transforms import matrix_to_rotation_6d


# Create mask for SOS, EOS, and padding tokens
def create_text_mask(text: torch.Tensor, sot_token: int, eot_token: int, first_k_tokens: int) -> torch.Tensor:
    """
    text: batch of text tokens (B, 77)
    sot_token: start of text token
    eot_token: end of text token
    first_k_tokens: number of tokens to consider

    For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
    the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.

    mask will be true at start of text, end of text, and padding tokens
    mask will be false at actual tokens
    """
    # make sure text is either dim 1 or 2 
    text_dim = len(text.shape)
    assert text_dim in (1, 2), "text must be either 1D or 2D"
    text_mask = torch.zeros_like(text, dtype=torch.bool)
    text_mask[text == sot_token] = True  # Start token
    text_mask[text == eot_token] = True  # End token
    text_mask[text == 0] = True  # Padding token
    # Only consider first k tokens
    if text_dim == 1:
        text_mask = text_mask[:first_k_tokens]    
    else:
        text_mask = text_mask[:, :first_k_tokens]  
    return text_mask


class TextAwareVisualExtraction(nn.Module):
    """Extract text-aware visual features using CLIP, following ClearCLIP approach"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, image_patch_feats: torch.Tensor, text_feats: torch.Tensor, image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        similarity = torch.einsum('bij,bkj->bik', text_feats, image_patch_feats)
        similarity = similarity / self.temperature.clamp(0, 100)

        if image_mask is not None:
            similarity = similarity.masked_fill(image_mask.logical_not().unsqueeze(1), -torch.inf)

        attention = F.softmax(similarity, dim=-1)

        # Get text-aware visual features by combining patch features according to attention (with position embedding)
        text_aware_features = torch.einsum('bik,bkj->bij', attention, image_patch_feats)
        
        return text_aware_features

    def gtlv_forward(self, image_patch_feats: torch.Tensor, text_feats: torch.Tensor, scale: float = 1.0, bias: float = 0.0) -> torch.Tensor:
        # image_patch_features: (B, L, C)
        # text_features: (B, C)
        text_feats_expanded = text_feats.unsqueeze(1).repeat(image_patch_feats.shape[0] // text_feats.shape[0], image_patch_feats.shape[1], 1)
        similarity = torch.cosine_similarity(image_patch_feats, text_feats_expanded, dim=-1)
        attention = F.softmax(similarity * scale + bias, dim=-1)

        # Get text-aware visual features by combining patch features according to attention
        text_aware_features = attention.unsqueeze(-1) * image_patch_feats
        
        return text_aware_features, attention

    def gtv_forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor, scale: float = 1.0, bias: float = 0.0) -> torch.Tensor:
        # image_patch_features: (B, C)
        # text_features: (B, C)
        text_feats_expanded = text_feats.repeat(image_feats.shape[0] // text_feats.shape[0], 1)
        similarity = torch.cosine_similarity(image_feats, text_feats_expanded, dim=-1)
        attention = similarity * scale + bias

        # Get text-aware visual features by combining patch features according to attention
        text_aware_features = attention.unsqueeze(-1) * image_feats

        return text_aware_features, attention
    
class ClipEncoder(nn.Module):
    REPO_ID = "ViT-L/14"
    # CACHE_DIR = "./cache/siglip-base-patch16-256"
    # PREPROCESS_DIR = os.path.join(CACHE_DIR, "preprocess")

    def __init__(self, pool_true_text: bool = False, first_k_tokens: int = 64):
        super().__init__()

        # cache_found = os.path.exists(self.CACHE_DIR)
        # run_fp16 = torch.cuda.is_bf16_supported()

        self.pool_true_text = pool_true_text
        self.first_k_tokens = first_k_tokens

        # Load CLIP model
        self.clip_model, _ = clip.load(self.REPO_ID)
        self.tokenizer = SimpleTokenizer()
        self.sot_token : int = self.tokenizer.encoder["<|startoftext|>"]
        self.eot_token : int = self.tokenizer.encoder["<|endoftext|>"]

        input_res = self.clip_model.visual.input_resolution
        self.image_preprocess = v2.Compose([
            v2.Resize(
                size=input_res, 
                interpolation=v2.InterpolationMode.BICUBIC
            ),
            v2.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])
        # Store activations dict at instance level
        self.activation = {}

        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook

        # Register hooks once
        self.hooks = [
            self.clip_model.visual.transformer.resblocks[-1].attn.register_forward_hook(
                get_activation('image_patches')
            ),
            self.clip_model.transformer.register_forward_hook(
                get_activation('text_features')
            )
        ]       

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.image_preprocess(images)
        with torch.no_grad():
            gx = self.clip_model.encode_image(images)
        return gx

    def encode_texts(self, text: List[str]) -> torch.Tensor:
        text = clip.tokenize(text).to(self.clip_model.text_projection.device) # (B, max_text_len)
        with torch.no_grad():
            gx = self.clip_model.encode_text(text)
        return gx

    def extract_token_features(
        self, 
        images: torch.Tensor,  # (B, T, num_cameras, C, H, W)
        text: List[str],    # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, num_cameras = images.shape[:3]

        # Reshape images to process all views at once
        images = rearrange(images, "b t n c h w -> (b t n) c h w") # (B*T*num_cameras, C, H, W)
        # preprocess images
        images = self.image_preprocess(images)

        # preprocess text
        text = clip.tokenize(text).to(images.device) # (B, max_text_len)
        # Create mask for SOS, EOS, and padding tokens
        if self.pool_true_text:
            text_mask = create_text_mask(text, self.sot_token, self.eot_token, self.first_k_tokens)
            text_mask = text_mask.to(images.device)
        else:
            text_mask = None

        # Get features
        with torch.no_grad():
            gx_text = self.clip_model.encode_text(text)
            gx = self.clip_model.encode_image(images)
        
        # Process text features. self.activation is a hook!!!
        text_features = self.activation['text_features'].permute(1, 0, 2)[:, :self.first_k_tokens]
        text_features = self.clip_model.ln_final(text_features).type(self.clip_model.dtype) @ self.clip_model.text_projection
        text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True) # (B, first_k_tokens, text_dim)
        
        # repeat text features T times 
        # should be equivalent to 
        # text_features = text_features.unsqueeze(1).repeat(1, 3, 1, 1).view(B*T, self.first_k_tokens, text_features.shape[-1])
        text_features_normalized = text_features_normalized.repeat_interleave(T, dim=0) # (B*T, first_k_tokens, text_dim)
        
        # Process patch features. self.activation is a hook!!!
        patch_features = self.activation['image_patches'][0]
        patch_features = patch_features.permute(1, 0, 2)
        
        # get rid of the cls token 
        patch_features = patch_features[:, 1:]

        patch_features = self.clip_model.visual.ln_post(patch_features)
        
        if self.clip_model.visual.proj is not None:
            patch_features = patch_features @ self.clip_model.visual.proj
            
        patch_features_normalized = patch_features / patch_features.norm(dim=-1, keepdim=True)
        
        # Reshape patch features back to separate cameras
        patch_features_normalized = patch_features_normalized.view(B * T, num_cameras, *patch_features_normalized.shape[1:])
        
        # Split for each camera
        patch_features_per_camera = [patch_features_normalized[:, i] for i in range(num_cameras)]
        
        output = {
            "gx": gx.float(),
            "gx_text": gx_text.float(),
            "x": patch_features.float(),
            "x_norm": patch_features_normalized.float(),
            "x_percam_norm": [x.float() for x in patch_features_per_camera], 
            "x_text": text_features.float(),
            "x_text_norm": text_features_normalized.float(), 
            "text_mask": text_mask,
        }
        return output


class FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = ClipEncoder()
        self.resize2 = v2.Resize(224, v2.InterpolationMode.BILINEAR)
        self.resize0 = v2.Resize(224, v2.InterpolationMode.NEAREST)
    
    def encode_token_features(self, rgb: Tensor, text: List[str]):
        clip_output = self.clip.extract_token_features(rgb, text)
        return clip_output
    
    def encode_images(self, rgb: Tensor):
        return self.clip.encode_images(rgb).float()
    
    def encode_texts(self, text: List[str]):
        return self.clip.encode_texts(text).float()
    
    def encode_aux(self, a: Tensor):
        B, C, H, W = a.shape
        assert H == W
        
        bool_type = a.dtype == torch.bool
        float_type = a.is_floating_point()
        int_type = (not bool_type) and (not float_type)
        
        if float_type:
            a_resize = self.resize2(a)
        else:
            a_resize = self.resize0(a.float())
        
        a_ds = F.avg_pool2d(a_resize, (14, 14))
        if bool_type:
            a_ds = a_ds > 1e-4
        elif int_type:
            a_ds = a_ds.to(a.dtype)
        a_ds = rearrange(a_ds, "b c h w -> b (h w) c")
        return a_ds


class Encoder(nn.Module):
    def __init__(self, hdim, ):
        super().__init__()
        self.frozen = FrozenEncoder()
        for p in self.frozen.parameters():
            p.requires_grad_(False)

        self.visual_extraction = TextAwareVisualExtraction()

        # siglip l2 norm the feature for image-text matching
        self.proj_vision_patches = nn.Sequential(nn.Linear(768, hdim), RmsNorm(hdim, affine=True))
        self.proj_vision_global = nn.Sequential(nn.Linear(768, hdim), RmsNorm(hdim, affine=False))
        self.proj_text_tokens = nn.Sequential(nn.Linear(768, hdim), RmsNorm(hdim, affine=True))
        self.proj_text_global = nn.Sequential(nn.Linear(768, hdim), RmsNorm(hdim, affine=False))


    def encode_images(self, rgb: Tensor):
        with torch.no_grad():
            x, gx = self.frozen.encode_images(rgb)
        return x, gx

    def encode_text(self, texts: List[str]):
        with torch.no_grad():
            x, gx = self.frozen.encode_texts(texts)
        return x, gx

    def encode_token_features(self, rgb: Tensor, texts: List[str]):
        with torch.no_grad():
            output = self.frozen.encode_token_features(rgb, texts)
        return output
    
    def encode_aux(self, a: Tensor):
        return self.frozen.encode_aux(a)
    
    def encode_extrinsic(self, extr: Tensor) -> Tensor:
        t3 = extr[..., :3, 3]
        r6 = matrix_to_rotation_6d(extr[..., :3, :3])
        return self.proj_extr(torch.cat([t3, r6], dim=-1))

    def encode_obs(
        self, 
        rgb: Tensor, 
        mask: Tensor, 
        norm_xy: Tensor, 
        extrinsic: Tensor, 
        **aux_tensors: Tensor
    ):
        """
        Args:
            rgb (Tensor): (B, T, N, 3, H, W)
            mask (Tensor): (B, T, N, H, W)
            norm_xy (Tensor): (B, T, N, 2, H, W)
            extrinsic (Tensor): (B, T, N, 4, 4), ^ref_cam T
            aux_tensors (Tensor): each of (B, T, N, C, H, W)

        Returns:
            obs (Dict[str, Tensor]): image observations
                - x:    tensor of shape (B, To, Ncam, L, C) patch feature
                - gx:   tensor of shape (B, To, Ncam, C), projected global feature, not masked
                - pe:   tensor of shape (B, To, Ncam, L, 6), ray pe
                - mask: tensor of shape (B, To, Ncam, L) or None, patch mask
                - aux:  {aux_user_key: tensor of shape (B, To, Ncam, L, ...)}
        """
        B, T, N, C, H, W = rgb.shape

        clip_output = self.encode_token_features(rgb, []) # list of (B*T, L, C)
        x_ds_list = clip_output["x_percam_norm"]

        gx = clip_output["gx"]
        x_ds = clip_output["x"]

        if mask is not None:
            mask = rearrange(mask, "b t n h w -> (b t n) () h w")
            mask_ds = self.encode_aux(mask)
            mask_ds = rearrange(mask_ds, "(b t n) l 1 -> (b t) n l", b=B, t=T, n=N)
            mask_ds_list = mask_ds.split(1, dim=1)
        else:
            mask_ds = None
            mask_ds_list = [None] * N

        x_ds = self.proj_vision_patches(x_ds)
        gx = self.proj_vision_global(gx)

        x_ds = rearrange(x_ds, "(b t n) l c -> b t n l c", b=B, t=T, n=N)
        gx = rearrange(gx, "(b t n) c -> b t n c", b=B, t=T, n=N)
        
        mask_ds = rearrange(mask_ds, "(b t) n l -> b t n l", b=B, t=T, n=N)

        aux_tensors["norm_xy"] = norm_xy
        aux_tensors = {n: (rearrange(a, "b t n ... -> (b t n) ...") 
                           if a is not None else a) for n, a in aux_tensors.items()}
        aux_ds = {n: self.encode_aux(a) if a is not None else a 
                  for n, a in aux_tensors.items()}
        aux_ds = {n: (rearrange(a, "(b t n) ... -> b t n ...", b=B, t=T, n=N) 
                      if a is not None else a) for n, a in aux_ds.items()}
        norm_xy_ds = aux_ds.pop("norm_xy")
        # pe = norm_xy_ds
        ray_pe = plucker_ray_pe(norm_xy_ds, extrinsic)

        return {
            "x": x_ds,                  # (B, T, N, L, C)
            "gx": gx,                   # (B, T, N, C)
            "ray_pe": ray_pe,           # (B, T, N, L, 6)
            "norm_xy_pe": norm_xy_ds,   # (B, T, N, L, 2)
            "mask": mask_ds,            # (B, T, N, L) or None
            "aux": aux_ds,              # (B, T, N, L, ...) of each entry
        }
    
    def forward(
        self, 
        rgb: Tensor, 
        prompt_text: List[str],
        mask: Tensor, 
        norm_xy: Tensor, 
        extrinsic: Tensor, 
        **aux_tensors: Tensor
    ):
        """
        Args:
            rgb (Tensor): (B, T, N, 3, H, W)
            prompt_text (List[str]): len = B
            mask (Tensor): (B, T, N, H, W)
            norm_xy (Tensor): (B, T, N, 2, H, W)
            extrinsic (Tensor): (B, T, N, 4, 4), ^ref_cam T
            aux_tensors (Tensor): each of (B, T, N, C, H, W)

        Returns:
            obs (Dict[str, Tensor]): image observations
                - x:    tensor of shape (B, To, Ncam, L, C) patch feature
                - gx:   tensor of shape (B, To, Ncam, C), projected global feature, not masked
                - pe:   tensor of shape (B, To, Ncam, L, 6), ray pe
                - mask: tensor of shape (B, To, Ncam, L) or None, patch mask
                - aux:  {aux_user_key: tensor of shape (B, To, Ncam, L, ...)}
        """
        B, T, N, C, H, W = rgb.shape

        clip_output = self.encode_token_features(rgb, prompt_text) # list of (B*T, L, C)
        x_ds_list = clip_output["x_percam_norm"]
        x_text = clip_output["x_text_norm"]
        text_mask = clip_output["text_mask"]

        gx = clip_output["gx"]
        gx_text = clip_output["gx_text"]
        x_ds = clip_output["x"]

        if mask is not None:
            mask = rearrange(mask, "b t n h w -> (b t n) () h w")
            mask_ds = self.encode_aux(mask)
            mask_ds = rearrange(mask_ds, "(b t n) l 1 -> (b t) n l", b=B, t=T, n=N)
            mask_ds_list = mask_ds.split(1, dim=1)
        else:
            mask_ds = None
            mask_ds_list = [None] * N

        # Process each camera's features
        scale = self.frozen.clip.clip_model.logit_scale.exp()        
        gx, gsim = self.visual_extraction.gtv_forward(gx, gx_text, scale)

        # Visualize text-image similarity scores on patches
        x_ds, x_sim = self.visual_extraction.gtlv_forward(x_ds, gx_text, scale) # (B*T*N, L, C)

        x_ds = self.proj_vision_patches(x_ds)
        gx = self.proj_vision_global(gx)
        x_text = self.proj_text_tokens(x_text)
        gx_text = self.proj_text_global(gx_text)

        x_ds = rearrange(x_ds, "(b t n) l c -> b t n l c", b=B, t=T, n=N)
        gx = rearrange(gx, "(b t n) c -> b t n c", b=B, t=T, n=N)
        
        mask_ds = rearrange(mask_ds, "(b t) n l -> b t n l", b=B, t=T, n=N)

        aux_tensors["norm_xy"] = norm_xy
        aux_tensors = {n: (rearrange(a, "b t n ... -> (b t n) ...") 
                           if a is not None else a) for n, a in aux_tensors.items()}
        aux_ds = {n: self.encode_aux(a) if a is not None else a 
                  for n, a in aux_tensors.items()}
        aux_ds = {n: (rearrange(a, "(b t n) ... -> b t n ...", b=B, t=T, n=N) 
                      if a is not None else a) for n, a in aux_ds.items()}
        norm_xy_ds = aux_ds.pop("norm_xy")
        # pe = norm_xy_ds
        ray_pe = plucker_ray_pe(norm_xy_ds, extrinsic)

        return {
            "x_text": x_text,           # (B, Ltext, C)
            "gx_text": gx_text,         # (B, C)
            "text_mask": text_mask,     # (B, Ltext)
            "x": x_ds,                  # (B, T, N, L, C)
            "gx": gx,                   # (B, T, N, C)
            "ray_pe": ray_pe,           # (B, T, N, L, 6)
            "norm_xy_pe": norm_xy_ds,   # (B, T, N, L, 2)
            "mask": mask_ds,            # (B, T, N, L) or None
            "aux": aux_ds,              # (B, T, N, L, ...) of each entry
        }


def plucker_ray_pe(norm_xy: Tensor, extrinsic: Tensor):
    """
    Args:
        norm_xy (Tensor): (..., L, 2)
        extrinsic (Tensor): (..., 4, 4), ^w_c T
    
    Returns:
        pe (Tensor): (..., L, 6)
    """
    homo_dir = F.pad(norm_xy, pad=(0, 1), mode="constant", value=1.0)
    direction = F.normalize(homo_dir, dim=-1)  # (..., L, 3)
    rotmat = extrinsic[..., :3, :3]  # (..., 3, 3)
    direction = direction @ rotmat.transpose(-1, -2)  # (..., L, 3), to world rays
    # (..., 3) -> (..., 1, 3) -> (..., L, 3)
    origin = extrinsic[..., :3, 3].unsqueeze(-2).expand_as(direction)  # (..., L, 3)
    pe = torch.cat([direction, torch.cross(direction, origin, dim=-1)], dim=-1)
    return pe