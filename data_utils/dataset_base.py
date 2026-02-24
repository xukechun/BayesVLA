import cv2
import h5py
import torch
import random
import numpy as np
from torch import Tensor
from einops import rearrange
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional
from torchvision.transforms import v2
from torch.utils.data import (
    Dataset, IterableDataset, 
    ConcatDataset, ChainDataset, 
    get_worker_info, DataLoader
)

from . import h5io, align


def infer_record_dt(t: np.ndarray, default: float = 1.0):
    if len(t) < 2:
        return default
    else:
        return (t[-1] - t[0]) / (len(t) - 1)


def find_closest_ind(train: np.ndarray, query: np.ndarray):
    bin_indices = np.digitize(query, train)
    sample_indices = []
    
    # query < train
    mask = (bin_indices == 0)
    if np.any(mask):
        sample_indices.append(np.array([0]*mask.sum(), dtype=bin_indices.dtype))
    
    # query in train
    mask = (bin_indices > 0) & (bin_indices < len(train))
    r_ind = bin_indices[mask]
    l_ind = r_ind - 1
    
    dist0 = np.abs(train[l_ind] - query[mask])
    dist1 = np.abs(train[r_ind] - query[mask])
    sample_indices.append(np.where(dist0 < dist1, l_ind, r_ind))

    # query > train
    mask = (bin_indices == len(train))
    if np.any(mask):
        sample_indices.append(np.array([len(train)-1]*mask.sum(), dtype=bin_indices.dtype))
    
    sample_indices = np.concatenate(sample_indices)
    return sample_indices


class DataSampler(object):
    @classmethod
    def pad2ncam(self, x: np.ndarray, num_camera: int, dim: int, zero_init: bool):
        pad_ncam = num_camera - x.shape[dim]
        if pad_ncam < 0:
            raise ValueError("[ERR ] current ncam = {}, which is larger than desired ncam = {}"
                             .format(x.shape[dim], num_camera))
        if pad_ncam > 0:
            pad = x.take([-1]*pad_ncam, axis=dim)
            if zero_init:
                pad[:] = 0
            x = np.concatenate([x, pad], axis=dim)
        return x

    @classmethod
    def sample_framedict(
        cls,
        obs_traj: List[Dict[str, np.ndarray]], 
        camera_names: List[str], 
        num_history_cameras: int, 
        num_history_states: int, 
        num_future_states: int, 
        latest: bool = False, 
        sample_state_gaps: int = 1,
        sample_camera_gaps: int = 1, 
        sample_dt: float = 1.0,
        record_dt: Optional[float] = None, 
        output_image_hw: Optional[Tuple[int, int]] = None, 
        enable_seg: bool = False, 
        pad2ncam: int = -1
    ):
        """
        obs_traj is a list of dict containing necessary keys listed as followings.
        - ee_pose: np.ndarray of shape (4, 4), ^{world}_{ee} T
        - gripper: float, value from [0 (close), 1 (open)]
        - timestamp: float, current timestamp
        - CAMERA_NAME_0: 
            - model: pinhole
            - camera:
                - width: int
                - height: int
                - K: np.ndarray of shape (3, 3) or (9,)
            - data:
                - color: np.ndarray, shape=(H, W, C)
                - seg: None | np.ndarray of shape (H, W) | isaacsim seg output
                - wcT: np.ndarray of shape (4, 4), ^{world}_{cam} T
        
        - CAMERA_NAME_1: similar as CAMERA_NAME_0
            - model: pinhole
            - ...
        """
        if not latest:
            last_obs_index = np.random.choice(len(obs_traj), 1)[0]
        else:
            last_obs_index = len(obs_traj) - 1

        all_timestamps = np.array([tau["timestamp"] for tau in obs_traj]).astype(np.float32)
        original_record_dt = infer_record_dt(all_timestamps, default=sample_dt)
        if record_dt is not None:
            # scale the original timestamp
            speedup = original_record_dt / record_dt
            all_timestamps = all_timestamps / speedup
        else:
            record_dt = original_record_dt
        
        # we don't interpolate the images, but find the image with closest timestamp
        current_time = all_timestamps[last_obs_index]
        prev_obs_sample_time = current_time + np.arange(-num_history_cameras+1, 1) * sample_camera_gaps * record_dt
        prev_obs_sample_ind = find_closest_ind(all_timestamps, prev_obs_sample_time)

        obs_across_cams: Dict[str, list] = {}
        for cam_name in camera_names:
            obs_cam = h5io.gather_frames(obs_traj, cam_name, prev_obs_sample_ind, compress=False)
            for k, v in obs_cam.items():
                if k not in obs_across_cams:
                    obs_across_cams[k] = []
                obs_across_cams[k].append(v)
        
        rgbs = np.stack(obs_across_cams["rgb"], axis=1)         # (T, ncam, 3, H, W)
        if rgbs.dtype == np.uint8:
            rgbs = rgbs.astype(np.float32) / 255.
        
        if ("mask" in obs_across_cams) and enable_seg:
            masks = np.stack(obs_across_cams["mask"], axis=1)   # (T, ncam, H, W)
        else:
            T, ncam, _, H, W = rgbs.shape
            masks = np.ones((T, ncam, H, W), dtype=bool)        # (T, ncam, H, W)
        
        cam_poses = np.stack(obs_across_cams["pose"], axis=1)   # (T, ncam, 4, 4)
        K = np.stack(obs_across_cams["K"], axis=0)              # (ncam, 3, 3)
        ee_poses = h5io.gather_ee_poses(obs_traj, prev_obs_sample_ind)  # (T, 4, 4)
        
        # interpolate the robot states
        all_ee_poses = np.stack([tau["ee_pose"] for tau in obs_traj], axis=0)  # (L, 4, 4)
        all_grippers = np.array([tau["gripper"] for tau in obs_traj])  # (L,)
        all_states = {"ee_pose": all_ee_poses, "gripper": all_grippers}
        interp_funcs = {"ee_pose": align.interp_SE3_sep, "gripper": align.interp_linear}
        
        history_time = current_time + np.arange(-num_history_states+1, 1) * sample_state_gaps * record_dt
        future_time = current_time + np.arange(1, num_future_states+1) * sample_state_gaps * record_dt
        
        history_queries = align.align_data(
            query_time=history_time,
            train_time=all_timestamps,
            train_data=all_states,
            interp_funcs=interp_funcs
        )
        history_states = h5io.compose_ee_gripper(
            ee_poses=history_queries["ee_pose"], 
            grippers=history_queries["gripper"]
        )

        future_queries = align.align_data(
            query_time=future_time,
            train_time=all_timestamps,
            train_data=all_states,
            interp_funcs=interp_funcs,
        )
        future_states = h5io.compose_ee_gripper(
            ee_poses=future_queries["ee_pose"],
            grippers=future_queries["gripper"]
        )

        # adjust output image size
        if output_image_hw is not None:
            H, W = output_image_hw
            rgbs, metadata = ImageProcessor.scale_to_fit(rgbs, H, W)
            masks, metadata = ImageProcessor.scale_to_fit(masks, H, W)
            K = ImageProcessor.tform_K_for_scale_to_fit(K, **metadata)

            rgbs, metadata = ImageProcessor.center_view(rgbs, H, W)
            masks, metadata = ImageProcessor.center_view(masks, H, W)
            K = ImageProcessor.tform_K_for_center_view(K, **metadata)

        # pad to n camera
        if pad2ncam > 0:
            rgbs = cls.pad2ncam(rgbs, pad2ncam, dim=1, zero_init=True)
            masks = cls.pad2ncam(masks, pad2ncam, dim=1, zero_init=True)
            cam_poses = cls.pad2ncam(cam_poses, pad2ncam, dim=1, zero_init=False)
            K = cls.pad2ncam(K, pad2ncam, dim=0, zero_init=False)

        return (
            rgbs,                               # (To, ncam, 3, H, W), ncam = 2 currently 
            masks,                              # (To, ncam, H, W)
            cam_poses.astype(np.float32),       # (To, ncam, 4, 4), ncam = 2 currently 
            ee_poses.astype(np.float32),        # (To, 4, 4)
            history_states.astype(np.float32),  # (nhist, 17)
            future_states.astype(np.float32),   # (Ta, 17)
            current_time,                       # scalar,
            K.astype(np.float32),               # (ncam, 3, 3)
        )

    @classmethod
    def sample_hdf5(
        cls,
        obs_traj: h5py.File, 
        camera_names: List[str], 
        num_history_cameras: int, 
        num_history_states: int, 
        num_future_states: int, 
        latest: bool = False, 
        sample_state_gaps: int = 1,
        sample_camera_gaps: int = 1,
        sample_dt: float = 1.0,
        record_dt: Optional[float] = None, 
        output_image_hw: Optional[Tuple[int, int]] = None, 
        enable_seg: bool = False, 
        pad2ncam: int = -1,
        video_root: Optional[str] = None,
        contact_phase: str = "pre",  # choices are ["pre", "post"]
        grasp_thres: float = 0.99,
        gripper_inverse: bool = False,
        complete_traj: bool = False
    ):
        """
        obs_traj is a tree-like data structure
        - ee_pose: np.ndarray of shape (T, 4, 4)
        - gripper: np.ndarray of shape (T,)
        - timestamp: np.ndarray of shape (T,)
        - CAMERA_NAME_0:
            - rgb: np.ndarray of shape (T, 3, H, W) or list of bytes (jpeg encoding)
            - pose: np.ndarray of shape (4, 4)
            - K: np.ndarray of shape (3, 3), camera intrinsic
        - CAMERA_NAME_1:
            - rgb: np.ndarray of shape (T, 3, H, W) or list of vlen
            - ...
        """
        obs_traj_len = obs_traj["ee_pose"].len()
        if not latest:
            if gripper_inverse:
                grasp_time = np.where(obs_traj["gripper"][:] > grasp_thres)[0]
            else:
                grasp_time = np.where(obs_traj["gripper"][:] < grasp_thres)[0]
            if contact_phase == "pre":
                # !!! Note current time should be the time before grasping !!!
                if grasp_time.shape[0] > 0:
                    before_grasp = np.arange(0, grasp_time[0])
                    last_obs_index = np.random.choice(before_grasp, 1)[0]
                else:
                    last_obs_index = np.random.choice(obs_traj_len, 1)[0]
            elif contact_phase == "post":
                if not complete_traj and grasp_time.shape[0] > 0:
                    after_grasp = np.arange(grasp_time[0], obs_traj_len)
                    last_obs_index = np.random.choice(after_grasp, 1)[0]
                else:
                    last_obs_index = np.random.choice(obs_traj_len, 1)[0]
        else:
            last_obs_index = obs_traj_len - 1
        
        all_timestamps = obs_traj["timestamp"][:]  # (L,)
        original_record_dt = infer_record_dt(all_timestamps, default=sample_dt)
        if record_dt is not None:
            # scale the original timestamp
            speedup = original_record_dt / record_dt
            all_timestamps = all_timestamps / speedup
        else:
            record_dt = original_record_dt

        # we don't interpolate the images, but find the image with closest timestamp
        current_time = all_timestamps[last_obs_index]
        prev_obs_sample_time = current_time + np.arange(-num_history_cameras+1, 1) * sample_camera_gaps * record_dt
        prev_obs_sample_ind = find_closest_ind(all_timestamps, prev_obs_sample_time)

        obs_across_cams: Dict[str, list] = {}
        for cam_name in camera_names:
            obs_cam = h5io.slice_encoded_frames(
                obs_traj[cam_name], 
                prev_obs_sample_ind,
                timestamp=all_timestamps,  # use original timestamp to iter video file
                video_root=video_root
            )
            for k, v in obs_cam.items():
                if k not in obs_across_cams:
                    obs_across_cams[k] = []
                obs_across_cams[k].append(v)
        
        rgbs = np.stack(obs_across_cams["rgb"], axis=1)         # (T, ncam, 3, H, W)
        if rgbs.dtype == np.uint8:
            rgbs = rgbs.astype(np.float32) / 255.
        
        if ("mask" in obs_across_cams) and enable_seg:
            masks = np.stack(obs_across_cams["mask"], axis=1)   # (T, ncam, H, W)
        else:
            T, ncam, _, H, W = rgbs.shape
            masks = np.ones((T, ncam, H, W), dtype=bool)        # (T, ncam, H, W)
        
        cam_poses = np.stack(obs_across_cams["pose"], axis=1)   # (T, ncam, 4, 4)
        K = np.stack(obs_across_cams["K"], axis=0)              # (ncam, 3, 3)
        ee_poses = h5io.slice_dset(obs_traj["ee_pose"], prev_obs_sample_ind)  # (T, 4, 4)

        # interpolate the robot states
        all_states = {
            "ee_pose": obs_traj["ee_pose"][:],  # (L, 4, 4)
            "gripper": obs_traj["gripper"][:],  # (L,)
        }
        # !!! IMPORTANT: for ALOHA data, use gripper desired instead of gripper !!!
        if "gripper_desired" in obs_traj.keys():
            all_states["gripper"] = obs_traj["gripper_desired"][:]

        interp_funcs = {
            "ee_pose": align.interp_SE3_sep, 
            "gripper": align.interp_linear
        }
        
        history_time = current_time + np.arange(-num_history_states+1, 1) * sample_state_gaps * record_dt
        future_time = current_time + np.arange(1, num_future_states+1) * sample_state_gaps * record_dt

        history_queries = align.align_data(
            query_time=history_time,
            train_time=all_timestamps,
            train_data=all_states,
            interp_funcs=interp_funcs
        )
        history_states = h5io.compose_ee_gripper(
            ee_poses=history_queries["ee_pose"], 
            grippers=history_queries["gripper"]
        )
        
        future_queries = align.align_data(
            query_time=future_time,
            train_time=all_timestamps,
            train_data=all_states,
            interp_funcs=interp_funcs,
        )
        future_states = h5io.compose_ee_gripper(
            ee_poses=future_queries["ee_pose"],
            grippers=future_queries["gripper"]
        )

        # adjust output image size
        if output_image_hw is not None:
            H, W = output_image_hw
            rgbs, metadata = ImageProcessor.scale_to_fit(rgbs, H, W)
            masks, metadata = ImageProcessor.scale_to_fit(masks, H, W)
            K = ImageProcessor.tform_K_for_scale_to_fit(K, **metadata)

            rgbs, metadata = ImageProcessor.center_view(rgbs, H, W)
            masks, metadata = ImageProcessor.center_view(masks, H, W)
            K = ImageProcessor.tform_K_for_center_view(K, **metadata)

        # pad to n camera
        if pad2ncam > 0:
            rgbs = cls.pad2ncam(rgbs, pad2ncam, dim=1, zero_init=True)
            masks = cls.pad2ncam(masks, pad2ncam, dim=1, zero_init=True)
            cam_poses = cls.pad2ncam(cam_poses, pad2ncam, dim=1, zero_init=False)
            K = cls.pad2ncam(K, pad2ncam, dim=0, zero_init=False)

        return (
            rgbs,                               # (To, ncam, 3, H, W), ncam = 2 currently 
            masks,                              # (To, ncam, H, W)
            cam_poses.astype(np.float32),       # (To, ncam, 4, 4), ncam = 2 currently 
            ee_poses.astype(np.float32),        # (To, 4, 4)
            history_states.astype(np.float32),  # (nhist, 17)
            future_states.astype(np.float32),   # (Ta, 17)
            current_time,                       # scalar,
            K.astype(np.float32),               # (ncam, 3, 3)
        )


def gen_norm_xy_map(H: int, W: int, K: np.ndarray):
    """
    Args:
        H (int): image height
        W (int): image width
        K (np.ndarray): (Ncam, 3, 3)
    
    Returns:
        norm_xy (np.ndarray): (Ncam, 2, H, W)
    """
    fx = K[:, 0, 0]; fy = K[:, 1, 1]; cx = K[:, 0, 2]; cy = K[:, 1, 2]  # (ncam,)
    XX, YY = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    grid = np.stack([XX, YY], axis=0)  # (2, H, W)
    cxy = np.stack([cx, cy], axis=-1)  # (ncam, 2)
    fxy = np.stack([fx, fy], axis=-1)  # (ncam, 2)
    norm_xy = (grid - cxy[:, :, None, None]) / fxy[:, :, None, None]  # (ncam, 2, H, W)
    return norm_xy


@dataclass
class DataConfig(object):
    ### total traj time of gt label is `sample_dt * num_future_states`
    sample_dt: float
    ### if None, declared from `timestamp` key from data, otherwise overwrite the data
    record_dt: Optional[float]
    ### image height and width, none means remain unchanged
    output_image_hw: Optional[Tuple[int, int]]

    ### used in training and real-world execution
    camera_names: Tuple[str]
    enable_seg: bool = False  # segment image patches if mask is available

    sample_state_gaps: int = 2
    sample_camera_gaps: int = 4

    num_history_cameras: int = 2
    num_history_states: int = 1
    num_future_states: int = 16  # future states as gt action

    video_root: Optional[str] = None  # for those rgb key as string, which means load video from video_root/rgb
    shuffle_cameras: bool = True

    grasp_thres: float = 0.99
    gripper_inverse: bool = False
    complete_traj: bool = False

    contact_phase: str = "pre"  # choices are ["pre", "post"]

class ImageProcessor(object):
    @classmethod
    def scale_to_fit(cls, x: np.ndarray, H, W):
        old_H, old_W = x.shape[-2:]
        scale_H = H / old_H
        scale_W = W / old_W
        scale = min(scale_H, scale_W)
        
        new_H = int(scale * old_H)
        new_W = int(scale * old_W)
        scale_H = new_H / old_H
        scale_W = new_W / old_W
        
        x: Tensor = torch.from_numpy(x)
        if x.is_floating_point():
            x_resize = v2.Resize((new_H, new_W), v2.InterpolationMode.BILINEAR)(x)
        else:
            x_resize = v2.Resize((new_H, new_W), v2.InterpolationMode.NEAREST)(x)
        x_resize: np.ndarray = x_resize.numpy()
        
        metadata = dict(
            old_H=old_H,
            old_W=old_W,
            new_H=new_H, 
            new_W=new_W,
        )
        
        return x_resize, metadata
    
    @classmethod
    def tform_K_for_scale_to_fit(
        cls, 
        K: np.ndarray, 
        old_H: int, old_W: int, new_H: int, new_W: int
    ):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        
        old_ox = (old_W - 1.0) * 0.5
        old_oy = (old_H - 1.0) * 0.5
        new_ox = (new_W - 1.0) * 0.5
        new_oy = (new_H - 1.0) * 0.5
        
        scale_x = new_W / old_W
        scale_y = new_H / old_H
        
        new_fx = fx * scale_x
        new_fy = fy * scale_y
        new_cx = (cx - old_ox) * scale_x + new_ox
        new_cy = (cy - old_oy) * scale_y + new_oy
        
        K_new = K.copy()
        K_new[..., 0, 0] = new_fx
        K_new[..., 1, 1] = new_fy
        K_new[..., 0, 2] = new_cx
        K_new[..., 1, 2] = new_cy
        return K_new
    
    @classmethod
    def center_view(cls, x: np.ndarray, H, W):
        old_H, old_W = x.shape[-2:]
        dx = (W - old_W) // 2
        dy = (H - old_H) // 2
        metadata = dict(dcx=dx, dcy=dy)
        
        if old_H == H and old_W == W:
            return x, metadata
        
        x: Tensor = torch.from_numpy(x)
        x_view: np.ndarray = v2.CenterCrop((H, W))(x).numpy()
        return x_view, metadata
    
    @classmethod
    def tform_K_for_center_view(cls, K: np.ndarray, dcx: int, dcy: int):
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        K_new = K.copy()
        K_new[..., 0, 2] = cx + dcx
        K_new[..., 1, 2] = cy + dcy
        return K_new


class H5DatasetMapBase(Dataset):

    config = DataConfig(
        sample_dt=1.0,
        record_dt=None,
        output_image_hw=None, 
        camera_names=(),
    )

    def __init__(
        self, 
        h5_filelist: List[str], 
        contact_phase: Optional[str] = None,
    ):
        self.h5_filelist = h5_filelist
        self.data_sampler = DataSampler()

        if isinstance(self.config.camera_names, str):
            # wrap to tuple
            self.config.camera_names = (self.config.camera_names,)

        if contact_phase is not None:
            assert contact_phase in ["pre", "post"], f"Invalid contact phase: {contact_phase}"
            self.config.contact_phase = contact_phase
        
        self.cam_num = len(self.config.camera_names)
        self.pad2ncam = self.cam_num


    def __len__(self):
        return len(self.h5_filelist)

    def __getitem__(self, i):
        h5_file = self.h5_filelist[i]
        with h5py.File(h5_file, "r") as h5:
            prompt_candidates = []
            for k, v in h5.attrs.items():
                if "prompt_text" in k:
                    v = v.strip()
                    if len(v):
                        prompt_candidates.append(v)
            
            prompt_text = random.sample(prompt_candidates, 1)[0] if len(prompt_candidates) else ""
            if len(prompt_text) == 0:
                prompt_text = "Do any possible actions"
            # prompt_text = h5.attrs.get("prompt_text", "")
            # only for pick-place
            if "move" in prompt_text:
                if "bowl" in prompt_text and "red bowl" not in prompt_text:
                    prompt_text = prompt_text.replace("bowl", "red bowl")
            

            if self.config.contact_phase == "pre":
                grasp_poses = h5.attrs.get("grasp_poses", None) # (N, 4, 4)
                if isinstance(grasp_poses, str):
                    import ast
                    grasp_poses = ast.literal_eval(grasp_poses)
                    grasp_poses = np.array(grasp_poses, dtype=np.float32)

                # !!! check grasp pose here !!! the last row of the grasp pose should be [0, 0, 0, 1]
                for pose in grasp_poses:
                    if pose[3, 3] != 1.0:
                        pose[3, 3] = 1.0
                # padding grasp poses to (30, 4, 4), and generate grasp masks
                if grasp_poses is not None:
                    grasp_masks = np.concatenate([np.ones((grasp_poses.shape[0])), np.zeros((30-grasp_poses.shape[0]))], axis=0).astype(bool) # for libero, here is 30
                    grasp_poses = np.concatenate([grasp_poses, np.eye(4)[None].repeat(30-grasp_poses.shape[0], axis=0)], axis=0)

                select_grasp_index = h5.attrs.get("select_grasp_index", None) # select_grasp_index before


            if self.config.shuffle_cameras:
                camera_names = list(self.config.camera_names).copy()
                random.shuffle(camera_names)
            else:
                camera_names = self.config.camera_names

            (
                obs_rgbs, obs_masks, obs_cam_poses, obs_ee_poses, 
                history_states, future_states, timestamps, K
            ) = self.data_sampler.sample_hdf5(
                obs_traj=h5, 
                camera_names=self.config.camera_names, 
                num_history_cameras=self.config.num_history_cameras, 
                num_history_states=self.config.num_history_states, 
                num_future_states=self.config.num_future_states,
                latest=False, 
                sample_state_gaps=self.config.sample_state_gaps, 
                sample_camera_gaps=self.config.sample_camera_gaps, 
                sample_dt=self.config.sample_dt,
                record_dt=self.config.record_dt, 
                output_image_hw=self.config.output_image_hw,
                enable_seg=self.config.enable_seg,
                pad2ncam=self.pad2ncam,
                video_root=self.config.video_root,
                contact_phase=self.config.contact_phase,
                grasp_thres=self.config.grasp_thres,
                gripper_inverse=self.config.gripper_inverse,
                complete_traj=self.config.complete_traj
            )
        
        T, ncam, C, H, W = obs_rgbs.shape
        norm_xys = gen_norm_xy_map(H, W, K).astype(np.float32)
        norm_xys = norm_xys[None].repeat(T, axis=0)  # (T, ncam, 2, H, W)
        
        out = {
            "K": K,                                 # (ncam, 3, 3)
            "obs_rgbs": obs_rgbs,                   # (To, ncam, 3, H, W)
            "obs_masks": obs_masks,                 # (To, ncam, H, W)
            "prompt_text": prompt_text,             # str
            "obs_norm_xys": norm_xys,               # (To, ncam, 2, H, W)
            "obs_extrinsics": obs_cam_poses,        # (To, ncam, 4, 4)
            "current_ee_pose": obs_ee_poses[-1],    # (4, 4)
            "history_ee_states": history_states,    # (nhist, 17)
            "gt_future_ee_states": future_states,   # (Ta, 17)
            "timestamps": timestamps,               # (To,)
        }

        if self.config.contact_phase == "pre":
            out["grasp_poses"] = grasp_poses                # (N, 4, 4)
            out["grasp_masks"] = grasp_masks                # (N)
            out["select_grasp_index"] = select_grasp_index  # int

        return out


class H5DatasetIterBase(H5DatasetMapBase, IterableDataset):
    
    def __init__(self, h5_filelist: List[str], contact_phase: Optional[str] = None):
        super().__init__(h5_filelist, contact_phase)
        self._shuffle_h5_list = False
    
    def __iter__(self):
        indices = np.arange(len(self.h5_filelist))
        if self._shuffle_h5_list:
            np.random.shuffle(indices)

        # worker_info = get_worker_info()
        # if (worker_info is not None) and (worker_info.num_workers > 1):
        #     # split workload
        #     splits = np.linspace(0, len(indices), num=worker_info.num_workers+1, endpoint=True)
        #     splits = splits.astype(np.int64).tolist()
        #     indices = indices[splits[worker_info.id]:splits[worker_info.id+1]].copy()
        
        for i in indices:
            yield self[int(i)]


def concat_datasets(
    datasets: List[Union[H5DatasetMapBase, H5DatasetIterBase]],
    shuffle: bool = None
):
    num_cams = [d.cam_num for d in datasets]
    pad2ncam = max(num_cams)
    for d in datasets:
        d.pad2ncam = pad2ncam
        print("[INFO] dataset {} uses {} cameras".format(d, d.cam_num))
    print("[INFO] Final padded camera num: {}".format(pad2ncam))
    
    if isinstance(datasets[0], H5DatasetIterBase):
        if shuffle:
            for d in datasets:
                d._shuffle_h5_list = True
        return ChainDataset(datasets)
    else:
        return ConcatDataset(datasets)


def get_dataloader(
    datasets: List[Union[H5DatasetMapBase, H5DatasetIterBase]],
    batch_size: int,
    num_workers: int = 0,
    shuffle: Optional[bool] = None, 
    persistent_workers: bool = False,
    sample_weights: Optional[list] = None,
    sample_multiplex: int = 1
):
    if sample_weights is not None:
        sample_weights = np.array(sample_weights)
        sample_weights = sample_weights / sample_weights.sum()
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights.tolist(), 
            num_samples=len(sample_weights) * sample_multiplex, 
            replacement=True
        )
    else:
        sampler = None

    if isinstance(datasets, (list, tuple)):
        datasets = concat_datasets(datasets, shuffle)
    elif isinstance(datasets, H5DatasetIterBase):
        if shuffle:
            datasets._shuffle_h5_list = True
    
    if isinstance(datasets, (H5DatasetIterBase, ChainDataset)):
        shuffle = None  # overwrite shuffle args
    
    dataloader = DataLoader(
        dataset=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        shuffle=shuffle, 
        sampler=sampler,
    )
    return dataloader


def generate_sample_weights(
    datasets: List[H5DatasetMapBase],
    dataset_weights: List[float]
):
    sample_weights = []
    for i, dataset in enumerate(datasets):
        sample_weights.append(
            np.array([dataset_weights[i] / len(dataset)] * len(dataset))
        )
    sample_weights = np.concatenate(sample_weights)
    sample_weights = sample_weights / sample_weights.sum()
    return sample_weights.tolist()


def rbd(d: Dict[str, Tensor]):
    """remove batch dimension"""
    return {k:v[0] if v is not None else v for k, v in d.items()}


def visualize_traj(data: Dict[str, Tensor], cam_idx: int = 0):

    # data["obs_rgbs"]: (To, ncam, C, H, W)
    rgb = rearrange(data["obs_rgbs"][-1, cam_idx], "c h w -> h w c")  # latest time, e2h cam
    # data["K"]: (ncam, 3, 3)
    K = data["K"][cam_idx]  # (3, 3)
    # data["obs_extrinsics"]: (To, ncam, 4, 4)
    wcT = data["obs_extrinsics"][-1, cam_idx]  # (4, 4)
    # data["gt_future_ee_states"]: (Ta, 4, 4)
    weTs = data["gt_future_ee_states"][:, :16].view(-1, 4, 4)  # (Ta, 4, 4)

    ceTs = torch.inverse(wcT)[None] @ weTs  # (Ta, 4, 4)
    cets = ceTs[:, :3, 3]  # (Ta, 3)

    proj_norm = cets[:, :2] / cets[:, 2:3]  # (Ta, 2)
    fxy = K[[0, 1], [0, 1]]; cxy = K[[0, 1], [2, 2]]
    proj_pix = proj_norm * fxy + cxy

    bgr = np.ascontiguousarray(rgb.cpu().numpy()[:, :, ::-1])
    proj_pix: np.ndarray = proj_pix.cpu().numpy()

    for x, y in proj_pix:
        cv2.circle(bgr, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imshow("traj", bgr)
    key = cv2.waitKey(0)
    return key

