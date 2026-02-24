import torch
import threading
import numpy as np
from torch import Tensor
from typing import Union

from models import vla
from train import Trainer
from .ensemble import TrajEnsembler
from data_utils.dataset_base import DataSampler, DataConfig, gen_norm_xy_map
from data_utils.datasets import DATA_CONFIGS

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def load_precontact_model(model_name, path, device):
    ckpt = torch.load(path, map_location=device)
    if model_name == "vla_small":
        model = vla.pre_vla_small().to(device)
    elif model_name == "vla_base":
        model = vla.pre_vla_base().to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    if "actor" in path:
        model.actor.load_state_dict(ckpt["weights"], strict=True)
    else:
        state_dict = model.state_dict()
        state_dict.update(ckpt["weights"])
        model.load_state_dict(state_dict) 
    model.eval()
    print(f"Loaded model from {path}")
    return model

def load_postcontact_model(model_name, path, device):
    ckpt = torch.load(path, map_location=device)
    if model_name == "vla_small":
        model = vla.post_vla_small().to(device)
    elif model_name == "vla_base":
        model = vla.post_vla_base().to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    if "actor" in path:
        model.actor.load_state_dict(ckpt["weights"], strict=True)
    else:
        model.load_state_dict(ckpt["weights"], strict=True)
    model.eval()

    print(f"Loaded model from {path}")
    return model

def get_grasp_trajectory(init_eef_pose, grasp_pose, steps_per_phase=50, tcp2ee=None):

    # !!! if x axis is positive, rotate the graps pose 180 degrees around z axis !!!
    if grasp_pose[0, 0] > 0:
        R_z_180 = np.array([
                        [-1,  0,  0, 0],
                        [ 0, -1,  0, 0],
                        [ 0,  0,  1, 0],
                        [ 0,  0,  0, 1]
                    ])
        grasp_pose = grasp_pose @ R_z_180

    def interpolate_pose(start_pos, start_quat, end_pos, end_quat, steps):
        poses = []
        r_start, r_end = R.from_quat(start_quat), R.from_quat(end_quat)
        key_times = [0, 1]  
        key_rots = R.concatenate([r_start, r_end])
        slerp = Slerp(key_times, key_rots)

        for i in range(steps):
            alpha = i / (steps - 1)
            pos = (1 - alpha) * start_pos + alpha * end_pos
            rot = slerp([alpha])[0].as_matrix()  
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = pos
            poses.append(T)
        return poses
    

    init_pos = init_eef_pose[:3, 3]
    init_quat = R.from_matrix(init_eef_pose[:3, :3]).as_quat()

    approach1_pos = np.array(grasp_pose[:3, 3]) + np.array([0, 0, 0.1])  
    approach1_quat = R.from_matrix(grasp_pose[:3, :3]).as_quat()
    
    approach2_pos = np.array(grasp_pose[:3, 3]) + np.array([0, 0, 0.01]) 
    approach2_quat = R.from_matrix(grasp_pose[:3, :3]).as_quat()

    to_grasp1 = interpolate_pose(init_pos, init_quat,
                                approach1_pos, approach1_quat, (steps_per_phase-10))
    to_grasp2 = interpolate_pose( approach1_pos, approach1_quat, 
                                 approach2_pos, approach2_quat, 10)
    wait = interpolate_pose( approach2_pos, approach2_quat, 
                                 approach2_pos, approach2_quat, 5)
    future_ee_poses = to_grasp1 + to_grasp2 + wait
    future_grippers = [1.0] * steps_per_phase + [0.0] * 5

    future_ee_poses = np.array(future_ee_poses) # [N, 4, 4]
    future_grippers = np.array(future_grippers) # [N]

    if tcp2ee is not None:
        future_ee_poses = future_ee_poses @ tcp2ee

    return future_ee_poses, future_grippers

class TrajPlanner(object):
    def __init__(
        self, 
        precontact_model_name: str,
        postcontact_model_name: str,
        precontact_ckpt_path: str, 
        postcontact_ckpt_path: str, 
        device: str = "cuda:0", 
        ensemble: int = -1
    ):
        self.vla_precontact = load_precontact_model(precontact_model_name, precontact_ckpt_path, device)
        self.vla_postcontact = load_postcontact_model(postcontact_model_name, postcontact_ckpt_path, device)

        self.ensemble = int(ensemble)
        self.ensembler_lock = threading.Lock()
        self.pos_ensembler = TrajEnsembler(int(ensemble))
        self.rot_ensembler = TrajEnsembler(int(ensemble))
        self.gripper_ensembler = TrajEnsembler(int(ensemble))

        self.obs_frames = []
        self.obs_lock = threading.Lock()
        
        self.device = device
        self.last_obs_data = None
        self.grasp_masks = None
    
    def reset(self):
        with self.ensembler_lock:
            self.pos_ensembler.reset()
            self.rot_ensembler.reset()
            self.gripper_ensembler.reset()
        with self.obs_lock:
            self.obs_frames.clear()
        return self
    
    def set_config(self, config: Union[str, dict, DataConfig]):
        if isinstance(config, str):
            config = DATA_CONFIGS[config]
        elif isinstance(config, dict):
            config = DataConfig(**config)
        elif isinstance(config, DataConfig):
            pass
        else:
            raise TypeError("Unsupported type of config: {}".format(type(config)))
        self.config = config

    def set_prompt(self, prompt_text: str):
        """
        Args:
            prompt_text (str):
        """
        self.prompt_text = prompt_text
        return self

    def set_grasp_poses(self, grasp_poses: np.ndarray):
        """
        Args:
            grasp_poses (np.array) of shape (L, 4, 4):
        """
        self.grasp_poses = torch.from_numpy(grasp_poses).to(self.device).unsqueeze(0)
        return self

    def set_grasp_masks(self, grasp_masks: np.ndarray):
        """
        Args:
            grasp_masks (np.array) of shape (L,):
        """
        self.grasp_masks = torch.from_numpy(grasp_masks).to(self.device).unsqueeze(0)
        return self

    def add_obs_frame(self, obs_frame: dict):
        """
        Args:
            obs_frame (dict) should contains necessary keys listed as followings.

            - CAM_NAME_0: 
                - model: pinhole
                - camera:
                    - width: int
                    - height: int
                    - K: np.ndarray of shape 9 (3x3), flattened
                - data:
                    - color: np.ndarray, shape=(H, W, C)
                    - seg: None | np.ndarray of shape (H, W) | isaacsim seg output
                    - wcT: np.ndarray of shape (4, 4), ^{world}_{cam} T
                    - timestep: float, current timestamp used for sync
            
            - CAM_NAME_1: similar as CAM_NAME_0
            - ee_pose: np.ndarray of shape (4, 4), ^{world}_{ee} T
            - openness: float, value from [0 (close), 1 (open)]
        """
        with self.obs_lock:
            self.obs_frames.append(obs_frame)
        return self
    
    def _make_data_for_infer(self, obs_frames: list, sample_num: int=1):
        """
        Args:
            obs_frames (list[dict]): list of obs_frame, 
                see annotations above
        """
        (
            obs_rgbs, obs_masks, obs_cam_poses, obs_ee_poses, 
            history_actions, future_actions, current_time, K
        ) = DataSampler.sample_framedict(
            obs_traj=obs_frames,
            camera_names=self.config.camera_names,
            num_history_cameras=self.config.num_history_cameras,
            num_history_states=self.config.num_history_states,
            num_future_states=self.config.num_future_states,
            latest=True,
            sample_camera_gaps=self.config.sample_camera_gaps,
            sample_state_gaps=self.config.sample_state_gaps,
            sample_dt=self.config.sample_dt,
            record_dt=self.config.record_dt,
            output_image_hw=self.config.output_image_hw,
            enable_seg=self.config.enable_seg,
        )

        T, ncam, C, H, W = obs_rgbs.shape
        norm_xys = gen_norm_xy_map(H, W, K).astype(np.float32)
        norm_xys = norm_xys[None].repeat(T, axis=0)  # (T, ncam, 2, H, W)

        obs_data = {
            "K": K,                                 # (ncam, 3, 3)
            "obs_rgbs": obs_rgbs,                   # (T, ncam, 3, H, W)
            "obs_masks": obs_masks,                 # (T, ncam, H, W)
            "prompt_text": [self.prompt_text],      # [str]
            "grasp_poses": self.grasp_poses,        # (N, 4, 4)
            "grasp_masks": self.grasp_masks,        # (N,)
            "obs_norm_xys": norm_xys,               # (To, ncam, 2, H, W)
            "obs_extrinsics": obs_cam_poses,        # (To, ncam, 4, 4)
            "current_ee_pose": obs_ee_poses[-1],    # (4, 4)
            "history_ee_states": history_actions,   # (nhist, 17)
            "gt_future_ee_states": future_actions,  # (Ta, 17)
            "timestamps": np.array(current_time),   # scalar
        }
        
        for k in obs_data:
            if isinstance(obs_data[k], np.ndarray):
                obs_data[k] = (torch.from_numpy(obs_data[k])
                                    .to(self.device)
                                    .unsqueeze(0)
                                    .repeat(sample_num, *([1] * obs_data[k].ndim)))
            if isinstance(obs_data[k], list):
                obs_data[k] = [obs_data[k][0]] * sample_num
        return obs_data
    
    def _run_postcontact_inference(self, obs_data):
        obs_data = Trainer.preprocess_data(
            obs_data, self.device
        )
        # with torch.inference_mode():
        with torch.no_grad():
            actions: Tensor = self.vla_postcontact(
                obs_rgbs=obs_data["obs_rgbs"], 
                obs_masks=obs_data.get("obs_masks", None),
                obs_norm_xys=obs_data["obs_norm_xys"],
                obs_extrinsics=obs_data["obs_extrinsics"],
                
                current_ee_pose=obs_data["current_ee_pose"],
                history_ee_states=obs_data["history_ee_states"],
                gt_future_ee_states=obs_data["gt_future_ee_states"], 
                inference=True,
                fp16=True,

                prompt_text=obs_data["prompt_text"],
            )  # (B, Ta, 17)
        return actions

    def _run_precontact_inference(self, obs_data):
        obs_data = Trainer.preprocess_data(
            obs_data, self.device
        )
        with torch.inference_mode():
            actions: Tensor = self.vla_precontact(
                obs_rgbs=obs_data["obs_rgbs"], 
                obs_masks=obs_data.get("obs_masks", None),
                obs_norm_xys=obs_data["obs_norm_xys"],
                obs_extrinsics=obs_data["obs_extrinsics"],
                obs_intrinsics=obs_data["K"],

                prompt_text=obs_data["prompt_text"],

                grasp_poses=obs_data["grasp_poses"],
                grasp_masks=obs_data["grasp_masks"],
                gt_select_grasp=-1,

                inference=True,
                fp16=True,
            ) # (4, 4)
        return actions

    def get_place_action(self, sample_num: int = 1):
        """
        Returns
        -------
            future_ee_poses (np.ndarray): shape (Ta, 4, 4), ^{world} _{ee} T
            future_grippers (np.ndarray): shape (Ta,), range [0 (close), 1 (open)]
            future_time (np.ndarray): shape (Ta,)
        """
        max_frames = max(
            self.config.num_history_cameras * self.config.sample_camera_gaps,
            self.config.num_history_states * self.config.sample_state_gaps
        )
        
        with self.obs_lock:
            while len(self.obs_frames) > max_frames:
                self.obs_frames.pop(0)
            obs_frames = self.obs_frames.copy()  # shallow copy
        
        if len(obs_frames) == 0:
            return None
        
        obs_data = self._make_data_for_infer(obs_frames, sample_num)
        actions = self._run_postcontact_inference(obs_data)
        
        self.last_obs_data = obs_data
        actions = actions.detach().cpu().numpy()  # (1, Ta, 17)
        B, Ta, _ = actions.shape
        
        ee_poses = np.reshape(actions[:, :, :16], (B, Ta, 4, 4))
        grippers = actions[:, :, -1]  # (B, Ta)
        
        # obs_data["timestamp"]: (B,)
        latest_time = obs_data["timestamps"][0].item()
        action_dt = self.config.sample_dt * self.config.sample_state_gaps
        future_time = (1 + np.arange(Ta)) * action_dt + latest_time
        # future_ee_poses = ee_poses[0]  # (Ta, 4, 4)
        # future_grippers = grippers[0]  # (Ta,)

        future_ee_poses = ee_poses  # (B, Ta, 4, 4)
        future_grippers = grippers  # (B, Ta)
        
        if self.ensemble != 0:
            with self.ensembler_lock:
                for i in range(B):
                    future_ee_poses[i, :, :3, 3] = self.pos_ensembler.update(
                        future_ee_poses[i, :, :3, 3], future_time, on_SO3=False
                    )
                    # future_ee_poses[:, :, :3, :3] = self.rot_ensembler.update(
                    #     future_ee_poses[:, :, :3, :3], future_time, on_SO3=True
                    # )
                    future_grippers[i, :] = self.gripper_ensembler.update(
                        future_grippers[i, :], future_time, on_SO3=False
                    )

        return future_ee_poses, future_grippers, future_time
    
    def get_grasp_action(self):
        """
        Returns
        -------
            future_grasp_pose (np.ndarray): shape (4, 4), ^{world} _{ee} T
        """
        max_frames = max(
            self.config.num_history_cameras * self.config.sample_camera_gaps,
            self.config.num_history_states * self.config.sample_state_gaps
        )
        
        with self.obs_lock:
            while len(self.obs_frames) > max_frames:
                self.obs_frames.pop(0)
            obs_frames = self.obs_frames.copy()  # shallow copy
        
        if len(obs_frames) == 0:
            return None
        
        obs_data = self._make_data_for_infer(obs_frames)
        actions = self._run_precontact_inference(obs_data)
        
        self.last_obs_data = obs_data
        actions = actions.detach().cpu().numpy()  # (4, 4)

        future_ee_poses, future_grippers = get_grasp_trajectory(
            init_eef_pose=obs_data["current_ee_pose"][0].detach().cpu().numpy(),
            grasp_pose=actions,
            steps_per_phase=30,
        )

        return actions, future_ee_poses, future_grippers