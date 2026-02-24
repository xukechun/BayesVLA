import os
import sys
import glob
import inspect
import traceback
import numpy as np
from typing import Dict, Union
from .data_loc import DSET, LOC
from .dataset_base import DataConfig, H5DatasetMapBase


def fwd_ee_origin(
    out: Dict[str, np.ndarray], 
    fwd_axis: int, 
    distance: float
):
    for key in ["current_ee_pose"]:
        out[key][..., :3, 3] += out[key][..., :3, fwd_axis] * distance
    for key in ["history_ee_states", "gt_future_ee_states"]:
        ee = out[key]  # (B, T, Nee, 17)
        Ta, Nee, _ = ee.shape
        pose = ee[..., :16].reshape(Ta, Nee, 4, 4)
        pose[..., :3, 3] += pose[..., :3, fwd_axis] * distance
        out[key][..., :16] = pose.reshape(Ta, Nee, 16)
    return out

def get_loc(cls: Union[str, H5DatasetMapBase]):
    # get file location for dataset cls
    if isinstance(cls, str):
        cls_name = cls
    else:
        cls_name = cls.__name__

    path = LOC[getattr(DSET, cls_name)]
    assert path is not None, "No path defined for the dataset `{}` in this machine".format(cls_name)
    return path


class PickPlaceCan(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        camera_names=("e2h_cam", "eih_cam"),
        shuffle_cameras=False,
        grasp_thres=0.99,
        # complete_traj=True
    )

    @classmethod
    def inst(cls, train_stage: int = 0, contact_phase: str = "pre"):
        h5_files = glob.glob(get_loc(cls))
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        if train_stage == 0:
            h5_files = h5_files[:int(len(h5_files) * 0.5)]
        # else:
        #     h5_files = h5_files[int(len(h5_files) * 0.5):]
        return cls(h5_files, contact_phase)


class OpenDrawer(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        camera_names=("agent_camera",),
        shuffle_cameras=False,
        grasp_thres=0.99,
        # complete_traj=True
    )

    @classmethod
    def inst(cls, train_stage: int = 0, contact_phase: str = "pre"):
        h5_files = glob.glob(get_loc(cls))
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        if train_stage == 0:
            h5_files = h5_files[:int(len(h5_files) * 0.5)]
        # else:
        #     h5_files = h5_files[int(len(h5_files) * 0.5):]
        return cls(h5_files, contact_phase)


class OpenOven(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        camera_names=("agent_camera",),
        shuffle_cameras=False,
        grasp_thres=0.99,
        # complete_traj=True
    )

    @classmethod
    def inst(cls, train_stage: int = 0, contact_phase: str = "pre"):
        h5_files = glob.glob(get_loc(cls))
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        if train_stage == 0:
            h5_files = h5_files[:int(len(h5_files) * 0.5)]
        # else:
        #     h5_files = h5_files[int(len(h5_files) * 0.5):]
        return cls(h5_files, contact_phase)


class LiberoObject(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        camera_names=("agentview", "eye_in_hand"),
        shuffle_cameras=False,
        grasp_thres=0.9,
        # complete_traj=True
    )

    @classmethod
    def inst(cls, train_stage: int = 1, contact_phase: str = "pre"):
        h5_files = [glob.glob(LOC[DSET.LiberoObject], recursive=True)]
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files, contact_phase)

    def modify_prompt(self, lang: str):
        # remove something like "LIVING ROOM SCENE6" in libero10 and libero90
        index = lang.find("SCENE")
        if index >= 0:
            lang = lang[index:]
            lang = " ".join(lang.split(" ")[1:])
        return lang

    def __getitem__(self, i):
        out = super().__getitem__(i)
        out["prompt_text"] = self.modify_prompt(out["prompt_text"])
        return out


class Droid(H5DatasetMapBase):
    data_root = get_loc(__qualname__)

    config = DataConfig(
        record_dt=1.0/15,
        sample_dt=1.0/15,
        output_image_hw=(256, 256),
        # output_image_hw=(180, 320),
        camera_names=("exterior_2_left", "exterior_1_left", "wrist_left"),
        sample_state_gaps=2,
        video_root=data_root,
        grasp_thres=0.99
    )

    @classmethod
    def filter_files(cls, filelist):
        filtered = []
        for f in filelist:
            # remove .h5 and episode_
            episode_index = int(os.path.split(f)[-1][:-3].replace("episode_", ""))
            if episode_index in [11907, 14419, 24440, 64837, 64871]:
                print("[INFO] in Droid dataset, remove file: {}".format(f))
            else:
                filtered.append(f)
        return filtered

    @classmethod
    def inst(cls, train_stage: int = 0, contact_phase: str = "pre"):
        h5_files = glob.glob(os.path.join(cls.data_root, "data/*/*.h5"), recursive=True)
        h5_files = cls.filter_files(h5_files)
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files, contact_phase)
    
    def sample_from_hdf5(self, h5, latest = False, debug_sample_index = None):
        out = super().sample_from_hdf5(h5, latest, debug_sample_index)
        out = fwd_ee_origin(out, fwd_axis=2, distance=0.15)
        return out
    
    def __getitem__(self, i):
        try:
            out = super().__getitem__(i)
        except Exception as e:
            # This occasionally fails when reading video files, I don't know why
            traceback.print_exc()
            with open("error_filelist.txt", "a") as fp:
                fp.write("Error in file reading: ")
                fp.write(self.h5_filelist[i] + "\n")
            print("[INFO] Retry another index")
            out = super().__getitem__((i+1)%len(self))
        
        return out
        
                
def get_subclasses(base_class):
    current_module = sys.modules[__name__]
    subclasses = []
    for name, obj in inspect.getmembers(current_module, inspect.isclass):
        if issubclass(obj, base_class) and obj is not base_class:
            subclasses.append(obj)
    return subclasses


DATA_CONFIGS: Dict[str, DataConfig] = {
    c.__name__: c.config for c in get_subclasses(H5DatasetMapBase)
}


if __name__ == "__main__":

    import torch
    from torch.utils.data import DataLoader
    dataset = LiberoObject.inst()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for data in dataloader:
        print("-"*61)
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print("  - {}: {}".format(k, v.shape))
            else:
                print("  - {}: {}".format(k, v))
        
        input("[INFO] Press Enter to continue: ")

