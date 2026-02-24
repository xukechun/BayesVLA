import torch
import numpy as np
from torch import Tensor

from ..action_expert import states2action, action2states
from ..dataset import get_train_loader, rbd, visualize_traj


train_loader = get_train_loader(4, 1)
for data in train_loader:
    print("-"*51)
    for k, v in data.items():
        print("k = {}, shape = {}, dtype = {}".format(k, v.shape, v.dtype))
    
    cur_wcT = data["obs_extrinsics"][:, -1, 0]  # (B, 4, 4)
    cur_weT = data["current_ee_pose"]  # (B, 4, 4)
    ee_states = data["gt_future_ee_states"]  # (B, Ta, 17)

    actions = states2action(cur_wcT, cur_weT, ee_states)
    with np.printoptions(2):
        print(actions[0].cpu().numpy())
    data["gt_future_ee_states"] = action2states(cur_wcT, cur_weT, actions)

    key = visualize_traj(rbd(data))
    if key == ord('q'):
        quit()


