import argparse
import numpy as np
from .planner import TrajPlanner
from shm_transport import expose, run_simple_server, setup_log_level


class Service(TrajPlanner):
    @expose()
    def get_config(self):
        return vars(self.config)
    
    @expose()
    def reset(self):
        super().reset()
    
    @expose()
    def set_config(self, config):
        super().set_config(config)
    
    @expose()
    def set_prompt(self, prompt_text: str):
        super().set_prompt(prompt_text)

    @expose()
    def set_grasp_poses(self, grasp_poses: np.ndarray):
        super().set_grasp_poses(grasp_poses)

    @expose()
    def set_grasp_masks(self, grasp_masks: np.ndarray):
        super().set_grasp_masks(grasp_masks)

    @expose()
    def add_obs_frame(self, obs_frame):
        super().add_obs_frame(obs_frame)
    
    @expose()
    def get_grasp_action(self):
        return super().get_grasp_action()

    @expose()
    def get_place_action(self, sample_num: int=1):
        return super().get_place_action(sample_num)


def run_service():
    import logging

    setup_log_level(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--precontact_ckpt", type=str, default="", help="precontact ckpt path")
    parser.add_argument("--postcontact_ckpt", type=str, default="", help="postcontact ckpt path")
    parser.add_argument("--uri", type=str, default="control", help="alias name of the hosting object")
    parser.add_argument("--ns_host", type=str, default="localhost", help="naming server host")
    parser.add_argument("--ns_port", type=int, default=9091, help="naming server port")
    parser.add_argument("--host", type=str, default="localhost", help="daemon host")
    parser.add_argument("--port", type=int, default=0, help="daemon port")
    parser.add_argument("--ensemble", type=int, default=4, help="ensemble traj for smoothness")
    opt = parser.parse_args()

    assert len(opt.precontact_ckpt), "Please specify a valid precontact ckpt path."
    assert len(opt.postcontact_ckpt), "Please specify a valid postcontact ckpt path."

    service = Service(
        precontact_model_name="vla_small",
        postcontact_model_name="vla_base",
        precontact_ckpt_path=opt.precontact_ckpt,
        postcontact_ckpt_path=opt.postcontact_ckpt, 
        device="cuda:0",
        ensemble=opt.ensemble
    )

    run_simple_server(
        service,
        uri_name=opt.uri,
        daemon_port=opt.port,
        ns_port=opt.ns_port,
    )


if __name__ == "__main__":
    run_service()


