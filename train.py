import os
import sys
import tyro
import torch
import argparse
from typing import Dict
from datetime import datetime
from torch import Tensor
import torch.optim as optim
from torchvision.transforms import v2
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models import vla
from configs import CONFIGS, TrainConfig
from data_utils.dataset_base import get_dataloader, generate_sample_weights, concat_datasets
from data_utils.dist_sampler import DistributedWeightedSampler, DistributedMultiplexSampler


def init_train_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-s", dest="save", type=str, default="", help="exp name to save")     # save
    parser.add_argument("-c", dest="conti", type=str, default="", help="exp name to resume")  # continue

    if "-h" in sys.argv or "--help" in sys.argv:
        print("=== argparse help ===")
        parser.print_help()
        print("\n=== tyro help ===")
        tyro.extras.get_parser(TrainConfig).print_help()
        sys.exit(0)

    args, remaining_argv = parser.parse_known_args()
    save: str = args.save
    conti: str = args.conti

    cfg = CONFIGS[args.config]
    cfg = tyro.cli(cfg.__class__, default=cfg, args=remaining_argv)
    return cfg, save, conti


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.sum = 0
        self.count = 0
    
    def append(self, val):
        self.sum += val
        self.count += 1
    
    def avg(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count


def reduce_metrics(metrics: dict, is_dist: bool = False):
    """Reduce metrics across all processes in distributed training"""
    if not is_dist:
        return metrics
    
    # Convert scalar metrics to tensors and reduce them
    reduced_metrics = {}
    for key, val in metrics.items():
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:  # scalar tensor
                tensor_val = val.detach().clone().to(torch.cuda.current_device())
                dist.all_reduce(tensor_val, op=dist.ReduceOp.SUM)
                reduced_metrics[key] = (tensor_val.item() / dist.get_world_size())
            else:
                reduced_metrics[key] = val  # non-scalar tensor, keep as is
        elif isinstance(val, (int, float)):
            tensor_val = torch.tensor(val, device=torch.cuda.current_device(), dtype=torch.float32)
            dist.all_reduce(tensor_val, op=dist.ReduceOp.SUM)
            reduced_metrics[key] = (tensor_val.item() / dist.get_world_size())
        else:
            reduced_metrics[key] = val  # other types, keep as is
    return reduced_metrics


def count_trainable(m: torch.nn.Module):
    count = 0
    for p in m.parameters():
        if p.requires_grad:
            count += p.numel()
    return count


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def get_data_loader_for_cfg(cfg: TrainConfig, is_dist: bool = False):
    if cfg.sample_multiplex > 1:
        assert cfg.dataset_weights is not None, \
            "sample_multiplex should be used together with dataset_weights"

    datasets = [D.inst(cfg.train_stage, cfg.contact_phase) for D in cfg.dataset_classes]
    
    if is_dist:
        # Use distributed samplers
        datasets_concat = concat_datasets(datasets, shuffle=(cfg.dataset_weights is None))
        
        if cfg.dataset_weights is None:
            sampler = DistributedMultiplexSampler(
                dataset=datasets_concat,
                shuffle=True,
                multiplex=cfg.sample_multiplex,
            )
        else:
            sample_weights = generate_sample_weights(datasets, cfg.dataset_weights)
            assert len(sample_weights) == len(datasets_concat)
            sampler = DistributedWeightedSampler(
                weights=sample_weights,
                num_samples=len(sample_weights) * cfg.sample_multiplex,
                replacement=True,
            )
        
        dataloader = DataLoader(
            dataset=datasets_concat,
            batch_size=cfg.bs,
            sampler=sampler,
            num_workers=cfg.workers,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=2,
        )
        return dataloader, sampler
    else:
        # Use regular dataloader (non-distributed)
        if cfg.dataset_weights is not None:
            sample_weights = generate_sample_weights(datasets, cfg.dataset_weights)
        else:
            sample_weights = None
        
        shuffle = True if (cfg.dataset_weights is None) else None
        dataloader = get_dataloader(
            datasets=datasets,
            batch_size=cfg.bs,
            num_workers=cfg.workers,
            shuffle=shuffle,
            persistent_workers=cfg.persistent_workers,
            sample_weights=sample_weights,
            sample_multiplex=cfg.sample_multiplex
        )
        return dataloader, None


class Trainer(object):
    LOG_DIR = "./logs/BayesVLA"
    CKPT_DIR = "./checkpoints/BayesVLA"

    def __init__(self):

        self.launch_time_str = datetime.now().strftime("%Y%m%d%H%M")
        self.cfg, save, conti = init_train_config()

        # Check if distributed training is enabled
        self.is_dist = is_distributed()
        if self.is_dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.model_device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.local_rank)
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.model_device = "cuda:0"
        
        if is_main_process():
            print("[INFO] Train config:")
            print(self.cfg)
            if self.is_dist:
                print(f"[INFO] Distributed training enabled: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")

        if self.cfg.contact_phase == "pre":
            self.model: vla.PreContactVLA = getattr(vla, "pre_vla_" + self.cfg.model.strip()
                                     )().to(self.model_device)
        elif self.cfg.contact_phase == "post":
            self.model: vla.PostContactVLA = getattr(vla, "post_vla_" + self.cfg.model.strip()
                                     )().to(self.model_device)
        else:
            raise ValueError(f"Invalid contact phase: {self.cfg.contact_phase}")
        
        self.train_loader, self.sampler = get_data_loader_for_cfg(self.cfg, self.is_dist)

        print("[INFO] Total {:.3f}M trainable parameters"
              .format(count_trainable(self.model) / 1e6))

        if self.cfg.contact_phase == "post" and self.cfg.train_stage == 1:
            learnable_layers = [
                "vlm.clip.proj",
                "vlm.ray_pe",
                "dp_head.traj_vl_attn",
                "dp_head.final_norm",
                "dp_head.act_head"
            ]
            for k, v in self.model.named_parameters():
                v.requires_grad_(False)
                for l in learnable_layers:
                    if l in k:
                        v.requires_grad_(True)
                        if is_main_process():
                            print("learnable layers in stage 1: ", k)
                        break

        params = [p for p in self.model.parameters() if p.requires_grad]

        # Wrap model with DDP if distributed
        if self.is_dist:
            # Set find_unused_parameters=True to handle cases where some parameters don't receive gradients
            # (e.g., when some layers are frozen in conti or conditional training)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        self.scaler = torch.cuda.amp.GradScaler(
            # "cuda", 
            enabled=self.cfg.fp16
        )

        # init
        self.save = False
        self.writer = None

        # Get the underlying model (unwrap DDP if needed)
        model_for_loading = self.model.module if self.is_dist else self.model

        if conti:
            self.save = conti
            ckpt = torch.load(os.path.join(self.CKPT_DIR, conti, "ckpt_best.pt"), 
                              map_location=self.model_device)
            model_for_loading.load_state_dict(ckpt["weights"])
            self.last_ep = ckpt["last_ep"]
            if is_main_process():
                print("Loaded model from {}".format(os.path.join(self.CKPT_DIR, conti, "ckpt_best.pt")))
            self.current_iters = ckpt["current_iters"]
        elif self.cfg.pretrained_ckpt:
            ckpt = torch.load(os.path.join(self.CKPT_DIR, self.cfg.pretrained_ckpt, "ckpt_best.pt"), 
                            map_location=self.model_device)
            if self.cfg.load_from_va:
                ######################## update matching parameters ########################
                state_dict = model_for_loading.state_dict()
                # state_dict.update({k: v for k, v in ckpt["weights"].items() if k in state_dict and "act_head" not in k})
                state_dict.update({k: v for k, v in ckpt["weights"].items() if k in state_dict})
                model_for_loading.load_state_dict(state_dict) 
            else:
                model_for_loading.load_state_dict(ckpt["weights"])
            if is_main_process():
                print("Loaded model from {}".format(os.path.join(self.CKPT_DIR, self.cfg.pretrained_ckpt, "ckpt_best.pt")))
            self.current_iters = 0
            self.last_ep = -1    
        else:
            self.current_iters = 0
            self.last_ep = -1

        # if save path is explicitly specified, then override
        if save:
            self.save = save

        self.optimizer = optim.AdamW(params, self.cfg.max_lr, 
                                     weight_decay=self.cfg.wd)

        if conti:
            if is_main_process():
                print("[INFO] resume training from iter: {}".format(ckpt["current_iters"]))
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scaler.load_state_dict(ckpt["scaler"])

        # modify lr of optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.cfg.max_lr

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(200e4),],
            gamma=0.1, last_epoch=self.last_ep
        )

        if conti:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        if is_main_process():
            model_for_counting = self.model.module if self.is_dist else self.model
            print("[INFO] Total {:.3f}M trainable parameters"
                  .format(count_trainable(model_for_counting) / 1e6))

        self.aug = v2.Identity()  # no aug
        self.best_score = None
        self.larger_better = False
        self._is_first_save = True

    @classmethod
    def min_center_crop(cls, a: Tensor):
        H, W = a.shape[-2:]
        A = min(H, W)
        crop = v2.CenterCrop((A, A))
        return crop(a)
    
    @classmethod
    def pad2square(cls, a: Tensor, pad_edge: bool = False) -> Tensor:
        H, W = a.shape[-2:]
        
        if H == W:
            return a
        
        if H > W:
            pad_l = pad_r = 0
            pad_t = (H - W) // 2
            pad_b = H - W - pad_t
            pad = v2.Pad((pad_l, pad_t, pad_r, pad_b), fill=0, 
                         padding_mode="edge" if pad_edge else "constant")
            return pad(a)
        
        if H < W:
            pad_t = pad_b = 0
            pad_l = (W - H) // 2
            pad_r = W - H - pad_l
            pad = v2.Pad((pad_l, pad_t, pad_r, pad_b), fill=0, 
                         padding_mode="edge" if pad_edge else "constant")
            return pad(a)
    
    @classmethod
    def preprocess_data(
        cls, 
        data: Dict[str, Tensor], 
        device,
    ):
        for k in data:
            if isinstance(data[k], Tensor):
                data[k] = data[k].to(device, non_blocking=True)

        for k in ["obs_rgbs", "obs_masks", "prompt_rgb", "prompt_mask"]:
            if (k in data) and (data[k] is not None):
                data[k] = cls.pad2square(data[k], pad_edge=False)
        
        for k in ["obs_norm_xys"]:
            if (k in data) and (data[k] is not None):
                data[k] = cls.pad2square(data[k], pad_edge=True)
        
        return data

    def calculate_metrics(self, data: Dict[str, Tensor]):
        data = self.preprocess_data(data, self.model_device)
        data["obs_rgbs"] = self.aug(data["obs_rgbs"])

        if self.cfg.contact_phase == "pre":
            total_loss, metrics = self.model(
                obs_rgbs=data["obs_rgbs"], 
                obs_masks=data.get("obs_masks", None),
                obs_norm_xys=data["obs_norm_xys"],
                obs_extrinsics=data["obs_extrinsics"],
                obs_intrinsics=data["K"],

                prompt_text=data["prompt_text"],

                grasp_poses=data["grasp_poses"],
                gt_select_grasp=data["select_grasp_index"],

                inference=False,
                fp16=self.scaler.is_enabled(),
            )
        elif self.cfg.contact_phase == "post":
            total_loss, metrics = self.model(
                obs_rgbs=data["obs_rgbs"], 
                obs_masks=data.get("obs_masks", None),
                obs_norm_xys=data["obs_norm_xys"],
                obs_extrinsics=data["obs_extrinsics"],

                current_ee_pose=data["current_ee_pose"],
                history_ee_states=data["history_ee_states"],
                gt_future_ee_states=data["gt_future_ee_states"], 
                inference=False,
                fp16=self.scaler.is_enabled(),
                
                prompt_text=data["prompt_text"],
            )

        return total_loss, metrics

    def log_metrics(self, metrics: dict):
        # Only log on main process
        if not is_main_process():
            return
        
        if self.save:
            if self.writer is None:
                log_dir = os.path.join(self.LOG_DIR, self.save)
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
            self.writer.add_scalar(
                "lr", self.scheduler.get_last_lr()[0], self.current_iters)
            for key, val in metrics.items():
                self.writer.add_scalar(key, val, self.current_iters)

    def save_model(self, fname: str, best_score: float, latest_score: float):
        # Only save on main process
        if not is_main_process():
            return

        if self.save and self._is_first_save:
            cfg_save_path = os.path.join(self.CKPT_DIR, self.save, "{}.json".format(self.launch_time_str))
            self.cfg.dump(cfg_save_path)
            self._is_first_save = False

        if self.save:
            ckpt_dir = os.path.join(self.CKPT_DIR, self.save)
            os.makedirs(ckpt_dir, exist_ok=True)

            # Get underlying model (unwrap DDP if needed)
            model_for_saving = self.model.module if self.is_dist else self.model

            to_save = {
                "weights": model_for_saving.state_dict(),
                "current_iters": self.current_iters,
                "last_ep": self.last_ep, 
                "lr": self.scheduler.get_last_lr()[0], 
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_score": best_score,
                "latest_score": latest_score
            }
            torch.save(to_save, os.path.join(ckpt_dir, fname))
            print("[INFO] Save to {}".format(os.path.join(ckpt_dir, fname)))

    def fitting(self):
        averages = {}
        self.model.train()

        while self.current_iters <= self.cfg.max_iterations:
            # Set epoch for distributed sampler
            if self.is_dist and self.sampler is not None:
                self.sampler.set_epoch(self.last_ep + 1)

            for data in self.train_loader:
                B = data["obs_rgbs"].shape[0]
                if B == 1:
                    print("[INFO] get data batch size = 1, skip.")
                    continue

                self.current_iters += 1
                self.optimizer.zero_grad()
                loss, metrics = self.calculate_metrics(data)
                # print("loss", loss, type(loss))
                # print("metrics", metrics)

                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main_process():
                        print("[INFO] NaN or Inf occured in loss, skip")
                    self.current_iters -= 1
                    continue

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if self.cfg.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                self.scheduler.step()

                print_strings = []
                for key, val in metrics.items():
                    if key not in averages:
                        averages[key] = AverageMeter()
                    averages[key].append(val)
                
                # Reduce metrics across processes for logging
                avg_metrics = {k: v.avg() for k, v in averages.items()}
                reduced_metrics = reduce_metrics(avg_metrics, self.is_dist)
                
                if is_main_process():
                    for key, val in reduced_metrics.items():
                        print_strings.append("{} = {:.3e}".format(key, val))
                    print("[INFO] {}/{} | {} | lr = {:.3e}".format(
                        self.current_iters, self.cfg.max_iterations, " | ".join(print_strings),
                        self.scheduler.get_last_lr()[0]))

                ### save ckpt and log
                if self.current_iters % self.cfg.save_latest_interval == 0:
                    avg_metrics = {k: v.avg() for k, v in averages.items()}
                    # Reduce metrics to get accurate total_loss across all processes
                    reduced_metrics = reduce_metrics(avg_metrics, self.is_dist)
                    latest_score = reduced_metrics.get("total_loss", avg_metrics.get("total_loss", 0.0))
                    
                    if (
                        (self.best_score is None) or
                        (self.larger_better and (latest_score > self.best_score)) or 
                        (not self.larger_better and (latest_score < self.best_score)) 
                    ):
                        self.best_score = latest_score
                        save_best = True
                    else:
                        save_best = False

                    self.save_model("ckpt_latest.pt", self.best_score, latest_score)
                    if save_best:
                        self.save_model("ckpt_best.pt", self.best_score, latest_score)
                
                if (self.current_iters % self.cfg.save_interval == 0) and (self.cfg.save_interval > 0):
                    avg_metrics = {k: v.avg() for k, v in averages.items()}
                    reduced_metrics = reduce_metrics(avg_metrics, self.is_dist)
                    latest_score = reduced_metrics.get("total_loss", avg_metrics.get("total_loss", 0.0))
                    self.save_model("ckpt_{:0>7d}.pt".format(self.current_iters), 
                                    self.best_score, latest_score)
                
                if self.current_iters % self.cfg.log_interval == 0:
                    avg_metrics = {"train/"+k: v.avg() for k, v in averages.items()}
                    reduced_metrics = reduce_metrics(avg_metrics, self.is_dist)
                    self.log_metrics(reduced_metrics)
                    for key in averages.keys():
                        averages[key].reset()

                if self.current_iters > self.cfg.max_iterations:
                    break

            self.last_ep += 1


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize the process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        return True
    else:
        return False


if __name__ == "__main__":

    # Opt-in interactive debugging to avoid blocking normal runs.
    if os.getenv("DEBUG_EMBED", "0") == "1":
        try:
            from IPython import embed
        except ImportError:
            print("[WARN] DEBUG_EMBED=1 but IPython is not installed.")
        else:
            embed()
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Initialize distributed training if environment variables are set
    is_distributed_training = setup_distributed()
    
    try:
        trainer = Trainer()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            trainer.fitting()
    finally:
        # Clean up distributed training
        if is_distributed_training and dist.is_initialized():
            dist.destroy_process_group()

