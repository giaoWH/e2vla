import os
import cv2
import copy
import glob
import torch
import threading
import numpy as np
from torch import Tensor
from typing import Any, Dict, Optional, Union

from models import vla
from models.action_expert import states2action
from .ensemble import TrajEnsembler
from data_utils import align
from data_utils.dataset_base import DataSampler, DataConfig, gen_norm_xy_map, rbd
from train_utils.ema_impl import ExponentialMovingAverage
from data_utils.datasets import DATA_CONFIGS
from .draw_traj import visualize_traj
from configs import TrainConfig


def parse_config(ckpt_dir: str):
    config_files = glob.glob(os.path.join(ckpt_dir, "*.json"))
    config_files.sort()
    
    assert len(config_files), "No config files found in {}".format(ckpt_dir)
    config_file = config_files[-1]
    print("[INFO] Use config file {}".format(config_file))
    
    cfg = TrainConfig.load(config_file)
    data_config = cfg.dataset_classes[0].config
    model_name = cfg.model
    
    data_config.shuffle_cameras = False  # overwrite
    print("[INFO] model = {}".format(model_name))
    print("[INFO] data config = {}".format(data_config))
    
    return model_name, data_config


def load_model(path, device, use_ema: bool = False):
    model_name, data_config = parse_config(os.path.dirname(path))
    
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model: vla.VLA = getattr(vla, "vla_{}".format(model_name))().to(device)
    model.actor.load_state_dict(ckpt["weights"])
    print("[INFO] Load weights from iter: {}".format(ckpt["current_iters"]))

    if use_ema:
        param = [p for p in model.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(param, 1.0)
        ema.load_state_dict(ckpt["ema"])
        ema.to(device)
        ema.copy_to(param)
        print("[INFO] EMA weights loaded")

    model.requires_grad_(False)
    model.eval()
    return model, data_config


class TrajPlanner(object):
    def __init__(
        self, 
        ckpt_path: str, 
        device: str = "cuda:0", 
        ensemble: int = -1,
        use_ema: bool = False
    ):
        self.model, self.config = load_model(ckpt_path, device, use_ema)

        self.ensemble = int(ensemble)
        self.ensembler_lock = threading.Lock()
        self.pos_ensembler = TrajEnsembler(int(ensemble))
        self.rot_ensembler = TrajEnsembler(int(ensemble))
        self.gripper_ensembler = TrajEnsembler(int(ensemble))

        self.obs_frames = []
        self.obs_lock = threading.Lock()
        
        self.device = device
        self.last_obs_data = None
        self.last_rtc_debug = None
    
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
        
        config: DataConfig = copy.deepcopy(config)
        config.shuffle_cameras = False  # do not shuffle cameras when inference
        self.config = config
    
    def set_prompt(self, prompt_text: str):
        """
        Args:
            prompt_text (str):
        """
        self.prompt_text = prompt_text
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
            - gripper: float, value from [0 (close), 1 (open)]
            - timestamp: float
        """
        max_frames = max(
            self.config.num_history_cameras * self.config.sample_camera_gaps,
            self.config.num_history_states * self.config.sample_state_gaps
        )
        
        def max_time(a: float, b: float):
            if a is None: return b
            elif b is None: return a
            else: return max(a, b)
        
        if (self.config.record_dt is None) and (self.config.sample_dt is None):
            with self.obs_lock:
                self.obs_frames.append(obs_frame)
                while len(self.obs_frames) > max_frames:
                    self.obs_frames.pop(0)
        else:
            time_interval = max_frames * max_time(self.config.record_dt, self.config.sample_dt)
            latest_time = obs_frame["timestamp"]
            earliest_time_thersh = latest_time - time_interval
            with self.obs_lock:
                self.obs_frames.append(obs_frame)
                pop_counts = 0
                for frame in self.obs_frames[1:]:
                    if frame["timestamp"] < earliest_time_thersh:
                        pop_counts += 1
                    else:
                        break                
                if pop_counts > 0:
                    self.obs_frames = self.obs_frames[pop_counts:]
            
        return self
    
    def _make_data_for_infer(self, obs_frames: list):
        """
        Args:
            obs_frames (list[dict]): list of obs_frame, 
                see annotations above
        """
        (
            obs_rgbs, obs_masks, obs_cam_poses, obs_ee_poses, 
            history_actions, future_actions, current_time, K, valid_ee_mask
        ) = DataSampler.sample_framedict(
            obs_traj=obs_frames,
            ee_indices=self.config.ee_indices,
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
            "obs_norm_xys": norm_xys,               # (To, ncam, 2, H, W)
            "obs_extrinsics": obs_cam_poses,        # (To, ncam, 4, 4)
            "current_ee_pose": obs_ee_poses[-1],    # (nee, 4, 4)
            "history_ee_states": history_actions,   # (nhist, nee, 17)
            "gt_future_ee_states": future_actions,  # (Ta, nee, 17)
            "timestamps": np.array(current_time),   # scalar
            "valid_ee_mask": valid_ee_mask,         # (nee,)
        }
        
        for k in obs_data:
            if isinstance(obs_data[k], np.ndarray):
                obs_data[k] = (torch.from_numpy(obs_data[k])
                                    .to(self.device)
                                    .unsqueeze(0))
        return obs_data

    def _rtc_enabled(self, rtc_context: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(rtc_context, dict):
            return False
        return bool(rtc_context.get("rtc_enabled", False))

    def _config_ee_indices(self):
        ee_indices = self.config.ee_indices
        if isinstance(ee_indices, int):
            return (ee_indices,)
        return tuple(ee_indices)

    def _make_model_future_time(self, obs_data: Dict[str, Any]) -> np.ndarray:
        Ta = obs_data["gt_future_ee_states"].shape[1]
        latest_time = obs_data["timestamps"][0].item()
        action_dt = self.config.sample_dt * self.config.sample_state_gaps
        return (1 + np.arange(Ta, dtype=np.float64)) * action_dt + latest_time

    def _normalize_rtc_traj(self, rtc_context: Dict[str, Any]):
        try:
            old_ee_poses = np.asarray(rtc_context["old_future_ee_poses"])
            old_grippers = np.asarray(rtc_context["old_future_grippers"])
            old_time = np.asarray(rtc_context["old_future_time"], dtype=np.float64)
        except KeyError as exc:
            print("[RTC] missing rtc_context field: {}".format(exc))
            return None

        if old_ee_poses.ndim == 5:
            if old_ee_poses.shape[0] != 1:
                print("[RTC] batched old_future_ee_poses is unsupported: {}".format(old_ee_poses.shape))
                return None
            old_ee_poses = old_ee_poses[0]
        if old_grippers.ndim == 3:
            if old_grippers.shape[0] != 1:
                print("[RTC] batched old_future_grippers is unsupported: {}".format(old_grippers.shape))
                return None
            old_grippers = old_grippers[0]

        if old_ee_poses.ndim == 3:
            old_ee_poses = old_ee_poses[:, None]
        if old_grippers.ndim == 1:
            old_grippers = old_grippers[:, None]

        if old_ee_poses.ndim != 4 or old_ee_poses.shape[-2:] != (4, 4):
            print("[RTC] invalid old_future_ee_poses shape: {}".format(old_ee_poses.shape))
            return None
        if old_grippers.ndim != 2:
            print("[RTC] invalid old_future_grippers shape: {}".format(old_grippers.shape))
            return None

        n = min(len(old_time), len(old_ee_poses), len(old_grippers))
        old_time = old_time[:n]
        old_ee_poses = old_ee_poses[:n]
        old_grippers = old_grippers[:n]

        finite = np.isfinite(old_time)
        if not finite.all():
            old_time = old_time[finite]
            old_ee_poses = old_ee_poses[finite]
            old_grippers = old_grippers[finite]

        if len(old_time) < 2:
            print("[RTC] need at least 2 old trajectory points, got {}".format(len(old_time)))
            return None

        order = np.argsort(old_time)
        old_time = old_time[order]
        old_ee_poses = old_ee_poses[order]
        old_grippers = old_grippers[order]

        keep = np.concatenate([[True], np.diff(old_time) > 1e-6])
        old_time = old_time[keep]
        old_ee_poses = old_ee_poses[keep]
        old_grippers = old_grippers[keep]

        if len(old_time) < 2:
            print("[RTC] old trajectory timestamps collapse after dedup")
            return None

        ee_indices = self._config_ee_indices()
        if old_ee_poses.shape[1] > max(ee_indices):
            old_ee_poses = old_ee_poses[:, ee_indices]
            old_grippers = old_grippers[:, ee_indices]
        elif old_ee_poses.shape[1] != len(ee_indices):
            print(
                "[RTC] old trajectory EE shape {} cannot match ee_indices {}".format(
                    old_ee_poses.shape[1],
                    ee_indices,
                )
            )
            return None

        return old_ee_poses, old_grippers, old_time

    def _build_model_rtc_context(
        self,
        obs_data: Dict[str, Any],
        rtc_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self._rtc_enabled(rtc_context):
            return None

        normalized = self._normalize_rtc_traj(rtc_context)
        if normalized is None:
            return None

        old_ee_poses, old_grippers, old_time = normalized
        B, Ta, Nee, _ = obs_data["gt_future_ee_states"].shape
        model_future_time = self._make_model_future_time(obs_data)
        valid_time_mask = (
            (model_future_time >= old_time[0])
            & (model_future_time <= old_time[-1])
        )
        valid_overlap = int(valid_time_mask.sum())
        delay_steps = max(0, int(rtc_context.get("delay_steps_est", 0)))
        min_overlap = max(0, int(rtc_context.get("rtc_min_overlap", 3)))
        valid_indices = np.flatnonzero(valid_time_mask)
        overlap_start_step = int(valid_indices[0]) if len(valid_indices) > 0 else Ta
        overlap_steps = int(valid_indices[-1] + 1) if len(valid_indices) > 0 else 0
        free_tail_steps = max(0, Ta - overlap_steps)
        has_target = valid_overlap >= min_overlap

        aligned = align.align_data(
            query_time=model_future_time,
            train_time=old_time,
            train_data={
                "ee_pose": old_ee_poses,
                "gripper": old_grippers,
            },
            interp_funcs={
                "ee_pose": align.interp_SE3_sep,
                "gripper": align.interp_linear,
            },
        )

        rtc_target_world_np = np.zeros((Ta, Nee, 17), dtype=np.float32)
        rtc_target_world_np[..., :16] = aligned["ee_pose"].astype(np.float32).reshape(Ta, Nee, 16)
        rtc_target_world_np[..., -1] = aligned["gripper"].astype(np.float32)
        rtc_target_world = torch.from_numpy(rtc_target_world_np).to(self.device)
        rtc_target_world = rtc_target_world.unsqueeze(0).repeat(B, 1, 1, 1)

        valid_ee_mask = obs_data["valid_ee_mask"].bool()
        valid_ee_per_batch = valid_ee_mask.sum(dim=-1)
        sel_index = torch.cat([
            torch.empty(n, dtype=torch.long).fill_(b)
            for b, n in enumerate(valid_ee_per_batch.tolist())
        ]).to(valid_ee_mask.device)

        if len(sel_index) == 0:
            print("[RTC] no valid EE in current observation")
            return None

        current_cam_pose = obs_data["obs_extrinsics"][:, -1, 0]
        current_ee_pose = obs_data["current_ee_pose"]
        rtc_target_action = states2action(
            current_cam_pose[sel_index],
            current_ee_pose[valid_ee_mask],
            rtc_target_world.transpose(1, 2)[valid_ee_mask],
        ).detach()

        alpha = max(0.0, float(rtc_context.get("rtc_mask_alpha", 1.0)))
        step_index = np.arange(Ta, dtype=np.float32)
        weights = np.zeros(Ta, dtype=np.float32)
        # Original RTC soft mask: hard prefix, decayed overlap, zero free tail.
        hard_prefix = step_index < min(delay_steps, overlap_steps)
        weights[hard_prefix] = 1.0

        soft_mask = (step_index >= delay_steps) & (step_index < overlap_steps)
        if soft_mask.any():
            denom = max(float(overlap_steps - delay_steps + 1), 1.0)
            c = (overlap_steps - step_index[soft_mask]) / denom
            if alpha <= 0:
                weights[soft_mask] = c
            else:
                weights[soft_mask] = (
                    np.exp(alpha * c) - 1.0
                ) / max(np.exp(alpha) - 1.0, 1e-6)
        weights[~valid_time_mask] = 0.0
        if not has_target:
            weights[:] = 0.0

        raw_group_weights = np.array(
            [
                max(0.0, float(rtc_context.get("rtc_pos_weight", 1.0))),
                max(0.0, float(rtc_context.get("rtc_rot_weight", 0.5))),
                max(0.0, float(rtc_context.get("rtc_gripper_weight", 0.2))),
            ],
            dtype=np.float32,
        )
        group_weight_norm = float(np.linalg.norm(raw_group_weights))
        if group_weight_norm > 1e-12:
            group_weights = raw_group_weights / group_weight_norm
        else:
            group_weights = np.zeros_like(raw_group_weights)

        dim_weights = np.array(
            [float(group_weights[0]) / np.sqrt(3.0)] * 3
            + [float(group_weights[1]) / np.sqrt(6.0)] * 6
            + [float(group_weights[2])],
            dtype=np.float32,
        )
        dim_weight_norm = float(np.linalg.norm(dim_weights))
        rtc_mask_np = weights[None, :, None] * dim_weights[None, None, :]
        rtc_mask = torch.from_numpy(rtc_mask_np).to(self.device)
        rtc_mask = rtc_mask.repeat(rtc_target_action.shape[0], 1, 1).detach()
        mask_nonzero_steps = int(np.sum(weights > 0.0))
        rtc_has_target = bool(has_target and np.any(weights > 0) and dim_weight_norm > 0.0)
        target_debug = {
            "rtc_has_target": rtc_has_target,
            "valid_overlap": int(valid_overlap),
            "delay_steps": int(delay_steps),
            "overlap_start_step": int(overlap_start_step),
            "overlap_steps": int(overlap_steps),
            "free_tail_steps": int(free_tail_steps),
            "mask_nonzero_steps": mask_nonzero_steps,
            "mask_sum": float(rtc_mask_np.sum()),
            "mask_square_sum": float(np.square(rtc_mask_np).sum()),
            "mask_max": float(rtc_mask_np.max()) if rtc_mask_np.size else 0.0,
            "dim_weight_raw": raw_group_weights.tolist(),
            "dim_weight_group_normalized": group_weights.tolist(),
            "dim_weight_full": dim_weights.tolist(),
            "dim_weight_group_norm": group_weight_norm,
            "dim_weight_full_norm": dim_weight_norm,
            "dim_weight_normalization": "group_l2_projection_unit_norm",
            "model_time_start": float(model_future_time[0]) if len(model_future_time) else None,
            "model_time_end": float(model_future_time[-1]) if len(model_future_time) else None,
            "old_time_start": float(old_time[0]) if len(old_time) else None,
            "old_time_end": float(old_time[-1]) if len(old_time) else None,
            "target_shape": tuple(int(x) for x in rtc_target_action.shape),
            "mask_shape": tuple(int(x) for x in rtc_mask.shape),
        }

        model_rtc_context = dict(rtc_context)
        model_rtc_context.update({
            "model_future_time": model_future_time,
            "rtc_target_world_states": rtc_target_world.detach(),
            "rtc_target_action": rtc_target_action,
            "rtc_mask": rtc_mask,
            "rtc_has_target": rtc_has_target,
            "delay_steps": delay_steps,
            "overlap_start_step": overlap_start_step,
            "overlap_steps": overlap_steps,
            "free_tail_steps": free_tail_steps,
            "valid_overlap": valid_overlap,
            "rtc_guidance_scale": float(rtc_context.get("rtc_guidance_scale", 0.5)),
            "rtc_guidance_mode": str(rtc_context.get("rtc_guidance_mode", "fixed")).lower(),
            "rtc_guidance_beta": float(rtc_context.get("rtc_guidance_beta", 5.0)),
            "rtc_tau_eps": float(rtc_context.get("rtc_tau_eps", 1e-4)),
            "rtc_max_grad_norm": float(rtc_context.get("rtc_max_grad_norm", 1.0)),
            "rtc_target_debug": target_debug,
        })
        self._print_rtc_context_summary(model_rtc_context)
        return model_rtc_context

    def _print_rtc_context_summary(self, rtc_context: Dict[str, Any]):
        target = rtc_context["rtc_target_action"]
        mask = rtc_context["rtc_mask"]
        target_debug = rtc_context.get("rtc_target_debug", {})
        print(
            "[RTC] has_target={}, delay={}, delay_time={:.1f} ms, "
            "obs_age={:.1f} ms, overlap_start={}, overlap_end={}, "
            "free_tail={}, valid_overlap={}, mask_nonzero_steps={}, mask_sum={:.4f}, "
            "mask_sq_sum={:.4f}, dim_group_norm={:.4f}, dim_full_norm={:.4f}, "
            "old_time=({},{}) model_time=({},{}) "
            "target_shape={}, mask_shape={}, target_minmax=({:.4f},{:.4f}), "
            "mask_minmax=({:.4f},{:.4f})".format(
                rtc_context["rtc_has_target"],
                rtc_context["delay_steps"],
                float(rtc_context.get("delay_time_est", 0.0)) * 1000.0,
                float(rtc_context.get("request_obs_age", 0.0)) * 1000.0,
                rtc_context["overlap_start_step"],
                rtc_context["overlap_steps"],
                rtc_context["free_tail_steps"],
                rtc_context["valid_overlap"],
                int(target_debug.get("mask_nonzero_steps", 0)),
                float(target_debug.get("mask_sum", 0.0)),
                float(target_debug.get("mask_square_sum", 0.0)),
                float(target_debug.get("dim_weight_group_norm", 0.0)),
                float(target_debug.get("dim_weight_full_norm", 0.0)),
                target_debug.get("old_time_start"),
                target_debug.get("old_time_end"),
                target_debug.get("model_time_start"),
                target_debug.get("model_time_end"),
                tuple(target.shape),
                tuple(mask.shape),
                float(target.min().item()),
                float(target.max().item()),
                float(mask.min().item()),
                float(mask.max().item()),
            )
        )

    def _capture_rng_state(self):
        state = {"cpu": torch.random.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        torch.random.set_rng_state(state["cpu"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _run_model_once(
        self,
        obs_data,
        model_rtc_context: Optional[Dict[str, Any]],
        use_rtc_guidance: bool,
    ):
        grad_context = torch.enable_grad() if use_rtc_guidance else torch.inference_mode()
        with grad_context:
            actions: Tensor = self.model(
                obs_rgbs=obs_data["obs_rgbs"],
                obs_masks=obs_data.get("obs_masks", None),
                obs_norm_xys=obs_data["obs_norm_xys"],
                obs_extrinsics=obs_data["obs_extrinsics"],
                prompt_text=obs_data["prompt_text"],

                current_ee_pose=obs_data["current_ee_pose"],
                history_ee_states=obs_data["history_ee_states"],
                gt_future_ee_states=obs_data["gt_future_ee_states"],
                valid_ee_mask=obs_data["valid_ee_mask"],
                inference=True,
                fp16=True,
                rtc_context=model_rtc_context,
            )  # (B, Ta, nee, 17)
        return actions

    def _future_states_to_action(self, obs_data, future_states: Tensor) -> Tensor:
        valid_ee_mask = obs_data["valid_ee_mask"].bool()
        valid_ee_per_batch = valid_ee_mask.sum(dim=-1)
        sel_index = torch.cat([
            torch.empty(n, dtype=torch.long).fill_(b)
            for b, n in enumerate(valid_ee_per_batch.tolist())
        ]).to(valid_ee_mask.device)
        current_cam_pose = obs_data["obs_extrinsics"][:, -1, 0]
        current_ee_pose = obs_data["current_ee_pose"]
        return states2action(
            current_cam_pose[sel_index],
            current_ee_pose[valid_ee_mask],
            future_states.transpose(1, 2)[valid_ee_mask],
        ).detach()

    @staticmethod
    def _masked_action_error(pred_action: Tensor, target: Tensor, mask: Tensor) -> Dict[str, Optional[float]]:
        with torch.no_grad():
            pred_action = pred_action.to(device=target.device, dtype=target.dtype)
            mask = mask.to(device=target.device, dtype=target.dtype)
            weighted_diff = mask * (pred_action - target)
            finite = torch.isfinite(weighted_diff).all()
            denom = mask.square().sum().clamp_min(1e-12)
            weighted_rmse = torch.sqrt(weighted_diff.square().sum() / denom)
            loss_mean = weighted_diff.square().mean()
        return {
            "weighted_rmse": float(weighted_rmse.detach().cpu().item()),
            "loss_mean": float(loss_mean.detach().cpu().item()),
            "finite": bool(finite.detach().cpu().item()),
        }

    @staticmethod
    def _to_debug_dict(
        model_rtc_context: Optional[Dict[str, Any]],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(model_rtc_context, dict):
            return extra
        debug = {
            "target": model_rtc_context.get("rtc_target_debug"),
            "denoise": model_rtc_context.get("rtc_denoise_debug"),
        }
        if extra:
            debug.update(extra)
        return debug

    def _run_inference(self, obs_data, rtc_context: Optional[Dict[str, Any]] = None):
        for k in obs_data:
            if isinstance(obs_data[k], Tensor):
                obs_data[k] = obs_data[k].to(self.device, non_blocking=True)

        self.last_rtc_debug = None
        model_rtc_context = self._build_model_rtc_context(obs_data, rtc_context)
        use_rtc_guidance = (
            isinstance(model_rtc_context, dict)
            and model_rtc_context.get("rtc_has_target", False)
        )
        debug_pair = bool(
            isinstance(model_rtc_context, dict)
            and model_rtc_context.get("rtc_debug_pair", False)
            and use_rtc_guidance
        )

        if debug_pair:
            rng_state = self._capture_rng_state()
            plain_context = dict(model_rtc_context)
            plain_context["rtc_has_target"] = False

            self._restore_rng_state(rng_state)
            plain_actions = self._run_model_once(
                obs_data=obs_data,
                model_rtc_context=plain_context,
                use_rtc_guidance=False,
            )
            plain_action = self._future_states_to_action(obs_data, plain_actions)
            plain_error = self._masked_action_error(
                plain_action,
                model_rtc_context["rtc_target_action"],
                model_rtc_context["rtc_mask"],
            )

            self._restore_rng_state(rng_state)
            actions = self._run_model_once(
                obs_data=obs_data,
                model_rtc_context=model_rtc_context,
                use_rtc_guidance=True,
            )
            guided_action = self._future_states_to_action(obs_data, actions)
            guided_error = self._masked_action_error(
                guided_action,
                model_rtc_context["rtc_target_action"],
                model_rtc_context["rtc_mask"],
            )
            ratio = None
            if plain_error["weighted_rmse"] > 1e-12:
                ratio = guided_error["weighted_rmse"] / plain_error["weighted_rmse"]
            paired_debug = {
                "plain_weighted_rmse": plain_error["weighted_rmse"],
                "guided_weighted_rmse": guided_error["weighted_rmse"],
                "guided_plain_rmse_ratio": ratio,
                "plain_loss_mean": plain_error["loss_mean"],
                "guided_loss_mean": guided_error["loss_mean"],
                "plain_finite": plain_error["finite"],
                "guided_finite": guided_error["finite"],
            }
            self.last_rtc_debug = self._to_debug_dict(
                model_rtc_context,
                {"paired": paired_debug},
            )
            print(
                "[RTC] debug_pair plain_rmse={:.6f}, guided_rmse={:.6f}, ratio={}".format(
                    paired_debug["plain_weighted_rmse"],
                    paired_debug["guided_weighted_rmse"],
                    "{:.4f}".format(ratio) if ratio is not None else None,
                )
            )
        else:
            actions = self._run_model_once(
                obs_data=obs_data,
                model_rtc_context=model_rtc_context,
                use_rtc_guidance=use_rtc_guidance,
            )
            if isinstance(model_rtc_context, dict):
                final_error = None
                if model_rtc_context.get("rtc_has_target", False):
                    pred_action = self._future_states_to_action(obs_data, actions)
                    final_error = self._masked_action_error(
                        pred_action,
                        model_rtc_context["rtc_target_action"],
                        model_rtc_context["rtc_mask"],
                    )
                self.last_rtc_debug = self._to_debug_dict(
                    model_rtc_context,
                    {"final": final_error},
                )
        return actions
    
    def _make_empty_action(self, B, Ta, Nee):
        actions = np.zeros((B, Ta, Nee, 16+1))
        actions[..., :16] = np.eye(4).ravel()
        return actions
    
    def _scatter_to_original_order(
        self, 
        nee_total: int,
        ee_indices: tuple, 
        action_selected: np.ndarray
    ):
        B, Ta, nee_selected, _ = action_selected.shape
        action_full = self._make_empty_action(B, Ta, nee_total)
        
        for i, ee_ind in enumerate(ee_indices):
            action_full[:, :, ee_ind] = action_selected[:, :, i]
        return action_full

    def get_action(
        self, 
        draw_traj: bool = False,
        compress_traj_img: bool = False,
        rtc_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Returns
        -------
            future_ee_poses (np.ndarray): shape (Ta, 4, 4), ^{world} _{ee} T
            future_grippers (np.ndarray): shape (Ta,), range [0 (close), 1 (open)]
            future_time (np.ndarray): shape (Ta,)
            traj_img (np.ndarray | None): shape (H, Ncam*W, C) if not compressed else (nbytes,)
        """
        with self.obs_lock:
            obs_frames = self.obs_frames.copy()  # shallow copy
        
        if len(obs_frames) == 0:
            return None
        
        obs_data = self._make_data_for_infer(obs_frames)
        actions = self._run_inference(obs_data, rtc_context=rtc_context)
        
        if draw_traj:
            traj_img = visualize_traj(
                data=rbd(obs_data),
                future_ee_states=[actions[0]],
                colors=[(0, 0, 255)]
            )
            if traj_img.dtype == np.float32:
                traj_img = (traj_img * 255.).clip(0, 255).astype(np.uint8)
            if compress_traj_img:
                traj_img = cv2.imencode(".jpg", traj_img)[1]
        else:
            traj_img = None
        
        self.last_obs_data = obs_data
        actions = actions.detach().cpu().numpy()  # (B, Ta, nee_sel, 17)
        actions = self._scatter_to_original_order(
            nee_total=obs_frames[-1]["ee_pose"].shape[0],
            ee_indices=self.config.ee_indices,
            action_selected=actions
        )
        B, Ta, nee, _ = actions.shape

        ee_poses = np.reshape(actions[:, :, :, :16], (B, Ta, nee, 4, 4))
        grippers = actions[:, :, :, -1]  # (B, Ta, nee)
        
        # obs_data["timestamp"]: (B,)
        latest_time = obs_data["timestamps"][0].item()
        action_dt = self.config.sample_dt * self.config.sample_state_gaps
        future_time = (1 + np.arange(Ta)) * action_dt + latest_time
        future_ee_poses = ee_poses[0]  # (Ta, nee, 4, 4)
        future_grippers = grippers[0]  # (Ta, nee)

        result = (future_ee_poses, future_grippers, future_time, traj_img)
        if isinstance(rtc_context, dict) and bool(rtc_context.get("rtc_return_debug", False)):
            return result + (self.last_rtc_debug,)
        return result
    
    def set_ensemble_nums(self, n: int):
        with self.ensembler_lock:
            self.ensemble = n
            self.pos_ensembler.reset()
            self.rot_ensembler.reset()
            self.gripper_ensembler.reset()

    def ensemble_traj(
        self, 
        future_ee_poses: np.ndarray,
        future_grippers: np.ndarray,
        future_time: np.ndarray
    ):
        if self.ensemble != 0:
            with self.ensembler_lock:
                future_ee_poses[..., :3, 3] = self.pos_ensembler.update(
                    future_ee_poses[..., :3, 3], future_time, on_SO3=False
                )
                # future_ee_poses[..., :3, :3] = self.rot_ensembler.update(
                #     future_ee_poses[..., :3, :3], future_time, on_SO3=True
                # )
                future_grippers = self.gripper_ensembler.update(
                    future_grippers, future_time, on_SO3=False
                )
        
        return future_ee_poses, future_grippers

