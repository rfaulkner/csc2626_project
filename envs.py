"""ManiSkill3 tabletop task families for CSC2626 interactive IL experiments.

Implements two 4-subgoal Panda manipulation families, each with a
`semantic` variant (randomized instruction order + distractor cubes) and an
`exec` variant (fixed order, smaller cubes, wider poses, random pushes):
  - OrderedStack-v0: stack 3 cubes in instructed order.
  - ColorSort-v0:    place 2 instructed cubes into instructed target bins.
"""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import numpy as np
import sapien
import torch

COLORS: tuple[str, ...] = ("red", "green", "blue", "yellow")
BIN_COLORS: tuple[str, ...] = ("red", "green", "blue")
RGBA: dict[str, list[float]] = {
    "red": [0.88, 0.18, 0.18, 1.0],
    "green": [0.18, 0.78, 0.24, 1.0],
    "blue": [0.20, 0.36, 0.90, 1.0],
    "yellow": [0.92, 0.80, 0.18, 1.0],
}
BIN_RGBA: dict[str, list[float]] = {
    k: [v[0] * 0.65, v[1] * 0.65, v[2] * 0.65, 1.0]
    for k, v in RGBA.items()
    if k in BIN_COLORS
}

STACK_VOCAB: list[str] = [f"pick {c} cube" for c in COLORS] + [
    f"stack {a} on {b}" for a in COLORS for b in COLORS if a != b
]
SORT_VOCAB: list[str] = [f"pick {c} cube" for c in COLORS] + [
    f"put {c} in {b} bin" for c in COLORS for b in BIN_COLORS
]
NONE_SUBGOAL = "<none>"


class _MultiCubeBaseEnv(BaseEnv):
    """Shared single-arm Panda tabletop base with variant-controlled physics."""

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda
    vocab: list[str] = []

    def __init__(
        self,
        *args,
        variant: str = "semantic",
        robot_uids: str = "panda",
        **kwargs,
    ):
        if variant not in ("semantic", "exec"):
            raise ValueError(f"Unknown variant: {variant!r}")
        self.variant = variant
        self.is_exec = variant == "exec"
        self.half_size_val = 0.016 if self.is_exec else 0.020
        self.qpos_noise = 0.05 if self.is_exec else 0.015
        self.push_prob = 0.03 if self.is_exec else 0.0
        self.push_vel = 0.055
        self.instruction: str = ""
        self.plan: list[str] = []
        self._push_rng = np.random.RandomState(0)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _make_cube(self, half_size: float, color: list[float], name: str, pose: sapien.Pose):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[half_size] * 3)
        if self.scene.can_render():
            builder.add_box_visual(
                half_size=[half_size] * 3,
                material=sapien.render.RenderMaterial(base_color=color),
            )
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _build_cubes(self):
        self.cube_half_size = common.to_tensor(
            [self.half_size_val] * 3, device=self.device
        )
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.qpos_noise
        )
        self.table_scene.build()
        self.cubes = {
            c: self._make_cube(
                half_size=self.half_size_val,
                color=RGBA[c],
                name=f"cube_{c}",
                pose=sapien.Pose(p=[0.5 + 0.1 * i, 0, 0.1]),
            )
            for i, c in enumerate(COLORS)
        }

    def _sample_cube_poses(
        self,
        b: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        active_colors: tuple[str, ...],
    ):
        sampler = randomization.UniformPlacementSampler(
            bounds=[[x_range[0], y_range[0]], [x_range[1], y_range[1]]],
            batch_size=b,
            device=self.device,
        )
        radius = float(np.hypot(self.half_size_val, self.half_size_val) + 0.028)
        yaw_lim = 0.35 if self.is_exec else 0.15
        for idx, c in enumerate(COLORS):
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = self.half_size_val
            if c in active_colors:
                xyz[:, :2] = sampler.sample(radius, 200, verbose=False)
            else:
                xyz[:, 0] = -0.34
                xyz[:, 1] = 0.30 + 0.06 * idx
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
                bounds=(-yaw_lim, yaw_lim),
            )
            self.cubes[c].set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def _before_control_step(self):
        if self.push_prob <= 0.0 or self._elapsed_steps[0].item() < 5:
            return
        if self._push_rng.rand() >= self.push_prob:
            return
        poses = {c: self.cubes[c].pose.p[0] for c in COLORS[:3]}
        candidates = []
        for c, p in poses.items():
            if bool(self.agent.is_grasping(self.cubes[c])[0].item()):
                continue
            if p[2].item() > self.half_size_val + 0.006:
                continue
            has_upper = any(
                other != c
                and poses[other][2].item() > self.half_size_val * 2.2
                and float(torch.linalg.norm(poses[other][:2] - p[:2]).item())
                < self.half_size_val * 2.5
                for other in poses
            )
            if not has_upper:
                candidates.append(c)
        if not candidates:
            return
        target = self.cubes[candidates[self._push_rng.randint(len(candidates))]]
        dv = self._push_rng.uniform(-self.push_vel, self.push_vel, size=2)
        vel = torch.tensor(
            [[float(dv[0]), float(dv[1]), 0.0]], device=self.device
        )
        target.set_linear_velocity(vel)

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        del info
        tcp = self.agent.tcp.pose.p
        extra: dict[str, torch.Tensor] = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
        }
        if "state" in self.obs_mode:
            for c in COLORS:
                p = self.cubes[c].pose.p
                extra[f"{c}_pose"] = self.cubes[c].pose.raw_pose
                extra[f"tcp_to_{c}"] = p - tcp
        return extra

    def is_picked(self, color: str) -> bool:
        return bool(self.agent.is_grasping(self.cubes[color])[0].item())

    def grasped_color(self) -> str | None:
        for c in COLORS:
            if bool(self.agent.is_grasping(self.cubes[c])[0].item()):
                return c
        return None

    def save_snapshot(self) -> dict[str, Any]:
        return {
            "state_dict": copy.deepcopy(self.get_state_dict()),
            "elapsed_steps": self._elapsed_steps.clone(),
            "instruction": self.instruction,
            "plan": list(self.plan),
            "spec": copy.deepcopy(self._task_spec()),
            "push_rng": copy.deepcopy(self._push_rng.get_state()),
        }

    def load_snapshot(self, snap: dict[str, Any]):
        self._restore_task_spec(snap["spec"])
        self.instruction = snap["instruction"]
        self.plan = list(snap["plan"])
        self._push_rng.set_state(snap["push_rng"])
        self.set_state_dict(copy.deepcopy(snap["state_dict"]))
        self._elapsed_steps = snap["elapsed_steps"].clone()
        self.agent.controller.reset()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        del obs, action
        return info["success"].float()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        del obs, action
        return info["success"].float()

    def _task_spec(self) -> Any:
        raise NotImplementedError

    def _restore_task_spec(self, spec: Any):
        raise NotImplementedError

    def oracle_subgoal(self) -> tuple[str, str]:
        raise NotImplementedError

    def is_subgoal_achieved(self, subgoal: str) -> bool:
        raise NotImplementedError


@register_env("OrderedStack-v0", max_episode_steps=220)
class OrderedStackEnv(_MultiCubeBaseEnv):
    """Stack 3 cubes in instructed order: c1 on c0, then c2 on c1."""

    vocab = STACK_VOCAB

    def __init__(self, *args, **kwargs):
        self.order: tuple[str, str, str] = ("green", "red", "blue")
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        del options
        self._build_cubes()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        del options
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            ep_seed = int(self._episode_seed[0])
            self._push_rng.seed(ep_seed + 17)
            rng = np.random.RandomState(ep_seed)
            if self.is_exec:
                self.order = ("green", "red", "blue")
                self._sample_cube_poses(
                    b, (-0.12, 0.10), (-0.20, 0.20), COLORS[:3]
                )
            else:
                perm = rng.choice(COLORS, size=3, replace=False)
                self.order = (str(perm[0]), str(perm[1]), str(perm[2]))
                self._sample_cube_poses(b, (-0.08, 0.08), (-0.16, 0.16), COLORS)
            c0, c1, c2 = self.order
            self.instruction = f"stack {c1} on {c0}, then {c2} on {c1}"
            self.plan = [
                f"pick {c1} cube",
                f"stack {c1} on {c0}",
                f"pick {c2} cube",
                f"stack {c2} on {c1}",
            ]

    def _is_on(self, top_c: str, bot_c: str, require_released: bool = True) -> bool:
        top = self.cubes[top_c]
        bot = self.cubes[bot_c]
        dp = (top.pose.p - bot.pose.p)[0]
        h = self.half_size_val
        xy_ok = float(torch.linalg.norm(dp[:2]).item()) <= np.hypot(h, h) + 0.007
        z_ok = abs(float(dp[2].item()) - 2.0 * h) <= 0.007
        if not (xy_ok and z_ok):
            return False
        if not require_released:
            return True
        released = not bool(self.agent.is_grasping(top)[0].item())
        static = bool(top.is_static(lin_thresh=2e-2, ang_thresh=0.6)[0].item())
        return released and static

    def evaluate(self) -> dict[str, torch.Tensor]:
        c0, c1, c2 = self.order
        s1 = self._is_on(c1, c0)
        s2 = s1 and self._is_on(c2, c1)
        return {
            "stage1": torch.tensor([s1], dtype=torch.bool, device=self.device),
            "success": torch.tensor([s2], dtype=torch.bool, device=self.device),
        }

    def oracle_subgoal(self) -> tuple[str, str]:
        c0, c1, c2 = self.order
        held = self.grasped_color()
        if not self._is_on(c1, c0):
            if held == c1:
                return self.plan[1], self.plan[0]
            return self.plan[0], NONE_SUBGOAL
        if not self._is_on(c2, c1):
            if held == c2:
                return self.plan[3], self.plan[2]
            return self.plan[2], self.plan[1]
        return self.plan[3], self.plan[3]

    def is_subgoal_achieved(self, subgoal: str) -> bool:
        parts = subgoal.split()
        if len(parts) == 3 and parts[0] == "pick":
            return self.is_picked(parts[1])
        if len(parts) == 4 and parts[0] == "stack":
            return self._is_on(parts[1], parts[3])
        return False

    def _task_spec(self):
        return self.order

    def _restore_task_spec(self, spec):
        self.order = tuple(spec)


@register_env("ColorSort-v0", max_episode_steps=220)
class ColorSortEnv(_MultiCubeBaseEnv):
    """Sort 2 instructed cubes into instructed target bins in order."""

    vocab = SORT_VOCAB
    BIN_POS: dict[str, tuple[float, float]] = {
        "red": (0.13, -0.15),
        "green": (0.13, 0.00),
        "blue": (0.13, 0.15),
    }

    def __init__(self, *args, **kwargs):
        self.pairs: list[tuple[str, str]] = [("red", "red"), ("green", "green")]
        self.bin_tol = 0.026 if kwargs.get("variant") == "exec" else 0.036
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        del options
        self._build_cubes()
        self.bins = {}
        if self.scene.can_render():
            for b_col, (bx, by) in self.BIN_POS.items():
                builder = self.scene.create_actor_builder()
                builder.add_box_visual(
                    half_size=[0.042, 0.042, 0.002],
                    material=sapien.render.RenderMaterial(base_color=BIN_RGBA[b_col]),
                )
                builder.set_initial_pose(sapien.Pose(p=[bx, by, 0.002]))
                self.bins[b_col] = builder.build_kinematic(name=f"bin_{b_col}")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        del options
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            for b_col, (bx, by) in self.BIN_POS.items():
                if b_col in self.bins:
                    self.bins[b_col].set_pose(sapien.Pose(p=[bx, by, 0.002]))
            ep_seed = int(self._episode_seed[0])
            self._push_rng.seed(ep_seed + 31)
            rng = np.random.RandomState(ep_seed)
            if self.is_exec:
                self.pairs = [("red", "red"), ("green", "green")]
                self._sample_cube_poses(
                    b, (-0.14, 0.02), (-0.20, 0.20), COLORS[:3]
                )
            else:
                cubes = rng.choice(COLORS, size=2, replace=False)
                bins = rng.choice(BIN_COLORS, size=2, replace=False)
                self.pairs = [(str(cubes[0]), str(bins[0])), (str(cubes[1]), str(bins[1]))]
                self._sample_cube_poses(
                    b, (-0.11, 0.02), (-0.16, 0.16), COLORS
                )
            (c1, b1), (c2, b2) = self.pairs
            self.instruction = (
                f"put {c1} cube in {b1} bin, then {c2} cube in {b2} bin"
            )
            self.plan = [
                f"pick {c1} cube",
                f"put {c1} in {b1} bin",
                f"pick {c2} cube",
                f"put {c2} in {b2} bin",
            ]

    def _is_in_bin(self, cube_c: str, bin_c: str) -> bool:
        cube = self.cubes[cube_c]
        p = cube.pose.p[0]
        bx, by = self.BIN_POS[bin_c]
        xy = float(np.hypot(float(p[0].item()) - bx, float(p[1].item()) - by))
        z_ok = float(p[2].item()) <= self.half_size_val + 0.008
        if not (xy <= self.bin_tol and z_ok):
            return False
        released = not bool(self.agent.is_grasping(cube)[0].item())
        static = bool(cube.is_static(lin_thresh=2e-2, ang_thresh=0.6)[0].item())
        return released and static

    def evaluate(self) -> dict[str, torch.Tensor]:
        (c1, b1), (c2, b2) = self.pairs
        s1 = self._is_in_bin(c1, b1)
        s2 = s1 and self._is_in_bin(c2, b2)
        return {
            "stage1": torch.tensor([s1], dtype=torch.bool, device=self.device),
            "success": torch.tensor([s2], dtype=torch.bool, device=self.device),
        }

    def oracle_subgoal(self) -> tuple[str, str]:
        (c1, b1), (c2, b2) = self.pairs
        held = self.grasped_color()
        if not self._is_in_bin(c1, b1):
            if held == c1:
                return self.plan[1], self.plan[0]
            return self.plan[0], NONE_SUBGOAL
        if not self._is_in_bin(c2, b2):
            if held == c2:
                return self.plan[3], self.plan[2]
            return self.plan[2], self.plan[1]
        return self.plan[3], self.plan[3]

    def is_subgoal_achieved(self, subgoal: str) -> bool:
        parts = subgoal.split()
        if len(parts) == 3 and parts[0] == "pick":
            return self.is_picked(parts[1])
        if len(parts) == 5 and parts[0] == "put":
            return self._is_in_bin(parts[1], parts[3])
        return False

    def _task_spec(self):
        return list(self.pairs)

    def _restore_task_spec(self, spec):
        self.pairs = [tuple(p) for p in spec]


TASK_IDS = {"stack": "OrderedStack-v0", "sort": "ColorSort-v0"}


def make_env(
    task: str,
    variant: str = "semantic",
    sim_backend: str = "physx_cpu",
    render_backend: str = "none",
) -> BaseEnv:
    """Instantiate a single-arm Panda environment in state/delta-EE mode."""
    env_id = TASK_IDS.get(task, task)
    return gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        sim_backend=sim_backend,
        render_backend=render_backend,
        variant=variant,
    )
