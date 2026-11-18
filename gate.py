"""Two-level uncertainty gate and held-out rollout threshold calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os

import numpy as np

from envs import NONE_SUBGOAL
from models import HierarchicalAgent


@dataclass
class GateThresholds:
    """Calibrated high-level entropy (tau_h) and Diff-DAgger loss (tau_l) cutoffs."""

    tau_h: float
    tau_l: float
    quantile_h: float = 0.85
    quantile_l: float = 0.85
    fire_rate: float = 0.20

    def query_high(self, u_h: float) -> bool:
        return u_h > self.tau_h

    def query_low(self, u_l: float) -> bool:
        return u_l > self.tau_l

    def should_query(self, u_h: float, u_l: float) -> bool:
        return self.query_high(u_h) or self.query_low(u_l)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GateThresholds":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


def calibrate_thresholds(
    env,
    agent: HierarchicalAgent,
    n_episodes: int = 20,
    seed_offset: int = 5000,
    quantile_h: float = 0.85,
    quantile_l: float = 0.85,
) -> GateThresholds:
    """Calibrate (tau_h, tau_l) once on held-out rollouts of the initial BC policy."""
    u_h_scores: list[float] = []
    u_l_scores: list[float] = []
    u = env.unwrapped

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        prev_z = NONE_SUBGOAL
        active_z: str | None = None
        steps = 0
        max_steps = env.spec.max_episode_steps or 220

        while steps < max_steps:
            if active_z is not None and u.is_subgoal_achieved(active_z):
                prev_z = active_z
            state = obs[0].float().cpu()
            active_z, u_h = agent.predict_subgoal(state, u.instruction, prev_z)
            chunk = agent.sample_chunk(state, active_z)
            u_l = agent.diffusion_loss_score(state, active_z, chunk)
            u_h_scores.append(u_h)
            u_l_scores.append(u_l)

            done = False
            for k in range(agent.exec_steps):
                obs, _, term, trunc, info = env.step(chunk[k].cpu().numpy())
                steps += 1
                if u.is_subgoal_achieved(active_z):
                    prev_z = active_z
                    break
                if bool(info["success"][0].item()) or bool(term[0]) or bool(trunc[0]) or steps >= max_steps:
                    done = True
                    break
            if done:
                break

    tau_h = float(np.quantile(u_h_scores, quantile_h)) if u_h_scores else 0.5
    tau_l = float(np.quantile(u_l_scores, quantile_l)) if u_l_scores else 0.5
    fire_rate = (
        float(np.mean([(uh > tau_h) or (ul > tau_l) for uh, ul in zip(u_h_scores, u_l_scores)]))
        if u_h_scores
        else 0.20
    )
    return GateThresholds(
        tau_h=tau_h,
        tau_l=tau_l,
        quantile_h=quantile_h,
        quantile_l=quantile_l,
        fire_rate=round(fire_rate, 4),
    )
