"""Language and closed-loop pick-and-place action oracles for ManiSkill3 tasks.

Supports mid-episode takeovers from arbitrary states, including releasing a
wrongly grasped cube and re-grasping if a cube is dropped mid-place.
"""

from __future__ import annotations

import numpy as np


class LanguageOracle:
    """Returns ground-truth subgoal z* (optionally corrupted with prob noise_p)."""

    def __init__(self, vocab: list[str], noise_p: float = 0.0, seed: int = 0):
        self.vocab = vocab
        self.noise_p = noise_p
        self.rng = np.random.RandomState(seed)

    def query(self, env) -> str:
        z_star, _ = env.unwrapped.oracle_subgoal()
        if self.noise_p > 0.0 and self.rng.rand() < self.noise_p:
            others = [z for z in self.vocab if z != z_star]
            return str(self.rng.choice(others))
        return z_star


class ScriptedPickPlaceExpert:
    """Closed-loop Cartesian PD expert in pd_ee_delta_pos ([-1, 1]^4)."""

    def __init__(self, gain: float = 7.5, max_step: float = 0.45):
        self.gain = gain
        self.max_step = max_step
        self._close_hold = 0
        self._open_hold = 0
        self._active_subgoal = ""

    def reset(self):
        self._close_hold = 0
        self._open_hold = 0
        self._active_subgoal = ""

    def _pd(self, target_xyz: np.ndarray, tcp_xyz: np.ndarray, grip: float) -> np.ndarray:
        err = target_xyz - tcp_xyz
        dist = float(np.linalg.norm(err))
        step = min(dist * self.gain, self.max_step) * (err / dist) if dist > 1e-6 else np.zeros(3)
        return np.array([step[0], step[1], step[2], grip], dtype=np.float32)

    def act(self, env, subgoal: str) -> np.ndarray:
        u = env.unwrapped
        tcp = u.agent.tcp.pose.p[0].cpu().numpy()
        finger_q = float(u.agent.robot.get_qpos()[0, -2:].mean().item())
        held = u.grasped_color()

        # Always finish upward release retraction before starting next subgoal.
        if self._open_hold > 0:
            self._open_hold -= 1
            return self._pd(tcp + np.array([0.0, 0.0, 0.06]), tcp, 1.0)

        if subgoal != self._active_subgoal:
            self._active_subgoal = subgoal
            self._close_hold = 0

        parts = subgoal.split()
        if parts[0] == "pick":
            return self._act_pick(u, tcp, finger_q, held, parts[1])
        if parts[0] == "stack":
            src, dst = parts[1], parts[3]
            dst_p = u.cubes[dst].pose.p[0].cpu().numpy()
            goal = np.array([dst_p[0], dst_p[1], dst_p[2] + 2.0 * u.half_size_val + 0.003])
            return self._act_place(u, tcp, finger_q, held, src, goal, tol=0.008)
        if parts[0] == "put":
            src, bin_c = parts[1], parts[3]
            bx, by = u.BIN_POS[bin_c]
            goal = np.array([bx, by, u.half_size_val + 0.004])
            return self._act_place(u, tcp, finger_q, held, src, goal, tol=0.015)
        raise ValueError(f"Unrecognized subgoal: {subgoal!r}")

    def _act_pick(
        self,
        u,
        tcp: np.ndarray,
        finger_q: float,
        held: str | None,
        color: str,
    ) -> np.ndarray:
        if held is not None and held != color:
            self._open_hold = 6
            return self._pd(tcp + np.array([0.0, 0.0, 0.06]), tcp, 1.0)

        cube_p = u.cubes[color].pose.p[0].cpu().numpy()
        clear_z = max(0.16, float(cube_p[2]) + 0.09)

        if held == color:
            self._close_hold = 0
            return self._pd(np.array([cube_p[0], cube_p[1], clear_z]), tcp, -1.0)

        if self._close_hold > 0:
            self._close_hold -= 1
            if self._close_hold == 0 and finger_q < 0.005:
                self._open_hold = 6
                return self._pd(cube_p + np.array([0.0, 0.0, 0.06]), tcp, 1.0)
            return self._pd(cube_p, tcp, -1.0)

        xy_err = float(np.linalg.norm(tcp[:2] - cube_p[:2]))
        if xy_err > 0.012:
            if tcp[2] < clear_z - 0.02:
                return self._pd(np.array([tcp[0], tcp[1], clear_z]), tcp, 1.0)
            return self._pd(np.array([cube_p[0], cube_p[1], clear_z]), tcp, 1.0)

        z_err = abs(float(tcp[2] - cube_p[2]))
        if z_err > 0.006:
            return self._pd(cube_p, tcp, 1.0)

        self._close_hold = 8
        return self._pd(cube_p, tcp, -1.0)

    def _act_place(
        self,
        u,
        tcp: np.ndarray,
        finger_q: float,
        held: str | None,
        src_color: str,
        goal_xyz: np.ndarray,
        tol: float,
    ) -> np.ndarray:
        src_p = u.cubes[src_color].pose.p[0].cpu().numpy()
        err = goal_xyz - src_p
        xy_to_goal = float(np.linalg.norm(err[:2]))
        z_to_goal = abs(float(err[2]))

        if held != src_color:
            if xy_to_goal <= tol * 2.0 and z_to_goal <= 0.014:
                return self._pd(tcp + np.array([0.0, 0.0, 0.07]), tcp, 1.0)
            return self._act_pick(u, tcp, finger_q, held, src_color)

        # Compensate for grasp offset (tcp - src_p) so the cube center hits goal_xyz.
        grasp_offset = tcp - src_p
        goal_tcp = goal_xyz + grasp_offset
        transit_z = max(0.17, float(goal_tcp[2]) + 0.075)

        if xy_to_goal > tol:
            if tcp[2] < transit_z - 0.025:
                return self._pd(np.array([tcp[0], tcp[1], transit_z]), tcp, -1.0)
            return self._pd(np.array([goal_tcp[0], goal_tcp[1], transit_z]), tcp, -1.0)

        if src_p[2] > goal_xyz[2] + 0.005:
            return self._pd(goal_tcp, tcp, -1.0)

        self._open_hold = 6
        return self._pd(tcp + np.array([0.0, 0.0, 0.04]), tcp, 1.0)
