"""Behavior Cloning pretraining, per-subgoal competence check, and gate calibration."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from envs import NONE_SUBGOAL, make_env
from gate import calibrate_thresholds
from models import HierarchicalAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_on_datasets(
    agent: HierarchicalAgent,
    dh: dict,
    dl: dict,
    steps_h: int = 1500,
    steps_l: int = 3000,
    batch_size: int = 128,
    reinit: bool = True,
) -> dict[str, float]:
    """Fit normalizer and train (pi_h, pi_l) on aggregated (D_h, D_l)."""
    all_states = torch.cat([dh["states"], dl["states"]], dim=0)
    agent.normalizer.fit(all_states)
    if reinit:
        agent.reset_models()

    # Pre-encode unique instructions once for fast pi_h minibatches.
    uniq_instrs = sorted(set(dh["instructions"]))
    instr_embs = agent.text_encoder.encode(uniq_instrs).cpu()
    instr_map = {ins: i for i, ins in enumerate(uniq_instrs)}
    dh_g_idx = torch.tensor(
        [instr_map[ins] for ins in dh["instructions"]], dtype=torch.long
    )

    nh = len(dh["target_idxs"])
    loss_h = 0.0
    for _ in range(steps_h):
        idx = torch.randint(0, nh, (min(batch_size, nh),))
        loss_h = agent.update_high(
            dh["states"][idx],
            instr_embs[dh_g_idx[idx]],
            dh["prev_idxs"][idx],
            dh["target_idxs"][idx],
        )

    nl = len(dl["z_idxs"])
    loss_l = 0.0
    for _ in range(steps_l):
        idx = torch.randint(0, nl, (min(batch_size, nl),))
        loss_l = agent.update_low(
            dl["states"][idx], dl["z_idxs"][idx], dl["chunks"][idx]
        )

    return {"loss_h": loss_h, "loss_l": loss_l}


def evaluate_subgoal_competence(
    env,
    agent: HierarchicalAgent,
    n_episodes: int = 20,
    seed_offset: int = 8000,
) -> float:
    """Measure low-level pi_l per-subgoal completion rate given oracle z*."""
    u = env.unwrapped
    max_steps = env.spec.max_episode_steps or 180
    completed_subgoals, total_subgoals = 0, 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        plan = list(u.plan)
        total_subgoals += len(plan)
        steps = 0
        while steps < max_steps:
            z_star, _ = u.oracle_subgoal()
            chunk = agent.sample_chunk(obs[0].float().cpu(), z_star)
            done = False
            for k in range(agent.exec_steps):
                obs, _, term, trunc, info = env.step(chunk[k].cpu().numpy())
                steps += 1
                if bool(info["success"][0].item()) or bool(term[0]) or bool(trunc[0]) or steps >= max_steps:
                    done = True
                    break
                if u.is_subgoal_achieved(z_star):
                    break
            if done:
                break
        # Count how many sequential stages in the plan finished.
        if bool(u.evaluate()["success"][0].item()):
            completed_subgoals += len(plan)
        else:
            z_rem, _ = u.oracle_subgoal()
            completed_subgoals += plan.index(z_rem) if z_rem in plan else 0

    return completed_subgoals / max(total_subgoals, 1)


def evaluate_policy(
    env,
    agent: HierarchicalAgent,
    n_episodes: int = 50,
    seed_offset: int = 10000,
) -> dict[str, float]:
    """Evaluate autonomous (pi_h, pi_l) and classify failures as semantic vs exec."""
    u = env.unwrapped
    max_steps = env.spec.max_episode_steps or 180
    successes, sem_fails, exec_fails = 0, 0, 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        prev_z = NONE_SUBGOAL
        active_z: str | None = None
        steps = 0
        wrong_subgoal_decisions, total_decisions = 0, 0
        succeeded = False

        while steps < max_steps:
            if active_z is not None and u.is_subgoal_achieved(active_z):
                prev_z = active_z
            z_star, _ = u.oracle_subgoal()
            state = obs[0].float().cpu()
            active_z, _ = agent.predict_subgoal(state, u.instruction, prev_z)
            total_decisions += 1
            if active_z != z_star:
                wrong_subgoal_decisions += 1

            chunk = agent.sample_chunk(state, active_z)
            done = False
            for k in range(agent.exec_steps):
                obs, _, term, trunc, info = env.step(chunk[k].cpu().numpy())
                steps += 1
                if bool(info["success"][0].item()):
                    succeeded = True
                    done = True
                    break
                if u.is_subgoal_achieved(active_z):
                    prev_z = active_z
                    break
                if bool(term[0]) or bool(trunc[0]) or steps >= max_steps:
                    done = True
                    break
            if done:
                break

        if succeeded:
            successes += 1
        elif wrong_subgoal_decisions / max(total_decisions, 1) >= 0.35:
            sem_fails += 1
        else:
            exec_fails += 1

    n = max(n_episodes, 1)
    return {
        "success": successes / n,
        "semantic_fail_rate": sem_fails / n,
        "exec_fail_rate": exec_fails / n,
    }


def unpack_demo_datasets(raw: dict) -> tuple[dict, dict]:
    dh = {
        "states": raw["dh_states"].float(),
        "instructions": list(raw["dh_instructions"]),
        "prev_idxs": raw["dh_prev_idxs"].long(),
        "target_idxs": raw["dh_target_idxs"].long(),
    }
    dl = {
        "states": raw["dl_states"].float(),
        "z_idxs": raw["dl_z_idxs"].long(),
        "chunks": raw["dl_chunks"].float(),
    }
    return dh, dl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["stack", "sort"], default="stack")
    p.add_argument("--variant", choices=["semantic", "exec"], default="semantic")
    p.add_argument("--demos", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--steps_h", type=int, default=1500)
    p.add_argument("--steps_l", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--exec_steps", type=int, default=4)
    p.add_argument("--diffusion_steps", type=int, default=20)
    p.add_argument("--calib_episodes", type=int, default=20)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--quantile_h", type=float, default=0.85)
    p.add_argument("--quantile_l", type=float, default=0.85)
    p.add_argument("--text_mode", choices=["clip", "hash"], default="clip")
    p.add_argument("--sim_backend", default="physx_cpu")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    demos_path = args.demos or f"data/demos_{args.task}_{args.variant}.pt"
    out_dir = args.out_dir or f"checkpoints/{args.task}_{args.variant}_seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    raw = torch.load(demos_path, map_location="cpu", weights_only=True)
    dh, dl = unpack_demo_datasets(raw)

    agent = HierarchicalAgent(
        state_dim=dh["states"].shape[1],
        vocab=list(raw["vocab"]),
        horizon=int(raw["horizon"]),
        exec_steps=args.exec_steps,
        diffusion_steps=args.diffusion_steps,
        text_mode=args.text_mode,
        device=args.device,
    )
    losses = train_on_datasets(
        agent, dh, dl, steps_h=args.steps_h, steps_l=args.steps_l, batch_size=args.batch_size
    )

    env = make_env(args.task, variant=args.variant, sim_backend=args.sim_backend)
    subgoal_sr = evaluate_subgoal_competence(
        env, agent, n_episodes=min(20, args.eval_episodes)
    )
    if subgoal_sr < 0.80:
        print(
            f"WARNING: Low-level per-subgoal success is {subgoal_sr:.1%} (<80%); "
            "consider adding demos or training steps before interaction studies."
        )
    metrics = evaluate_policy(env, agent, n_episodes=args.eval_episodes)
    thresholds = calibrate_thresholds(
        env,
        agent,
        n_episodes=args.calib_episodes,
        quantile_h=args.quantile_h,
        quantile_l=args.quantile_l,
    )
    env.close()

    agent.save(os.path.join(out_dir, "bc_ckpt.pt"))
    thresholds.save(os.path.join(out_dir, "thresholds.json"))
    summary = {**losses, "subgoal_success": subgoal_sr, **metrics, "tau_h": thresholds.tau_h, "tau_l": thresholds.tau_l}
    with open(os.path.join(out_dir, "bc_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.task}/{args.variant}] BC summary: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
