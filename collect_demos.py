"""Collect scripted expert demonstrations segmented into D_h and D_l."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from envs import NONE_SUBGOAL, make_env
from expert import ScriptedPickPlaceExpert


def build_chunks_within_subgoal(
    actions: list[np.ndarray], z_idxs: list[int], horizon: int
) -> torch.Tensor:
    """Slice length-H action chunks, padding at subgoal transitions."""
    n = len(actions)
    chunks = np.zeros((n, horizon, actions[0].shape[0]), dtype=np.float32)
    for t in range(n):
        last_valid = actions[t]
        for k in range(horizon):
            idx = t + k
            if idx < n and z_idxs[idx] == z_idxs[t]:
                last_valid = actions[idx]
            chunks[t, k] = last_valid
    return torch.from_numpy(chunks)


def collect_demonstrations(
    task: str,
    variant: str,
    num_demos: int = 50,
    horizon: int = 8,
    seed: int = 0,
    sim_backend: str = "physx_cpu",
) -> dict:
    env = make_env(task, variant=variant, sim_backend=sim_backend)
    u = env.unwrapped
    vocab = list(u.vocab)
    z_to_idx = {z: i for i, z in enumerate(vocab)}
    none_idx = len(vocab)
    expert = ScriptedPickPlaceExpert()
    max_steps = env.spec.max_episode_steps or 220

    dh_states, dh_instrs, dh_prev, dh_target = [], [], [], []
    dl_states, dl_z, dl_chunks = [], [], []

    attempts, collected, ep_lengths = 0, 0, []
    while collected < num_demos and attempts < num_demos * 8:
        obs, _ = env.reset(seed=seed + attempts)
        attempts += 1
        expert.reset()

        ep_s, ep_prev, ep_z, ep_act = [], [], [], []
        succeeded = False
        for _ in range(max_steps):
            z_star, prev_z = u.oracle_subgoal()
            act = expert.act(env, z_star)
            ep_s.append(obs[0].float().cpu())
            ep_prev.append(none_idx if prev_z == NONE_SUBGOAL else z_to_idx[prev_z])
            ep_z.append(z_to_idx[z_star])
            ep_act.append(act)

            obs, _, term, trunc, info = env.step(act)
            if bool(info["success"][0].item()):
                succeeded = True
                break
            if bool(term[0]) or bool(trunc[0]):
                break

        if not succeeded:
            continue

        collected += 1
        ep_lengths.append(len(ep_act))
        states_t = torch.stack(ep_s, dim=0)
        dh_states.append(states_t)
        dh_instrs.extend([u.instruction] * len(ep_act))
        dh_prev.extend(ep_prev)
        dh_target.extend(ep_z)

        dl_states.append(states_t)
        dl_z.extend(ep_z)
        dl_chunks.append(build_chunks_within_subgoal(ep_act, ep_z, horizon))

    env.close()
    if collected < num_demos:
        raise RuntimeError(
            f"Only collected {collected}/{num_demos} demos in {attempts} attempts."
        )

    print(
        f"[{task}/{variant}] Collected {collected} demos in {attempts} attempts "
        f"(expert success={collected / attempts:.1%}, mean_steps={np.mean(ep_lengths):.1f})"
    )
    return {
        "task": task,
        "variant": variant,
        "vocab": vocab,
        "horizon": horizon,
        "dh_states": torch.cat(dh_states, dim=0),
        "dh_instructions": dh_instrs,
        "dh_prev_idxs": torch.tensor(dh_prev, dtype=torch.long),
        "dh_target_idxs": torch.tensor(dh_target, dtype=torch.long),
        "dl_states": torch.cat(dl_states, dim=0),
        "dl_z_idxs": torch.tensor(dl_z, dtype=torch.long),
        "dl_chunks": torch.cat(dl_chunks, dim=0),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["stack", "sort"], default="stack")
    p.add_argument("--variant", choices=["semantic", "exec"], default="semantic")
    p.add_argument("--num_demos", type=int, default=50)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim_backend", default="physx_cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = args.out or f"data/demos_{args.task}_{args.variant}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    data = collect_demonstrations(
        args.task,
        args.variant,
        num_demos=args.num_demos,
        horizon=args.horizon,
        seed=args.seed,
        sim_backend=args.sim_backend,
    )
    torch.save(data, out)
    print(
        f"Saved {out}: |D_h|={len(data['dh_target_idxs'])}, |D_l|={len(data['dl_z_idxs'])}"
    )


if __name__ == "__main__":
    main()
