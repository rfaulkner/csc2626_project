"""Robot-gated interactive imitation learning loop (BC, LANG, ACT, RANDOM-*)."""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from collect_demos import build_chunks_within_subgoal
from envs import NONE_SUBGOAL, make_env
from expert import LanguageOracle, ScriptedPickPlaceExpert
from gate import GateThresholds
from models import HierarchicalAgent
from train_bc import (
    evaluate_policy,
    set_seed,
    train_on_datasets,
    unpack_demo_datasets,
)

CHANNELS = ("bc", "lang", "act", "act_label", "random_lang", "random_act")


def counterfactual_would_fail(
    cf_env,
    snap: dict,
    agent: HierarchicalAgent,
    prev_z: str,
) -> bool:
    """Roll out unassisted learner from saved state to test if query prevented failure."""
    cu = cf_env.unwrapped
    cu.load_snapshot(snap)
    obs = cu.get_obs()
    max_steps = cf_env.spec.max_episode_steps or 180
    steps = int(cu.elapsed_steps[0].item())
    active_z: str | None = None

    while steps < max_steps:
        if active_z is not None and cu.is_subgoal_achieved(active_z):
            prev_z = active_z
        state = obs[0].float().cpu()
        active_z, _ = agent.predict_subgoal(state, cu.instruction, prev_z)
        chunk = agent.sample_chunk(state, active_z)
        for k in range(agent.exec_steps):
            obs, _, term, trunc, info = cu.step(chunk[k].cpu().numpy())
            steps += 1
            if bool(info["success"][0].item()):
                return False
            if cu.is_subgoal_achieved(active_z):
                prev_z = active_z
                break
            if bool(term[0]) or bool(trunc[0]) or steps >= max_steps:
                return True
    return True


def run_interaction_round(
    env,
    cf_env,
    agent: HierarchicalAgent,
    gate: GateThresholds,
    lang_oracle: LanguageOracle,
    act_expert: ScriptedPickPlaceExpert,
    dh: dict,
    dl: dict,
    channel: str,
    round_budget: float,
    lam: float,
    k_max: int,
    random_prob: float,
    seed_base: int,
    episodes: int,
    cf_limit: int,
    rng: np.random.RandomState,
) -> dict[str, float]:
    """Collect one round of interactive rollouts and aggregate supervision into Dh/Dl."""
    u = env.unwrapped
    max_steps = env.spec.max_episode_steps or 180
    spent = 0.0
    queries = 0
    cf_checks, cf_true_positives = 0, 0

    new_dh_s, new_dh_g, new_dh_prev, new_dh_tgt = [], [], [], []
    new_dl_s, new_dl_z, new_dl_chunks = [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        act_expert.reset()
        prev_z = NONE_SUBGOAL
        active_z: str | None = None
        steps = 0

        while steps < max_steps:
            if active_z is not None and u.is_subgoal_achieved(active_z):
                prev_z = active_z

            state = obs[0].float().cpu()
            active_z, u_h = agent.predict_subgoal(state, u.instruction, prev_z)
            chunk = agent.sample_chunk(state, active_z)
            u_l = agent.diffusion_loss_score(state, active_z, chunk)

            if channel == "bc":
                fire = False
            elif channel.startswith("random_"):
                fire = bool(rng.rand() < random_prob)
            else:
                fire = gate.should_query(u_h, u_l)

            min_query_cost = 1.0 if channel in ("lang", "random_lang") else 1.0 + lam
            can_query = fire and (spent + min_query_cost <= round_budget + 1e-9)
            if can_query:
                if cf_env is not None and cf_checks < cf_limit:
                    snap = u.save_snapshot()
                    cf_checks += 1
                    if counterfactual_would_fail(cf_env, snap, agent, prev_z):
                        cf_true_positives += 1

                queries += 1
                if channel in ("lang", "random_lang"):
                    z_star = lang_oracle.query(env)
                    spent += 1.0
                    new_dh_s.append(state)
                    new_dh_g.append(u.instruction)
                    new_dh_prev.append(agent.encode_prev_subgoal(prev_z))
                    new_dh_tgt.append(agent.subgoal_to_idx[z_star])
                    active_z = z_star
                    chunk = agent.sample_chunk(state, active_z)
                else:
                    # Action takeover (act, random_act, act_label).
                    z_star, oracle_prev = u.oracle_subgoal()
                    if channel == "act_label":
                        new_dh_s.append(state)
                        new_dh_g.append(u.instruction)
                        new_dh_prev.append(agent.encode_prev_subgoal(oracle_prev))
                        new_dh_tgt.append(agent.subgoal_to_idx[z_star])

                    k_budget = (
                        k_max
                        if lam <= 0.0
                        else min(k_max, int((round_budget - spent - 1.0 + 1e-9) // lam))
                    )
                    act_expert.reset()
                    tk_states, tk_z, tk_acts = [], [], []
                    k_steps = 0
                    z_idx = agent.subgoal_to_idx[z_star]
                    done_ep = False
                    while k_steps < k_budget and steps < max_steps:
                        a_star = act_expert.act(env, z_star)
                        tk_states.append(obs[0].float().cpu())
                        tk_z.append(z_idx)
                        tk_acts.append(a_star)
                        obs, _, term, trunc, info = env.step(a_star)
                        steps += 1
                        k_steps += 1
                        if bool(info["success"][0].item()):
                            done_ep = True
                            break
                        if u.is_subgoal_achieved(z_star):
                            prev_z = z_star
                            active_z = None
                            break
                        if bool(term[0]) or bool(trunc[0]):
                            done_ep = True
                            break

                    spent += 1.0 + lam * k_steps
                    if tk_acts:
                        new_dl_s.append(torch.stack(tk_states, dim=0))
                        new_dl_z.extend(tk_z)
                        new_dl_chunks.append(
                            build_chunks_within_subgoal(tk_acts, tk_z, agent.horizon)
                        )
                    if done_ep or steps >= max_steps:
                        break
                    continue

            # Execute learner action chunk.
            done_ep = False
            for k in range(agent.exec_steps):
                obs, _, term, trunc, info = env.step(chunk[k].cpu().numpy())
                steps += 1
                if bool(info["success"][0].item()):
                    done_ep = True
                    break
                if u.is_subgoal_achieved(active_z):
                    prev_z = active_z
                    break
                if bool(term[0]) or bool(trunc[0]) or steps >= max_steps:
                    done_ep = True
                    break
            if done_ep:
                break

    if new_dh_s:
        dh["states"] = torch.cat([dh["states"], torch.stack(new_dh_s, dim=0)], dim=0)
        dh["instructions"].extend(new_dh_g)
        dh["prev_idxs"] = torch.cat(
            [dh["prev_idxs"], torch.tensor(new_dh_prev, dtype=torch.long)], dim=0
        )
        dh["target_idxs"] = torch.cat(
            [dh["target_idxs"], torch.tensor(new_dh_tgt, dtype=torch.long)], dim=0
        )
    if new_dl_s:
        dl["states"] = torch.cat([dl["states"], torch.cat(new_dl_s, dim=0)], dim=0)
        dl["z_idxs"] = torch.cat(
            [dl["z_idxs"], torch.tensor(new_dl_z, dtype=torch.long)], dim=0
        )
        dl["chunks"] = torch.cat([dl["chunks"], torch.cat(new_dl_chunks, dim=0)], dim=0)

    precision = cf_true_positives / cf_checks if cf_checks > 0 else float("nan")
    return {"round_cost": spent, "queries": queries, "query_precision": precision}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["stack", "sort"], default="stack")
    p.add_argument("--variant", choices=["semantic", "exec"], default="semantic")
    p.add_argument("--channel", choices=CHANNELS, required=True)
    p.add_argument("--budget", type=float, default=100.0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--episodes_per_round", type=int, default=20)
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--k_max", type=int, default=45)
    p.add_argument("--random_prob", type=float, default=0.15)
    p.add_argument("--oracle_noise", type=float, default=0.0)
    p.add_argument("--steps_h", type=int, default=1500)
    p.add_argument("--steps_l", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--cf_samples_per_round", type=int, default=15)
    p.add_argument("--demos", default=None)
    p.add_argument("--bc_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--text_mode", choices=["clip", "hash"], default="clip")
    p.add_argument("--sim_backend", default="physx_cpu")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed + 999)
    demos_path = args.demos or f"data/demos_{args.task}_{args.variant}.pt"
    bc_dir = args.bc_dir or f"checkpoints/{args.task}_{args.variant}_seed{args.seed}"
    out_dir = args.out_dir or (
        f"results/{args.task}_{args.variant}/{args.channel}_lam{args.lam:g}_B{args.budget:g}/seed{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)

    raw = torch.load(demos_path, map_location="cpu", weights_only=True)
    dh, dl = unpack_demo_datasets(raw)
    agent = HierarchicalAgent.load(
        os.path.join(bc_dir, "bc_ckpt.pt"),
        text_mode=args.text_mode,
        device=args.device,
    )
    gate = GateThresholds.load(os.path.join(bc_dir, "thresholds.json"))

    env = make_env(args.task, variant=args.variant, sim_backend=args.sim_backend)
    cf_env = (
        make_env(args.task, variant=args.variant, sim_backend=args.sim_backend)
        if args.cf_samples_per_round > 0 and args.channel != "bc"
        else None
    )
    lang_oracle = LanguageOracle(
        agent.vocab, noise_p=args.oracle_noise, seed=args.seed + 77
    )
    act_expert = ScriptedPickPlaceExpert()

    csv_path = os.path.join(out_dir, "metrics.csv")
    fields = [
        "task",
        "variant",
        "channel",
        "lam",
        "budget",
        "seed",
        "round",
        "cum_cost",
        "queries",
        "query_precision",
        "success",
        "semantic_fail_rate",
        "exec_fail_rate",
        "dh_size",
        "dl_size",
    ]

    cum_cost = 0.0
    total_queries = 0
    eval0 = evaluate_policy(env, agent, n_episodes=args.eval_episodes, seed_offset=20000)
    rows = [
        {
            "task": args.task,
            "variant": args.variant,
            "channel": args.channel,
            "lam": args.lam,
            "budget": args.budget,
            "seed": args.seed,
            "round": 0,
            "cum_cost": 0.0,
            "queries": 0,
            "query_precision": float("nan"),
            **eval0,
            "dh_size": len(dh["target_idxs"]),
            "dl_size": len(dl["z_idxs"]),
        }
    ]
    print(f"[Round 0/{args.rounds}] cost=0.0 success={eval0['success']:.3f}")

    round_share = args.budget / max(args.rounds, 1)
    for r in range(1, args.rounds + 1):
        if args.channel == "bc":
            r_stats = {"round_cost": 0.0, "queries": 0, "query_precision": float("nan")}
            eval_r = eval0
        else:
            r_stats = run_interaction_round(
                env=env,
                cf_env=cf_env,
                agent=agent,
                gate=gate,
                lang_oracle=lang_oracle,
                act_expert=act_expert,
                dh=dh,
                dl=dl,
                channel=args.channel,
                round_budget=round_share,
                lam=args.lam,
                k_max=args.k_max,
                random_prob=args.random_prob,
                seed_base=30000 + args.seed * 1000 + r * 100,
                episodes=args.episodes_per_round,
                cf_limit=args.cf_samples_per_round,
                rng=rng,
            )
            cum_cost += r_stats["round_cost"]
            total_queries += int(r_stats["queries"])
            train_on_datasets(
                agent,
                dh,
                dl,
                steps_h=args.steps_h,
                steps_l=args.steps_l,
                batch_size=args.batch_size,
            )
            eval_r = evaluate_policy(
                env, agent, n_episodes=args.eval_episodes, seed_offset=20000
            )

        row = {
            "task": args.task,
            "variant": args.variant,
            "channel": args.channel,
            "lam": args.lam,
            "budget": args.budget,
            "seed": args.seed,
            "round": r,
            "cum_cost": round(cum_cost, 4),
            "queries": total_queries,
            "query_precision": r_stats["query_precision"],
            **eval_r,
            "dh_size": len(dh["target_idxs"]),
            "dl_size": len(dl["z_idxs"]),
        }
        rows.append(row)
        print(
            f"[Round {r}/{args.rounds}] cost={cum_cost:.1f} queries={total_queries} "
            f"prec={r_stats['query_precision']:.2f} success={eval_r['success']:.3f} "
            f"(sem_fail={eval_r['semantic_fail_rate']:.2f}, exec_fail={eval_r['exec_fail_rate']:.2f})"
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    agent.save(os.path.join(out_dir, "final_ckpt.pt"))

    env.close()
    if cf_env is not None:
        cf_env.close()


if __name__ == "__main__":
    main()
