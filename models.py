"""Hierarchical policy: Subgoal Transformer pi_h + FiLM-CLIP Diffusion Policy pi_l."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs import NONE_SUBGOAL


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    alphas_bar = torch.cos((t + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-4, 0.999).float()


class TextEncoder:
    """Frozen CLIP text encoder with caching and deterministic offline fallback."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_path: str = "data/clip_cache.pt",
        mode: str = "clip",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.cache_path = cache_path
        self.mode = mode
        self.device = device
        self.emb_dim = 512
        self._cache: dict[str, torch.Tensor] = {}
        self._tokenizer = None
        self._model = None
        if cache_path and os.path.exists(cache_path):
            loaded = torch.load(cache_path, map_location="cpu", weights_only=True)
            self._cache = {k: v.float() for k, v in loaded.items()}

    def _lazy_load_clip(self) -> bool:
        if self._model is not None:
            return True
        if self.mode == "hash":
            return False
        try:
            from transformers import AutoTokenizer, CLIPTextModel

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = (
                CLIPTextModel.from_pretrained(self.model_name)
                .to(self.device)
                .eval()
                .requires_grad_(False)
            )
            return True
        except Exception:
            self.mode = "hash"
            return False

    def _hash_embed(self, text: str) -> torch.Tensor:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**31 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        vec = torch.randn(self.emb_dim, generator=gen)
        return F.normalize(vec, dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        missing = [t for t in set(texts) if t not in self._cache]
        if missing:
            if self._lazy_load_clip():
                toks = self._tokenizer(
                    missing, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                pooled = self._model(**toks).pooler_output
                normed = F.normalize(pooled.float(), dim=-1).cpu()
                for t, v in zip(missing, normed):
                    self._cache[t] = v
                if self.cache_path:
                    os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
                    torch.save(self._cache, self.cache_path)
            else:
                for t in missing:
                    self._cache[t] = self._hash_embed(t)
        return torch.stack([self._cache[t] for t in texts], dim=0).to(self.device)


class StateNormalizer:
    """Standard-score state normalizer serializable as plain tensors."""

    def __init__(self, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.mean = mean
        self.std = std

    def fit(self, states: torch.Tensor):
        self.mean = states.mean(dim=0)
        self.std = states.std(dim=0).clamp(min=1e-4)

    def normalize(self, s: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return s
        return (s - self.mean.to(s.device)) / self.std.to(s.device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    def load_state_dict(self, d: dict[str, torch.Tensor]):
        self.mean = d["mean"].float()
        self.std = d["std"].float()


class HighLevelTransformer(nn.Module):
    """Categorical subgoal policy pi_h(z | s, g, z_prev) over finite Z."""

    def __init__(
        self,
        state_dim: int,
        text_dim: int,
        num_subgoals: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, d_model)
        self.instr_proj = nn.Linear(text_dim, d_model)
        # +1 index reserved for NONE_SUBGOAL at episode start.
        self.prev_emb = nn.Embedding(num_subgoals + 1, d_model)
        self.role_emb = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_subgoals))

    def forward(
        self, s_norm: torch.Tensor, g_emb: torch.Tensor, prev_idx: torch.Tensor
    ) -> torch.Tensor:
        tokens = torch.stack(
            [self.state_proj(s_norm), self.instr_proj(g_emb), self.prev_emb(prev_idx)],
            dim=1,
        )
        h = self.encoder(tokens + self.role_emb)
        return self.head(h.mean(dim=1))


class FiLMResidualBlock(nn.Module):
    """Feature-wise Linear Modulation block conditioned on (s, CLIP(z), t)."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + gamma) + beta
        h = F.mish(self.fc1(h))
        return x + self.fc2(h)


class FiLMDiffusionNet(nn.Module):
    """Noise-prediction network eps_theta(a_t, t | s, CLIP(z)) over action chunks."""

    def __init__(
        self,
        state_dim: int,
        text_dim: int,
        action_dim: int = 4,
        horizon: int = 8,
        hidden_dim: int = 256,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        flat_dim = horizon * action_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2), nn.Mish()
        )
        self.subgoal_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 2), nn.Mish()
        )
        cond_dim = (hidden_dim // 2) * 3
        self.in_proj = nn.Linear(flat_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, cond_dim) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, flat_dim)
        )

    def _sinusoidal_t(self, t: torch.Tensor) -> torch.Tensor:
        half = 32
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(
        self,
        noisy_chunk: torch.Tensor,
        t: torch.Tensor,
        s_norm: torch.Tensor,
        z_emb: torch.Tensor,
    ) -> torch.Tensor:
        b = noisy_chunk.shape[0]
        cond = torch.cat(
            [
                self.state_mlp(s_norm),
                self.subgoal_mlp(z_emb),
                self.time_mlp(self._sinusoidal_t(t)),
            ],
            dim=-1,
        )
        h = self.in_proj(noisy_chunk.reshape(b, -1))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(h).reshape(b, self.horizon, self.action_dim)


class HierarchicalAgent:
    """Two-level policy (pi_h, pi_l) with DDPM training, sampling, and Diff-DAgger score."""

    def __init__(
        self,
        state_dim: int,
        vocab: list[str],
        action_dim: int = 4,
        horizon: int = 8,
        exec_steps: int = 4,
        diffusion_steps: int = 20,
        lr: float = 3e-4,
        text_mode: str = "clip",
        cache_path: str = "data/clip_cache.pt",
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.vocab = list(vocab)
        self.subgoal_to_idx = {z: i for i, z in enumerate(self.vocab)}
        self.none_idx = len(self.vocab)
        self.action_dim = action_dim
        self.horizon = horizon
        self.exec_steps = exec_steps
        self.diffusion_steps = diffusion_steps
        self.lr = lr
        self.device = device

        self.text_encoder = TextEncoder(
            cache_path=cache_path, mode=text_mode, device=device
        )
        self.normalizer = StateNormalizer()
        self.pi_h = HighLevelTransformer(
            state_dim, self.text_encoder.emb_dim, len(self.vocab)
        ).to(device)
        self.pi_l = FiLMDiffusionNet(
            state_dim, self.text_encoder.emb_dim, action_dim, horizon
        ).to(device)
        self.pi_l_ema = deepcopy(self.pi_l).eval().requires_grad_(False)
        self.ema_rate = 0.995

        self.opt_h = torch.optim.AdamW(self.pi_h.parameters(), lr=lr, weight_decay=1e-4)
        self.opt_l = torch.optim.AdamW(self.pi_l.parameters(), lr=lr, weight_decay=1e-4)

        betas = cosine_beta_schedule(diffusion_steps).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.vocab_embs = self.text_encoder.encode(self.vocab)

    def reset_models(self):
        """Reinitialize weights before retraining on aggregated DAgger datasets."""
        self.pi_h = HighLevelTransformer(
            self.state_dim, self.text_encoder.emb_dim, len(self.vocab)
        ).to(self.device)
        self.pi_l = FiLMDiffusionNet(
            self.state_dim, self.text_encoder.emb_dim, self.action_dim, self.horizon
        ).to(self.device)
        self.pi_l_ema = deepcopy(self.pi_l).eval().requires_grad_(False)
        self.opt_h = torch.optim.AdamW(
            self.pi_h.parameters(), lr=self.lr, weight_decay=1e-4
        )
        self.opt_l = torch.optim.AdamW(
            self.pi_l.parameters(), lr=self.lr, weight_decay=1e-4
        )

    def encode_prev_subgoal(self, z_prev: str) -> int:
        if z_prev == NONE_SUBGOAL:
            return self.none_idx
        return self.subgoal_to_idx[z_prev]

    def update_high(
        self,
        states: torch.Tensor,
        g_embs: torch.Tensor,
        prev_idxs: torch.Tensor,
        target_idxs: torch.Tensor,
    ) -> float:
        self.pi_h.train()
        s_norm = self.normalizer.normalize(states.to(self.device))
        logits = self.pi_h(s_norm, g_embs.to(self.device), prev_idxs.to(self.device))
        loss = F.cross_entropy(logits, target_idxs.to(self.device))
        self.opt_h.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.pi_h.parameters(), 1.0)
        self.opt_h.step()
        return float(loss.item())

    def update_low(
        self,
        states: torch.Tensor,
        z_idxs: torch.Tensor,
        action_chunks: torch.Tensor,
    ) -> float:
        self.pi_l.train()
        s_norm = self.normalizer.normalize(states.to(self.device))
        z_emb = self.vocab_embs[z_idxs.to(self.device)]
        x0 = action_chunks.to(self.device).clamp(-1.0, 1.0)
        b = x0.shape[0]
        t = torch.randint(0, self.diffusion_steps, (b,), device=self.device)
        eps = torch.randn_like(x0)
        ab = self.alphas_bar[t].view(b, 1, 1)
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps
        pred_eps = self.pi_l(xt, t, s_norm, z_emb)
        loss = F.mse_loss(pred_eps, eps)
        self.opt_l.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.pi_l.parameters(), 1.0)
        self.opt_l.step()
        with torch.no_grad():
            for p, p_ema in zip(self.pi_l.parameters(), self.pi_l_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1.0 - self.ema_rate)
        return float(loss.item())

    @torch.no_grad()
    def predict_subgoal(
        self, state_1d: torch.Tensor, instruction: str, prev_subgoal: str
    ) -> tuple[str, float]:
        self.pi_h.eval()
        s_norm = self.normalizer.normalize(state_1d.unsqueeze(0).to(self.device))
        g_emb = self.text_encoder.encode([instruction])
        prev_t = torch.tensor(
            [self.encode_prev_subgoal(prev_subgoal)],
            dtype=torch.long,
            device=self.device,
        )
        logits = self.pi_h(s_norm, g_emb, prev_t)[0]
        probs = F.softmax(logits, dim=-1)
        entropy = float(-(probs * (probs + 1e-8).log()).sum().item())
        idx = int(probs.argmax().item())
        return self.vocab[idx], entropy

    @torch.no_grad()
    def sample_chunk(self, state_1d: torch.Tensor, subgoal: str) -> torch.Tensor:
        self.pi_l_ema.eval()
        s_norm = self.normalizer.normalize(state_1d.unsqueeze(0).to(self.device))
        z_emb = self.vocab_embs[self.subgoal_to_idx[subgoal]].unsqueeze(0)
        xt = torch.randn((1, self.horizon, self.action_dim), device=self.device)
        for step in range(self.diffusion_steps - 1, -1, -1):
            t_b = torch.tensor([step], dtype=torch.long, device=self.device)
            eps = self.pi_l_ema(xt, t_b, s_norm, z_emb)
            alpha = self.alphas[step]
            ab = self.alphas_bar[step]
            mean = (xt - ((1.0 - alpha) / (1.0 - ab).sqrt()) * eps) / alpha.sqrt()
            if step > 0:
                xt = mean + self.betas[step].sqrt() * torch.randn_like(xt)
            else:
                xt = mean
            xt = xt.clamp(-1.0, 1.0)
        return xt[0]

    @torch.no_grad()
    def diffusion_loss_score(
        self,
        state_1d: torch.Tensor,
        subgoal: str,
        generated_chunk: torch.Tensor,
        n_mc: int = 8,
    ) -> float:
        """Diff-DAgger OOD uncertainty score u_l(s) on generated action chunk."""
        self.pi_l_ema.eval()
        s_norm = self.normalizer.normalize(
            state_1d.unsqueeze(0).to(self.device)
        ).expand(n_mc, -1)
        z_emb = (
            self.vocab_embs[self.subgoal_to_idx[subgoal]]
            .unsqueeze(0)
            .expand(n_mc, -1)
        )
        x0 = generated_chunk.unsqueeze(0).to(self.device).expand(n_mc, -1, -1)
        # Stratified diffusion timesteps + fixed-seed noise for low-variance scoring.
        t = torch.linspace(
            1, self.diffusion_steps - 1, n_mc, device=self.device
        ).long()
        gen = torch.Generator(device=self.device).manual_seed(0)
        eps = torch.randn(x0.shape, generator=gen, device=self.device)
        ab = self.alphas_bar[t].view(n_mc, 1, 1)
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps
        pred = self.pi_l_ema(xt, t, s_norm, z_emb)
        return float(F.mse_loss(pred, eps).item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "vocab": self.vocab,
                "action_dim": self.action_dim,
                "horizon": self.horizon,
                "exec_steps": self.exec_steps,
                "diffusion_steps": self.diffusion_steps,
                "normalizer": self.normalizer.state_dict(),
                "pi_h": self.pi_h.state_dict(),
                "pi_l": self.pi_l.state_dict(),
                "pi_l_ema": self.pi_l_ema.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        text_mode: str = "clip",
        cache_path: str = "data/clip_cache.pt",
        device: str = "cpu",
    ) -> "HierarchicalAgent":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        agent = cls(
            state_dim=int(ckpt["state_dim"]),
            vocab=list(ckpt["vocab"]),
            action_dim=int(ckpt["action_dim"]),
            horizon=int(ckpt["horizon"]),
            exec_steps=int(ckpt["exec_steps"]),
            diffusion_steps=int(ckpt["diffusion_steps"]),
            text_mode=text_mode,
            cache_path=cache_path,
            device=device,
        )
        agent.normalizer.load_state_dict(ckpt["normalizer"])
        agent.pi_h.load_state_dict(ckpt["pi_h"])
        agent.pi_l.load_state_dict(ckpt["pi_l"])
        agent.pi_l_ema.load_state_dict(ckpt["pi_l_ema"])
        return agent
