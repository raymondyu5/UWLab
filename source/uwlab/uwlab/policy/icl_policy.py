"""Episodic in-context-learning student policy (port of ManiSkill's icl_episodic.ICLPolicy, vector feedback only).

pi(a | s, c_h): the context c_h is a set of tokens built from subsampled (s, a) steps of the h previous episodes of the same task plus one
feedback token per episode; a learned null token is always present so h = 0 works. The current state cross-attends to the memory produced by
a transformer encoder over the context tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ICLPolicy(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, y_dim, k: int, n_ctx_steps: int,
                 d_model: int = 128, n_heads: int = 4, n_ctx_layers: int = 2, n_dec_layers: int = 2, use_pcd: bool = False, f_dim: int = 0):
        """k: max number of context episodes; n_ctx_steps: context tokens (steps) per episode. y_dim: int (one vector feedback token per episode)
        or (L, D + 1): L frozen-LM feature vectors of size D per episode (language feedback), with the attention mask as the last channel."""
        super().__init__()
        d, H = d_model, n_heads
        self.s_dim, self.a_dim = s_dim, a_dim
        y_shape = tuple(y_dim) if isinstance(y_dim, (tuple, list)) else (y_dim,)
        self.lm = len(y_shape) == 2
        y_dim = y_shape[-1]
        for name, n in (("s", s_dim), ("a", a_dim), ("y", y_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(n))
            self.register_buffer(f"{name}_std", torch.ones(n))
        self.f_dim = f_dim  # dense per-step feedback appended to every context step token: (B, k, Tc, S + A + f_dim), standardised with fixed f_mean / f_std
        if f_dim:
            self.register_buffer("f_mean", torch.zeros(f_dim))
            self.register_buffer("f_std", torch.ones(f_dim))
        self.step_in = nn.Linear(s_dim + a_dim + f_dim, d)
        self.y_in = nn.Linear(y_dim - 1 if self.lm else y_dim, d)
        self.y_ln = nn.LayerNorm(y_dim - 1) if self.lm else None  # frozen LM features are not standardised; LayerNorm instead
        self.word_pos = nn.Embedding(y_shape[0], d) if self.lm else None
        self.ep_emb = nn.Embedding(k, d)
        self.pos_emb = nn.Embedding(n_ctx_steps, d)
        self.type_emb = nn.Embedding(3, d)  # 0 null, 1 step, 2 feedback
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.ctx_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True), n_ctx_layers, enable_nested_tensor=False)
        self.q_in = nn.Sequential(nn.Linear(s_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.use_pcd = use_pcd  # per-episode point cloud (B, P, 3, already standardised): PointNet -> one vector added to every query
        if use_pcd:
            self.pcd_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
            self.pcd_out = nn.Sequential(nn.Linear(256, d), nn.ReLU(), nn.Linear(d, d))
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, H, batch_first=True) for _ in range(n_dec_layers)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_dec_layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)) for _ in range(n_dec_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, a_dim))

    def set_norm(self, s: torch.Tensor, a: torch.Tensor, y: torch.Tensor):
        for name, x in (("s", s), ("a", a), ("y", y)):
            if name == "y" and self.lm:
                continue
            x = x.reshape(-1, x.shape[-1]).float()
            getattr(self, f"{name}_mean").copy_(x.mean(0))
            getattr(self, f"{name}_std").copy_(x.std(0).clamp_min(1e-3))

    def norm_a(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.a_mean) / self.a_std

    def encode_context(self, ctx_steps, ctx_y, n_eps, ctx_valid):
        """ctx_steps (B, k, Tc, S+A) raw; ctx_y (B, k, Y) raw; n_eps (B,) number of valid past episodes; ctx_valid (B, k, Tc) False for
        steps after the episode ended. Returns memory (B, 1 + k*(Tc+M), d), M feedback tokens per episode (1, or L for language), and a key padding mask (True = ignore)."""
        B, k, Tc, _ = ctx_steps.shape
        dev = ctx_steps.device
        s = (ctx_steps[..., : self.s_dim] - self.s_mean) / self.s_std
        a = (ctx_steps[..., self.s_dim:self.s_dim + self.a_dim] - self.a_mean) / self.a_std
        ep = self.ep_emb(torch.arange(k, device=dev))[None, :, None]  # (1,k,1,d)
        pos = self.pos_emb(torch.arange(Tc, device=dev))[None, None]  # (1,1,Tc,d)
        if self.f_dim:
            a = torch.cat([a, (ctx_steps[..., self.s_dim + self.a_dim:] - self.f_mean) / self.f_std], -1)
        step_tok = self.step_in(torch.cat([s, a], -1)) + ep + pos + self.type_emb.weight[1]
        if self.lm:  # (B,k,L,D+1) -> L feedback tokens per episode; padded word pieces are masked
            f = ctx_y.float()
            y_tok = (self.y_in(self.y_ln(f[..., :-1])) + self.word_pos(torch.arange(f.shape[2], device=dev))[None, None]
                     + ep + self.type_emb.weight[2])
            y_valid = f[..., -1] > 0.5
        else:
            y_tok = self.y_in((ctx_y - self.y_mean) / self.y_std)[:, :, None] + ep + self.type_emb.weight[2]  # (B,k,1,d)
            y_valid = torch.ones(B, k, 1, dtype=torch.bool, device=dev)
        tokens = torch.cat([step_tok, y_tok], dim=2).reshape(B, k * (Tc + y_tok.shape[2]), -1)
        ep_valid = (torch.arange(k, device=dev)[None] < n_eps[:, None])[:, :, None]
        valid = torch.cat([ctx_valid & ep_valid, y_valid & ep_valid], dim=2).reshape(B, -1)
        tokens = torch.cat([self.null.expand(B, 1, -1) + self.type_emb.weight[0], tokens], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), ~valid], dim=1)
        return self.ctx_enc(tokens, src_key_padding_mask=pad), pad

    def encode_pcd(self, pcd):
        """pcd (B, P, 3) -> (B, d)."""
        return self.pcd_out(self.pcd_mlp(pcd).max(1).values)

    def decode(self, s, memory, pad, normalized: bool = False, pcd_feat=None):
        """s (B, T, S) raw states -> actions (B, T, A): raw joint targets, or standardised if normalized. pcd_feat (B, d) from encode_pcd."""
        q = self.q_in((s - self.s_mean) / self.s_std)
        if pcd_feat is not None:
            q = q + pcd_feat[:, None]
        for attn, ln, mlp in zip(self.cross, self.ln_q, self.mlp):
            q = q + attn(ln(q), memory, memory, key_padding_mask=pad, need_weights=False)[0]
            q = q + mlp(q)
        out = self.head(q)
        return out if normalized else out * self.a_std + self.a_mean
