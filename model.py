"""
MedLLaMA — Core Architecture
==============================
LLaMA-inspired Transformer built from scratch.

Key innovations over vanilla Transformer:
  1. RMSNorm          — faster than LayerNorm, no mean subtraction
  2. RoPE             — Rotary Positional Embeddings (relative positions)
  3. SwiGLU           — Swish-Gated Linear Unit FFN (outperforms ReLU/GELU)
  4. Grouped Query Attention (GQA) — fewer KV heads → less memory, faster inference
  5. Pre-normalization — norm before attention/FFN (more stable training)
  6. No bias terms    — following LLaMA design
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MedLLaMAConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. RMS Normalization
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler than LayerNorm — only rescales, no recentering.
    Formula: x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x²) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # learnable scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.gamma


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rotary Positional Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_seq_len: int,
                           theta: float = 10_000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the sin/cos tables for RoPE.
    
    For each position m and dimension pair (2i, 2i+1):
      theta_i = 1 / (theta ^ (2i / head_dim))
      rotation angle = m * theta_i
    
    Returns cos and sin of shape (max_seq_len, head_dim/2).
    """
    # Frequency for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    # Outer product: (max_seq_len, head_dim/2)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding to query or key tensor.
    
    x shape: (B, n_heads, T, head_dim)
    Rotates adjacent pairs (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
    """
    B, H, T, D = x.shape
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    # Split into pairs
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices

    # Rotate
    x_rot_even = x1 * cos - x2 * sin
    x_rot_odd  = x1 * sin + x2 * cos

    # Interleave back
    rotated = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (B, H, T, D/2, 2)
    return rotated.flatten(-2)  # (B, H, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Grouped Query Attention (GQA)
# ─────────────────────────────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023).
    
    Standard MHA: n_heads Q, K, V each
    GQA:          n_heads Q, n_kv_heads K, V  (n_heads > n_kv_heads)
    MQA:          n_heads Q, 1 K, 1 V  (extreme case)
    
    Each group of (n_heads // n_kv_heads) query heads shares 1 K/V head.
    This drastically reduces the KV-cache size at inference time.
    """

    def __init__(self, cfg: MedLLaMAConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.head_dim
        self.scale      = cfg.head_dim ** -0.5
        self.groups     = cfg.n_heads // cfg.n_kv_heads  # queries per KV head

        # Projections — no bias (LLaMA style)
        self.q_proj  = nn.Linear(cfg.model_dim, cfg.n_heads    * cfg.head_dim, bias=False)
        self.k_proj  = nn.Linear(cfg.model_dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj  = nn.Linear(cfg.model_dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj  = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.model_dim,   bias=False)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        # ── Project ──────────────────────────────────────────────
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_heads, T, head_dim)
        # k: (B, n_kv_heads, T, head_dim)

        # ── Apply RoPE to Q and K ─────────────────────────────────
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ── Expand K,V for grouped queries ────────────────────────
        # Repeat each KV head `groups` times to match query heads
        k = k.repeat_interleave(self.groups, dim=1)  # (B, n_heads, T, head_dim)
        v = v.repeat_interleave(self.groups, dim=1)

        # ── Scaled Dot-Product Attention ──────────────────────────
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, T)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)          # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network (Noam Shazeer, 2020).
    
    Standard FFN:  FFN(x) = Linear(GELU(Linear(x)))
    SwiGLU:        FFN(x) = Linear(SiLU(Linear(x)) ⊗ Linear(x))
    
    Uses a gating mechanism: the gate (SiLU activated) modulates the value branch.
    Empirically outperforms GELU/ReLU FFNs on language tasks.
    
    Note: hidden_dim is typically 8/3 * model_dim, rounded to multiple of 256.
    Two up-projections (gate + value) + one down-projection.
    """

    def __init__(self, cfg: MedLLaMAConfig):
        super().__init__()
        hidden = cfg.ffn_hidden_dim

        self.gate_proj = nn.Linear(cfg.model_dim, hidden, bias=False)  # W_gate
        self.up_proj   = nn.Linear(cfg.model_dim, hidden, bias=False)  # W_v
        self.down_proj = nn.Linear(hidden, cfg.model_dim, bias=False)  # W_down
        self.dropout   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate branch: SiLU activation
        gate = F.silu(self.gate_proj(x))
        # Value branch: linear
        value = self.up_proj(x)
        # Hadamard product (gating)
        return self.dropout(self.down_proj(gate * value))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Transformer Block (Pre-Norm)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One LLaMA-style decoder block:
    
      x → RMSNorm → GQA → +x → RMSNorm → SwiGLU → +x
    
    Pre-normalization (norm before sublayer) is more stable than
    post-normalization for deep networks.
    """

    def __init__(self, cfg: MedLLaMAConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.model_dim, eps=cfg.rms_norm_eps)
        self.attn  = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.model_dim, eps=cfg.rms_norm_eps)
        self.ffn   = SwiGLU(cfg)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full MedLLaMA Model
# ─────────────────────────────────────────────────────────────────────────────

class MedLLaMA(nn.Module):
    """
    MedLLaMA: LLaMA-style causal language model for
    medical Diagnosis → Prescription generation.
    
    Input:  token sequence  [DIAG] fever, cough ... [SEP]
    Output: logits over vocabulary at each position
    
    Architecture summary
    --------------------
    Embedding → N × TransformerBlock → RMSNorm → LM Head
    
    Parameter count (small config ~50M):
      Embedding:  vocab_size × model_dim
      Per block:  attn (Q+K+V+O) + ffn (gate+up+down) + 2 norms
      Head:       model_dim × vocab_size  (shared weights with embedding)
    """

    def __init__(self, cfg: MedLLaMAConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding table
        self.embed = nn.Embedding(cfg.vocab_size, cfg.model_dim,
                                  padding_idx=cfg.pad_token_id)

        # Stacked transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # Final normalisation before LM head
        self.norm_out = RMSNorm(cfg.model_dim, eps=cfg.rms_norm_eps)

        # Language model head (no bias, tied weights with embedding)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)

        # Weight tying: embedding ↔ LM head (reduces params, improves quality)
        self.lm_head.weight = self.embed.weight

        # Precompute RoPE tables
        cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Causal mask (lower-triangular)
        self._causal_mask_cache: dict = {}

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        LLaMA-style weight initialization:
        - Linear layers: N(0, 0.02)
        - Embeddings: N(0, 0.02)
        - Output projections scaled by 1/sqrt(2 * n_layers)
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        # Scale residual projections (attn output + ffn down) for stable training
        for name, param in module.named_parameters():
            if name in ("o_proj.weight", "down_proj.weight"):
                nn.init.normal_(param, mean=0.0,
                                 std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular causal mask of shape (1, 1, T, T)."""
        if T not in self._causal_mask_cache:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
            self._causal_mask_cache[T] = mask.unsqueeze(0).unsqueeze(0)
        return self._causal_mask_cache[T]

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) token IDs
            targets:   (B, T) shifted labels for training (None at inference)
        
        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (if targets provided)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ── Embed tokens ───────────────────────────────────────
        x = self.embed(input_ids)  # (B, T, D)

        # ── RoPE cache slices ──────────────────────────────────
        cos = self.rope_cos[:T].to(device)
        sin = self.rope_sin[:T].to(device)

        # ── Causal mask ────────────────────────────────────────
        mask = self._get_causal_mask(T, device)

        # ── Transformer blocks ─────────────────────────────────
        for layer in self.layers:
            x = layer(x, cos, sin, mask)

        # ── Final norm + LM head ───────────────────────────────
        x = self.norm_out(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # ── Loss (training) ────────────────────────────────────
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            # Ignore pad tokens in loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id
            )

        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"MedLLaMA("
                f"layers={self.cfg.n_layers}, "
                f"dim={self.cfg.model_dim}, "
                f"heads={self.cfg.n_heads}(Q)/{self.cfg.n_kv_heads}(KV), "
                f"params={self.num_parameters()/1e6:.1f}M)")
