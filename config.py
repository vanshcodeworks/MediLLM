"""
MedLLaMA Configuration
======================
LLaMA-inspired transformer for Diagnosis → Prescription generation.
Adjust model_dim / n_layers / n_heads to scale up or down.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MedLLaMAConfig:
    # ── Vocabulary ──────────────────────────────────────────────
    vocab_size: int = 8000          # BPE vocab (medical corpus)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # ── Architecture ────────────────────────────────────────────
    model_dim: int = 512            # d_model  (LLaMA calls it `dim`)
    n_layers: int = 8               # transformer blocks
    n_heads: int = 8                # query heads
    n_kv_heads: int = 2            # key/value heads (GQA: n_heads / n_kv_heads = 4)
    head_dim: int = 64             # model_dim // n_heads
    ffn_hidden_dim: int = 1376     # SwiGLU hidden (≈ 2.7 × model_dim, then / 2 for gating)
    max_seq_len: int = 512

    # ── Normalization ───────────────────────────────────────────
    rms_norm_eps: float = 1e-5

    # ── RoPE ────────────────────────────────────────────────────
    rope_theta: float = 10_000.0

    # ── Dropout / Regularisation ────────────────────────────────
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # ── Training ────────────────────────────────────────────────
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 5000
    batch_size: int = 8
    grad_clip: float = 1.0
    device: str = "cuda"            # set "cuda" if GPU available

    # ── Generation ──────────────────────────────────────────────
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # ── Paths ───────────────────────────────────────────────────
    data_path: str = "data/medical_data.json"
    tokenizer_path: str = "checkpoints/tokenizer.json"
    checkpoint_dir: str = "checkpoints/"
    log_every: int = 50
    save_every: int = 500

    def __post_init__(self):
        assert self.model_dim % self.n_heads == 0, "model_dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = self.model_dim // self.n_heads


# Pre-built size configs (like LLaMA naming)
def tiny_config() -> MedLLaMAConfig:
    """~7M params — fast experiments / CPU"""
    return MedLLaMAConfig(model_dim=256, n_layers=4, n_heads=4, n_kv_heads=2,
                          ffn_hidden_dim=688, max_seq_len=256)


def small_config() -> MedLLaMAConfig:
    """~50M params — single GPU training"""
    return MedLLaMAConfig(model_dim=512, n_layers=8, n_heads=8, n_kv_heads=2,
                          ffn_hidden_dim=1376, max_seq_len=512)


def base_config() -> MedLLaMAConfig:
    """~125M params — research-grade"""
    return MedLLaMAConfig(model_dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
                          ffn_hidden_dim=2048, max_seq_len=1024, vocab_size=16_000)
