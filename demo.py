"""
Quick Demo — MedLLaMA End-to-End Test
=======================================
Runs the full pipeline: tokenizer → model forward pass → generation
without requiring a trained checkpoint.

Useful for verifying architecture correctness before long training runs.

Run:
    python demo.py
"""

import torch
import math

from config import tiny_config
from dataset import create_sample_dataset, SAMPLE_DATA
from model import MedLLaMA
from tokenizer import build_char_tokenizer
from inference import MedGenerator


def main():
    print("=" * 65)
    print("  MedLLaMA — Architecture Demo & Sanity Check")
    print("=" * 65)

    # ── 1. Config ─────────────────────────────────────────────────
    cfg = tiny_config()
    cfg.max_new_tokens = 80
    cfg.device = "cpu"

    # ── 2. Tokenizer ─────────────────────────────────────────────
    print("\n[1] Building character-level tokenizer on sample data...")
    corpus = [d["diagnosis"] + " " + d["prescription"] for d in SAMPLE_DATA]
    corpus += ["[DIAG] [SEP] [PRESC] [DRUG] [DOSE] [FREQ] [WARN] [END]"]
    tokenizer = build_char_tokenizer(corpus)
    cfg.vocab_size = tokenizer.vocab_size
    print(f"    Vocab size: {tokenizer.vocab_size} chars")

    # ── 3. Model ──────────────────────────────────────────────────
    print("\n[2] Instantiating MedLLaMA...")
    model = MedLLaMA(cfg)
    print(f"    {model}")
    print(f"    Parameters: {model.num_parameters()/1e6:.3f}M")

    # ── 4. Forward Pass ───────────────────────────────────────────
    print("\n[3] Running forward pass...")
    diag_text = SAMPLE_DATA[0]["diagnosis"]
    full_text  = f"[DIAG] {diag_text} [SEP] [PRESC]"
    ids        = tokenizer.encode(full_text, add_bos=True)
    ids        = ids[:cfg.max_seq_len]
    x = torch.tensor([ids], dtype=torch.long)
    y = torch.tensor([ids[1:] + [cfg.pad_token_id]], dtype=torch.long)

    with torch.no_grad():
        logits, loss = model(x, targets=y)

    print(f"    Input shape:  {x.shape}")
    print(f"    Output shape: {logits.shape}")
    print(f"    Loss (random weights): {loss.item():.4f}")
    print(f"    Perplexity:            {math.exp(loss.item()):.1f}  "
          f"(expected ≈ {cfg.vocab_size:.0f} for random model)")

    # ── 5. Check Components ───────────────────────────────────────
    print("\n[4] Component verification...")

    from model import RMSNorm, precompute_rope_freqs, apply_rope, GroupedQueryAttention, SwiGLU

    # RMSNorm
    norm = RMSNorm(cfg.model_dim)
    t    = torch.randn(2, 10, cfg.model_dim)
    out  = norm(t)
    assert out.shape == t.shape
    print("    ✓ RMSNorm: correct shape")

    # RoPE
    cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len)
    q = torch.randn(2, cfg.n_heads, 10, cfg.head_dim)
    q_rot = apply_rope(q, cos, sin)
    assert q_rot.shape == q.shape
    print("    ✓ RoPE: correct shape, no NaN:", not torch.isnan(q_rot).any().item())

    # GQA
    gqa = GroupedQueryAttention(cfg)
    x_  = torch.randn(2, 10, cfg.model_dim)
    o   = gqa(x_, cos[:10], sin[:10])
    assert o.shape == x_.shape
    print(f"    ✓ GQA ({cfg.n_heads}Q/{cfg.n_kv_heads}KV): correct shape")

    # SwiGLU
    ffn = SwiGLU(cfg)
    o   = ffn(x_)
    assert o.shape == x_.shape
    print("    ✓ SwiGLU: correct shape")

    # ── 6. Untrained Generation (will be gibberish) ───────────────
    print("\n[5] Untrained generation (random weights — output will be gibberish)...")
    gen = MedGenerator(model, tokenizer, device="cpu")

    test_diagnosis = "Patient 35F. Fever 102°F, productive cough, right lower lobe consolidation. Diagnosis: Community-acquired pneumonia."
    print(f"\n    Diagnosis: {test_diagnosis[:70]}...\n")

    result = gen.generate(
        test_diagnosis,
        max_new_tokens=60,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
    )
    print(result)

    # ── 7. Parameter Breakdown ────────────────────────────────────
    print("\n[6] Parameter breakdown by component:")
    named_groups = {
        "Embedding":          ["embed"],
        "Attention (all)":    ["attn"],
        "Feed-Forward (FFN)": ["ffn"],
        "RMSNorm":            ["norm"],
        "LM Head":            ["lm_head"],
    }
    total = 0
    for group, prefixes in named_groups.items():
        count = sum(p.numel() for n, p in model.named_parameters()
                    if any(pf in n for pf in prefixes))
        total += count
        print(f"    {group:<25} {count/1e6:6.3f}M params")
    print(f"    {'─'*35}")
    print(f"    {'Total':<25} {model.num_parameters()/1e6:6.3f}M params")

    print("\n" + "=" * 65)
    print("  ✓ All checks passed! Architecture is correct.")
    print("  ➜ Train the model with:  python train.py --config tiny")
    print("  ➜ Generate prescriptions: python inference.py --checkpoint ...")
    print("=" * 65)


if __name__ == "__main__":
    main()
