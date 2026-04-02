"""
Training Script — MedLLaMA
===========================
Trains the LLaMA-style transformer on diagnosis→prescription pairs.

Run:
    python train.py                    # default small config
    python train.py --config tiny      # fast CPU test
    python train.py --steps 2000       # custom steps

Features:
  • AdamW optimizer with cosine LR schedule + linear warmup
  • Gradient clipping
  • Perplexity tracking
  • Checkpoint saving / resuming
  • Validation loss monitoring
"""

import argparse
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import MedLLaMAConfig, tiny_config, small_config, base_config
from dataset import create_sample_dataset, get_dataloaders
from model import MedLLaMA
from tokenizer import MedTokenizer, build_char_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Learning Rate Schedule: Linear Warmup + Cosine Decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int) -> LambdaLR:
    """
    Cosine decay with linear warmup (LLaMA training schedule).
    
    lr(t) = 
      t / warmup_steps * lr_max              if t < warmup_steps
      lr_min + 0.5*(lr_max-lr_min)*(1+cos(π*t'/max_steps))  otherwise
    
    where lr_min = 0.1 * lr_max (LLaMA default)
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine  # decay to 10% of peak lr

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, cfg: MedLLaMAConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.step   = 0
        self.best_val_loss = float('inf')

        # Create output dirs
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # ── Data ──────────────────────────────────────────────────
        if not os.path.exists(cfg.data_path):
            print("No dataset found. Creating sample medical data...")
            create_sample_dataset(cfg.data_path)

        # ── Tokenizer ─────────────────────────────────────────────
        if os.path.exists(cfg.tokenizer_path):
            print(f"Loading tokenizer from {cfg.tokenizer_path}")
            self.tokenizer = MedTokenizer.load(cfg.tokenizer_path)
        else:
            print("Training tokenizer on medical corpus...")
            import json
            with open(cfg.data_path) as f:
                data = json.load(f)
            corpus = [d["diagnosis"] + " " + d["prescription"] for d in data]
            # For small datasets, use character-level fallback
            if len(corpus) < 100:
                self.tokenizer = build_char_tokenizer(
                    corpus + ["[DIAG] [SEP] [PRESC] [DRUG] [DOSE] [FREQ] [WARN] [END]"]
                )
            else:
                self.tokenizer = MedTokenizer.train(corpus, vocab_size=cfg.vocab_size)
            self.tokenizer.save(cfg.tokenizer_path)

        # Update vocab size from actual tokenizer
        cfg.vocab_size = self.tokenizer.vocab_size
        print(f"Tokenizer: {self.tokenizer}")

        # ── Model ─────────────────────────────────────────────────
        self.model = MedLLaMA(cfg).to(self.device)
        print(f"Model: {self.model}")
        print(f"Parameters: {self.model.num_parameters()/1e6:.2f}M")

        # ── Optimizer ─────────────────────────────────────────────
        # Separate weight decay for 2D params (weights) vs 1D (norms, biases)
        decay_params    = [p for n, p in self.model.named_parameters()
                          if p.dim() >= 2 and p.requires_grad]
        no_decay_params = [p for n, p in self.model.named_parameters()
                          if p.dim() < 2 and p.requires_grad]

        self.optimizer = AdamW([
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)  # LLaMA optimizer settings

        self.scheduler = get_lr_scheduler(self.optimizer, cfg.warmup_steps, cfg.max_steps)

        # ── Data loaders ──────────────────────────────────────────
        self.train_loader, self.val_loader = get_dataloaders(
            cfg.data_path, self.tokenizer, cfg.max_seq_len, cfg.batch_size
        )
        self._train_iter = iter(self.train_loader)

    def _next_batch(self):
        """Infinite training batch iterator."""
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    # ── Training Step ────────────────────────────────────────────

    def train_step(self) -> float:
        self.model.train()
        x, y = self._next_batch()
        x, y = x.to(self.device), y.to(self.device)

        # Forward + loss
        _, loss = self.model(x, targets=y)

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    # ── Validation ───────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(1, n_batches)

    # ── Checkpointing ─────────────────────────────────────────────

    def save(self, tag: str = "latest"):
        path = os.path.join(self.cfg.checkpoint_dir, f"medllama_{tag}.pt")
        torch.save({
            "step":       self.step,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "val_loss":   self.best_val_loss,
            "config":     self.cfg,
        }, path)
        print(f"  ✓ Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step           = ckpt["step"]
        self.best_val_loss  = ckpt.get("val_loss", float('inf'))
        print(f"Resumed from step {self.step}, val_loss={self.best_val_loss:.4f}")

    # ── Main Training Loop ────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  MedLLaMA Training")
        print(f"  Device: {self.device} | Steps: {cfg.max_steps}")
        print(f"  Batch: {cfg.batch_size} | LR: {cfg.lr} | Warmup: {cfg.warmup_steps}")
        print(f"{'='*60}\n")

        losses    = []
        t0        = time.time()

        for self.step in range(self.step + 1, cfg.max_steps + 1):
            loss = self.train_step()
            losses.append(loss)

            # ── Logging ──────────────────────────────────────────
            if self.step % cfg.log_every == 0:
                avg_loss = sum(losses[-cfg.log_every:]) / len(losses[-cfg.log_every:])
                perplexity = math.exp(min(avg_loss, 20))
                lr_now   = self.scheduler.get_last_lr()[0]
                dt       = (time.time() - t0) / cfg.log_every
                t0       = time.time()

                print(f"Step {self.step:5d}/{cfg.max_steps} | "
                      f"loss={avg_loss:.4f} | ppl={perplexity:.1f} | "
                      f"lr={lr_now:.2e} | {dt*1000:.0f}ms/step")

            # ── Validation & Checkpointing ────────────────────────
            if self.step % cfg.save_every == 0:
                val_loss = self.evaluate()
                val_ppl  = math.exp(min(val_loss, 20))
                print(f"\n  ▶ Validation — loss={val_loss:.4f} | ppl={val_ppl:.1f}")

                self.save("latest")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save("best")
                    print(f"  ★ New best model!")
                print()

        # Final validation
        val_loss = self.evaluate()
        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Final val loss: {val_loss:.4f} | ppl: {math.exp(min(val_loss,20)):.1f}")
        print(f"{'='*60}")
        self.save("final")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MedLLaMA")
    parser.add_argument("--config", choices=["tiny", "small", "base"], default="small")
    parser.add_argument("--steps",  type=int, default=None)
    parser.add_argument("--lr",     type=float, default=None)
    parser.add_argument("--batch",  type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg_map = {"tiny": tiny_config, "small": small_config, "base": base_config}
    cfg = cfg_map[args.config]()

    if args.steps:  cfg.max_steps = args.steps
    if args.lr:     cfg.lr        = args.lr
    if args.batch:  cfg.batch_size = args.batch
    if args.device: cfg.device    = args.device

    # Auto-select device
    if cfg.device == "cpu" and torch.cuda.is_available():
        cfg.device = "cuda"
        print("CUDA detected — using GPU")

    trainer = Trainer(cfg)

    if args.resume:
        trainer.load(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
