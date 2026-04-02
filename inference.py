"""
Inference — MedLLaMA
======================
Generate prescriptions from medical diagnoses.

Usage:
    python inference.py --diagnosis "Patient 45M, fever 102F, productive cough..."

Or in Python:
    from inference import MedGenerator
    gen = MedGenerator.from_checkpoint("checkpoints/medllama_best.pt")
    result = gen.generate("Patient 35F with UTI symptoms...")
    print(result)
"""

import argparse
import math
import re
from typing import List, Optional

import torch
import torch.nn.functional as F

from config import MedLLaMAConfig
from model import MedLLaMA
from tokenizer import MedTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Sampling Strategies
# ─────────────────────────────────────────────────────────────────────────────

def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out all logits except the top-k highest."""
    if top_k == 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    threshold  = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float('-inf'))


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: keep smallest set of tokens
    whose cumulative probability ≥ top_p.
    """
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens after cumulative prob exceeds top_p
    remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[remove] = float('-inf')

    # Scatter back to original order
    return logits.scatter(-1, sorted_idx, sorted_logits)


def apply_repetition_penalty(logits: torch.Tensor,
                              generated_ids: List[int],
                              penalty: float) -> torch.Tensor:
    """
    Reduce probability of recently generated tokens.
    penalty > 1.0 discourages repetition.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    for token_id in set(generated_ids[-64:]):  # only look back 64 tokens
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class MedGenerator:
    """
    Autoregressive text generation for prescription synthesis.
    
    Given a diagnosis, generates a structured prescription using:
    - Temperature scaling
    - Top-k + Top-p (nucleus) sampling
    - Repetition penalty
    """

    def __init__(self, model: MedLLaMA, tokenizer: MedTokenizer,
                 device: str = "cpu"):
        self.model     = model.eval()
        self.tokenizer = tokenizer
        self.device    = torch.device(device)
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "MedGenerator":
        """Load model and tokenizer from a checkpoint file."""
        ckpt = torch.load(path, map_location=device)
        cfg  = ckpt["config"]
        model = MedLLaMA(cfg)
        model.load_state_dict(ckpt["model"])

        tok_path = cfg.tokenizer_path
        tokenizer = MedTokenizer.load(tok_path)
        print(f"Loaded MedLLaMA from {path} (step {ckpt['step']})")
        return cls(model, tokenizer, device)

    @torch.no_grad()
    def generate(self, diagnosis: str,
                 max_new_tokens: int = 200,
                 temperature: float = 0.7,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1,
                 greedy: bool = False) -> str:
        """
        Generate a prescription from a diagnosis text.
        
        Args:
            diagnosis:          Clinical diagnosis string
            max_new_tokens:     Maximum tokens to generate
            temperature:        Sampling temperature (lower = more conservative)
            top_k:              Top-k sampling (0 = disabled)
            top_p:              Nucleus sampling threshold
            repetition_penalty: > 1.0 discourages repeating tokens
            greedy:             If True, use greedy decoding (most likely token)
        
        Returns:
            Formatted prescription string
        """
        cfg = self.model.cfg

        # Build the prompt
        prompt = f"[DIAG] {diagnosis} [SEP] [PRESC]"
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = input_ids[:cfg.max_seq_len - max_new_tokens]  # leave room

        generated  = list(input_ids)
        prompt_len = len(generated)

        eos_id = self.tokenizer.special_tokens["<eos>"]
        end_id = self.tokenizer.special_tokens.get("[END]", eos_id)

        for _ in range(max_new_tokens):
            # Trim to context window
            ctx = generated[-cfg.max_seq_len:]
            x   = torch.tensor([ctx], dtype=torch.long, device=self.device)

            logits, _ = self.model(x)
            next_logits = logits[0, -1, :].clone()  # (vocab_size,)

            if greedy:
                next_id = next_logits.argmax().item()
            else:
                # Apply repetition penalty
                next_logits = apply_repetition_penalty(
                    next_logits, generated[prompt_len:], repetition_penalty)

                # Temperature scaling
                if temperature != 1.0:
                    next_logits = next_logits / temperature

                # Top-k filtering
                next_logits = top_k_filtering(next_logits, top_k)

                # Top-p (nucleus) filtering
                next_logits = top_p_filtering(next_logits, top_p)

                # Sample
                probs   = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_id)

            # Stop at EOS or [END]
            if next_id in (eos_id, end_id):
                break

        # Decode only the generated part
        generated_ids = generated[prompt_len:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special=False)
        return self._format_prescription(raw_text)

    def _format_prescription(self, raw: str) -> str:
        """
        Post-process generated prescription into clean structured format.
        """
        # Remove EOS / padding tokens
        raw = re.sub(r'<[^>]+>', '', raw).strip()
        raw = raw.replace("[END]", "").replace("[PRESC]", "").strip()

        # Structure the output nicely
        lines = []
        lines.append("━" * 55)
        lines.append("  PRESCRIPTION")
        lines.append("━" * 55)

        # Parse drug entries
        parts = re.split(r'\[DRUG\]', raw)
        for part in parts:
            if not part.strip():
                continue
            drug  = re.search(r'^([^\[]+)', part)
            dose  = re.search(r'\[DOSE\]\s*([^\[]+)', part)
            freq  = re.search(r'\[FREQ\]\s*([^\[]+)', part)
            warn  = re.search(r'\[WARN\]\s*([^\[]+)', part)

            if drug:
                lines.append(f"\n  ℞  {drug.group(1).strip()}")
                if dose: lines.append(f"     Dose:      {dose.group(1).strip()}")
                if freq: lines.append(f"     Frequency: {freq.group(1).strip()}")
            if warn:
                lines.append(f"\n  ⚠  {warn.group(1).strip()}")

        if not any("℞" in l for l in lines):
            # Fallback: return raw text if parsing failed
            lines.append(raw)

        lines.append("\n" + "━" * 55)
        return "\n".join(lines)

    def generate_beam(self, diagnosis: str,
                      beam_width: int = 4,
                      max_new_tokens: int = 150) -> str:
        """
        Beam search decoding (more deterministic, higher quality).
        
        Maintains `beam_width` hypotheses and keeps the most probable.
        Slower but often produces better structured output.
        """
        cfg = self.model.cfg
        prompt = f"[DIAG] {diagnosis} [SEP] [PRESC]"
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        prompt_len = len(input_ids)

        eos_id = self.tokenizer.special_tokens["<eos>"]
        end_id = self.tokenizer.special_tokens.get("[END]", eos_id)

        # Beams: list of (score, token_ids)
        beams = [(0.0, list(input_ids))]
        completed = []

        for step in range(max_new_tokens):
            candidates = []

            for score, ids in beams:
                if ids[-1] in (eos_id, end_id):
                    completed.append((score / len(ids), ids))
                    continue

                ctx = ids[-cfg.max_seq_len:]
                x   = torch.tensor([ctx], dtype=torch.long, device=self.device)
                logits, _ = self.model(x)
                log_probs  = F.log_softmax(logits[0, -1], dim=-1)

                top_scores, top_ids = log_probs.topk(beam_width)
                for s, tid in zip(top_scores.tolist(), top_ids.tolist()):
                    candidates.append((score + s, ids + [tid]))

            if not candidates:
                break

            # Keep top beam_width
            candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            beams = candidates[:beam_width]

        # Add remaining beams to completed
        completed.extend([(s / len(ids), ids) for s, ids in beams])
        if not completed:
            return "Generation failed."

        # Best beam
        best = max(completed, key=lambda x: x[0])
        generated_ids = best[1][prompt_len:]
        raw = self.tokenizer.decode(generated_ids, skip_special=False)
        return self._format_prescription(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Demo
# ─────────────────────────────────────────────────────────────────────────────

def interactive_demo(generator: MedGenerator):
    """Command-line interactive prescription generation."""
    print("\n" + "═" * 60)
    print("  MedLLaMA — Interactive Prescription Generator")
    print("  Type 'quit' to exit | 'beam' for beam search mode")
    print("═" * 60 + "\n")

    use_beam = False
    while True:
        diagnosis = input("Enter diagnosis: ").strip()
        if not diagnosis:
            continue
        if diagnosis.lower() == "quit":
            break
        if diagnosis.lower() == "beam":
            use_beam = not use_beam
            print(f"Beam search: {'ON' if use_beam else 'OFF'}")
            continue

        print("\nGenerating prescription...\n")
        if use_beam:
            result = generator.generate_beam(diagnosis)
        else:
            result = generator.generate(diagnosis)
        print(result)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedLLaMA Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/medllama_best.pt")
    parser.add_argument("--diagnosis",  type=str, default=None,
                        help="Diagnosis text (if None, enter interactive mode)")
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--temp",       type=float, default=0.7)
    parser.add_argument("--top_k",      type=int,   default=50)
    parser.add_argument("--top_p",      type=float, default=0.9)
    parser.add_argument("--beam",       action="store_true",
                        help="Use beam search instead of sampling")
    parser.add_argument("--max_tokens", type=int,   default=200)
    args = parser.parse_args()

    generator = MedGenerator.from_checkpoint(args.checkpoint, args.device)

    if args.diagnosis:
        if args.beam:
            result = generator.generate_beam(args.diagnosis, max_new_tokens=args.max_tokens)
        else:
            result = generator.generate(
                args.diagnosis,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        print(result)
    else:
        interactive_demo(generator)


if __name__ == "__main__":
    main()
