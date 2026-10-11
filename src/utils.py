"""
src/utils.py
Shared utilities for all notebooks in the prompt-compression project.
Every notebook imports from here instead of redefining functions.
"""

import torch
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DistilBertTokenizerFast,
)

# ── Tokenizers (loaded once, reused everywhere) ───────────────────────────────

_gpt2_tokenizer: GPT2TokenizerFast | None = None
_gpt2_model:     GPT2LMHeadModel   | None = None
_distilbert_tokenizer: DistilBertTokenizerFast | None = None


def get_gpt2():
    global _gpt2_tokenizer, _gpt2_model
    if _gpt2_tokenizer is None:
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _gpt2_model     = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    return _gpt2_tokenizer, _gpt2_model


def get_distilbert_tokenizer() -> DistilBertTokenizerFast:
    global _distilbert_tokenizer
    if _distilbert_tokenizer is None:
        _distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
    return _distilbert_tokenizer


# ── Perplexity scoring (GPT-2 surprisal) ─────────────────────────────────────

def score_tokens_perplexity(text: str) -> dict:
    """
    Score each token in `text` by its surprisal under GPT-2.
    Returns a dict with keys:
        tokens     : list[str]   — decoded subword tokens (positions 1..n)
        surprisal  : list[float] — per-token negative log-probability
    """
    gpt2_tok, gpt2_model = get_gpt2()
    ids = gpt2_tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = gpt2_model(ids).logits
    log_probs  = F.log_softmax(logits, dim=-1)
    token_lp   = log_probs[0, :-1].gather(1, ids[0, 1:].unsqueeze(1)).squeeze()
    surprisal  = -token_lp
    tokens     = [gpt2_tok.decode([t]) for t in ids[0, 1:]]
    return {"tokens": tokens, "surprisal": surprisal.tolist()}


def build_training_sample(context: str) -> dict:
    """
    Convert a raw text string into a training sample dict:
        text    : original string
        tokens  : GPT-2 subword tokens
        targets : min-max normalised surprisal scores in [0, 1]
    Used in 02_dataset.ipynb to build train_samples.json.
    """
    result = score_tokens_perplexity(context)
    scores = torch.tensor(result["surprisal"])
    targets = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return {
        "text":    context,
        "tokens":  result["tokens"],
        "targets": targets.tolist(),
    }


# ── Baseline compression methods ──────────────────────────────────────────────

import random


def compress_random(text: str, ratio: float) -> str:
    """Drop tokens uniformly at random, keeping `ratio` fraction."""
    gpt2_tok, _ = get_gpt2()
    tokens = gpt2_tok.tokenize(text)
    k      = max(1, int(len(tokens) * ratio))
    kept   = sorted(random.sample(range(len(tokens)), k))
    return gpt2_tok.convert_tokens_to_string([tokens[i] for i in kept])


def compress_perplexity(text: str, ratio: float) -> str:
    """Keep the top-`ratio` fraction of tokens ranked by GPT-2 surprisal."""
    result = score_tokens_perplexity(text)
    tokens, scores = result["tokens"], result["surprisal"]
    k       = max(1, int(len(tokens) * ratio))
    top_idx = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    )
    return "".join([tokens[i] for i in top_idx])


# ── Learned compression (requires a loaded TokenScorer) ───────────────────────

def compress_learned(text: str, scorer, ratio: float = 0.5,
                     device: str = "cpu") -> dict:
    """
    Compress `text` using the trained TokenScorer.
    Returns a dict with:
        compressed     : str
        original_len   : int
        compressed_len : int
        savings_pct    : float
        scores         : list[float]
        tokens         : list[str]
        kept_mask      : list[bool]
    """
    tok = get_distilbert_tokenizer()
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=512).to(device)
    with torch.no_grad():
        scores = scorer(enc.input_ids, enc.attention_mask).squeeze()
    tokens  = tok.convert_ids_to_tokens(enc.input_ids.squeeze())
    k       = max(1, int(len(tokens) * ratio))
    top_idx = sorted(torch.topk(scores, k).indices.tolist())
    compressed = tok.convert_tokens_to_string([tokens[i] for i in top_idx])
    return {
        "compressed":     compressed,
        "original_len":   len(tokens),
        "compressed_len": k,
        "savings_pct":    round((1 - k / len(tokens)) * 100, 1),
        "scores":         scores.tolist(),
        "tokens":         tokens,
        "kept_mask":      [i in top_idx for i in range(len(tokens))],
    }