
import torch
from transformers import DistilBertTokenizerFast

_tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def compress(text: str, scorer, ratio: float = 0.5, device: str = "cpu") -> dict:
    enc = _tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        scores = scorer(enc.input_ids, enc.attention_mask).squeeze()
    tokens = _tok.convert_ids_to_tokens(enc.input_ids.squeeze())
    k = max(1, int(len(tokens) * ratio))
    top_idx = sorted(torch.topk(scores, k).indices.tolist())
    compressed = _tok.convert_tokens_to_string([tokens[i] for i in top_idx])
    return {
        "compressed": compressed,
        "original_len": len(tokens),
        "compressed_len": k,
        "savings_pct": round((1 - k / len(tokens)) * 100, 1),
        "scores": scores.tolist(),
        "tokens": tokens,
        "kept_mask": [i in top_idx for i in range(len(tokens))],
    }
