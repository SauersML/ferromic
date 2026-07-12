"""Evo 2 7B zero-shot delta-likelihood variant-effect scoring (Arm 1 third method, and the
shared ClinVar positive-control scorer).

For each variant ``(chrom, g_pos_1based, g_ref, g_alt)`` a window of reference sequence is
extracted, ref and alt versions are built, and each is scored with Evo 2 teacher-forcing
log-likelihood::

    delta_ll = sum_t [ log P(alt_t | alt_<t) - log P(ref_t | ref_<t) ]

More negative = more disruptive. Both the full-window sum and a local (+/-16 bp) sum around
the variant are reported.

Requires a GPU and the Evo 2 7B weights (``arcinstitute/evo2_7b``); not exercised by the
reproduction tests, which consume the frozen per-method score table.
"""
from __future__ import annotations

import math

WINDOW = 4096


def token_logprobs(model, seq: str):
    """Teacher-forcing per-token log-probabilities for positions 1..L-1 of ``seq``."""
    import torch

    ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to("cuda")
    with torch.no_grad():
        out, _ = model(ids)
        logits = out[0] if isinstance(out, (list, tuple)) else out  # [1, L, V]
        logp = torch.log_softmax(logits[0].float(), dim=-1)          # [L, V]
        tgt = ids[0].long()
        return logp[:-1].gather(-1, tgt[1:, None]).squeeze(-1)       # [L-1]


def delta_ll(model, fa, chrom: str, pos: int, ref: str, alt: str,
             window: int = WINDOW) -> dict:
    """Evo 2 delta-log-likelihood for a single SNV. Returns a dict with ``evo2_delta_ll``,
    ``evo2_delta_ll_local``, ``evo2_refcheck``. ``fa`` is a pyfaidx.Fasta-like object."""
    half = window // 2
    s, e = pos - half, pos + half
    win = fa[chrom][s - 1:e].seq.upper()
    ci = pos - s  # 0-based index of the variant in the window
    if ci < 0 or ci >= len(win) or win[ci] != ref:
        return {"evo2_delta_ll": math.nan, "evo2_delta_ll_local": math.nan,
                "evo2_refcheck": f"MISMATCH:{win[ci] if 0 <= ci < len(win) else '?'}"}
    alt_win = win[:ci] + alt + win[ci + 1:]
    lp_ref = token_logprobs(model, win)
    lp_alt = token_logprobs(model, alt_win)
    n = min(len(lp_ref), len(lp_alt))
    lo, hi = max(0, ci - 16), min(n, ci + 16)
    return {
        "evo2_delta_ll": float((lp_alt[:n] - lp_ref[:n]).sum()),
        "evo2_delta_ll_local": float((lp_alt[lo:hi] - lp_ref[lo:hi]).sum()),
        "evo2_refcheck": "OK",
    }


def load_model(name: str = "evo2_7b"):
    from evo2 import Evo2
    return Evo2(name)


def score_table(variant_rows, fa, model=None, window: int = WINDOW) -> list[dict]:
    """Score variants (rows with chrom/g_pos_1based/g_ref/g_alt). Loads Evo 2 if ``model`` is
    None. Extra input columns are preserved on the output rows."""
    if model is None:
        model = load_model()
    out = []
    for r in variant_rows:
        res = delta_ll(model, fa, str(r["chrom"]), int(r["g_pos_1based"]),
                       str(r["g_ref"]), str(r["g_alt"]), window)
        out.append({**dict(r), **res})
    return out
