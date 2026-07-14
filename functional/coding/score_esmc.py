"""ESM C masked-marginal log-likelihood ratio (LLR) for each missense variant.

``LLR = log P(alt_aa | context) - log P(ref_aa | context)`` at the variant residue.
More negative = the alternate residue is more deleterious (destabilising) than reference.
Uses ``esmc_300m``; proteins longer than the model context are cropped to a window centred
on the variant.

Requires a GPU and the ESM C weights (``pip install esm``); it is not exercised by the
reproduction tests, which consume the frozen per-method score table.
"""
from __future__ import annotations

import math

WINDOW = 1022  # keep within ESM C context; centre on the variant


def _crop(seq: str, pos0: int, window: int = WINDOW) -> tuple[str, int]:
    """Crop ``seq`` to ``window`` residues centred on ``pos0``; return (subseq, local_index)."""
    n = len(seq)
    if n <= window:
        return seq, pos0
    half = window // 2
    s = max(0, pos0 - half)
    e = min(n, s + window)
    s = max(0, e - window)
    return seq[s:e], pos0 - s


def esmc_llr(client, tok, seq: str, pos0: int, ref_aa: str, alt_aa: str) -> tuple[float, str]:
    """Masked-marginal LLR at residue ``pos0`` (0-based). Returns ``(llr, observed_ref_aa)``.

    ``client`` is a loaded ``esm.models.esmc.ESMC``; ``tok`` is the ESM C tokenizer
    (``get_esmc_model_tokenizers()``). Kept dependency-free at import so the module can be
    imported (and its cropping logic tested) without the ESM package installed.
    """
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig

    sub, local = _crop(seq, pos0)
    tensor = client.encode(ESMProtein(sequence=sub))
    logits_out = client.logits(tensor, LogitsConfig(sequence=True))
    lg = logits_out.logits.sequence[0]  # [L+special, vocab]; ESM C prepends BOS -> residue i at i+1
    logprobs = torch.log_softmax(lg[local + 1].float(), dim=-1)
    vocab = tok.get_vocab()
    try:
        llr = float(logprobs[vocab[alt_aa]] - logprobs[vocab[ref_aa]])
    except (KeyError, IndexError):
        llr = math.nan
    return llr, (sub[local] if 0 <= local < len(sub) else "?")


def load_model(model: str = "esmc_300m"):
    """Load an ESM C model onto CUDA if available, else CPU, plus its tokenizer."""
    import torch
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    client = ESMC.from_pretrained(model).to(dev)
    client.eval()
    return client, get_esmc_model_tokenizers()


def score_table(scored_rows, proteins: dict, model: str = "esmc_300m") -> list[dict]:
    """Score every missense row (``aa_num`` 1-based, ``aa_ref``/``aa_alt``) that has a protein
    sequence in ``proteins`` (transcript_id -> sequence). Returns rows with ``esmc_llr`` +
    ``esmc_refcheck``."""
    import torch

    client, tok = load_model(model)
    out = []
    with torch.no_grad():
        for r in scored_rows:
            if r.get("consequence") != "missense":
                continue
            seq = proteins.get(r["transcript_id"])
            base = dict(r)
            if not isinstance(seq, str):
                base.update(esmc_llr=math.nan, esmc_refcheck="NO_PROTEIN")
            else:
                pos0 = int(r["aa_num"]) - 1
                if pos0 >= len(seq):
                    base.update(esmc_llr=math.nan, esmc_refcheck="POS_OOR")
                else:
                    val, seen = esmc_llr(client, tok, seq, pos0, r["aa_ref"], r["aa_alt"])
                    base.update(esmc_llr=val,
                                esmc_refcheck="OK" if seen == r["aa_ref"] else f"MISMATCH:{seen}")
            out.append(base)
    return out
