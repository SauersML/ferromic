"""Path resolution + provenance recording for the ``recurrence/`` analyses.

The recurrence pipeline is self-contained: its inputs (the committed real-inversion
tables) and its recorded results live under this package. This module centralises
those locations and writes a provenance sidecar next to every generated output, so
each committed result is traceable to the exact inputs and library versions that
produced it.

The one genuinely external, optional input is the reference simulation pipeline
(``simulations/refsim/`` + msprime + Biopython + IQ-TREE), used only by the
``simulate`` stage to regenerate the training set. The committed, gzip'd training
set (``data/sim_features.csv.gz``) lets every downstream stage and every test run
without that stack.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))     # recurrence/
REPO_ROOT = os.path.dirname(HERE)                      # ferromic/
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

# committed inputs (small, real-inversion reference tables)
INV_PROPERTIES = os.path.join(DATA, "inv_properties.tsv")
OUTPUT_CSV = os.path.join(DATA, "output.csv")
# committed training set (regenerable via the ``simulate`` stage)
SIM_FEATURES = os.path.join(DATA, "sim_features.csv.gz")

# recorded reference results
MODEL_FULL = os.path.join(RESULTS, "model.json")
MODEL_TRANSFERABLE = os.path.join(RESULTS, "transferable_model.json")
SIM_METRICS = os.path.join(RESULTS, "sim_metrics.json")
TF_SIM_METRICS = os.path.join(RESULTS, "tf_sim_metrics.json")
SIM_TEST_PRED = os.path.join(RESULTS, "sim_test_pred.csv.gz")
TF_SIM_TEST_PRED = os.path.join(RESULTS, "tf_sim_test_pred.csv.gz")
REAL_SCORES = os.path.join(RESULTS, "real_scores.csv")
CONCORDANCE = os.path.join(RESULTS, "concordance.json")


def _versions() -> dict:
    v = {}
    for mod in ("numpy", "scipy", "sklearn", "pandas", "msprime"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def _relpath(p: str) -> str:
    """Path relative to the repo root when the input lives inside the repo, else the
    absolute path. Keeps committed provenance portable (no machine-specific prefix)."""
    ap = os.path.abspath(p)
    try:
        rel = os.path.relpath(ap, REPO_ROOT)
        return rel if not rel.startswith("..") else ap
    except ValueError:
        return ap


def write_provenance(out_path: str, inputs: dict, extra: dict | None = None) -> None:
    """Record resolved input paths, library versions, and a UTC timestamp next to a
    generated output, so every committed result is traceable to what produced it."""
    rec = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "inputs": {k: _relpath(p) for k, p in inputs.items()},
        "versions": _versions(),
    }
    if extra:
        rec.update(extra)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=2)
