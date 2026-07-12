"""Locate the committed data/result files that ship inside ``functional/structural``.

Small derived inputs (target table, analysis-locus list, per-window gene spans, config)
and the recorded result tables live in the package so the reproduction runs without the
large external inputs. Large inputs (2bit reference, 1000G panel, GENCODE GTF) still
resolve through :mod:`functional.paths`.
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "data", "structural")
RESULTS_DIR = os.path.join(_HERE, "..", "results", "structural")


def data_path(name: str) -> str:
    return os.path.abspath(os.path.join(DATA_DIR, name))


def results_path(name: str) -> str:
    return os.path.abspath(os.path.join(RESULTS_DIR, name))


def load_data(name: str):
    with open(data_path(name)) as fh:
        return json.load(fh)


def load_results(name: str):
    with open(results_path(name)) as fh:
        return json.load(fh)
