"""Shared test fixtures + path helpers for the functional analyses.

Pure-logic reproductions (coding combination, CDS mapping, per-locus integration) run from
the committed reference tables and always execute. Data-dependent reproductions (Geuvadis
eQTL, AlphaGenome splice) need large external inputs and skip unless those are configured via
``FUNCTIONAL_DATA_ROOT`` / ``FUNCTIONAL_*`` env vars (see ``functional/paths.py``).
"""
import csv
import gzip
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                      # functional/
DATA = os.path.join(PKG, "data")
RESULTS = os.path.join(PKG, "results")


def read_tsv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@pytest.fixture
def data_dir():
    return DATA


@pytest.fixture
def results_dir():
    return RESULTS
