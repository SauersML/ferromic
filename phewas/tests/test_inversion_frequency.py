import os
import sys

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from phewas import inversion_frequency as inv_freq


def test_summarize_population_frequencies_computes_ci():
    dosages = pd.DataFrame(
        {
            "invA": [0.0, 1.0, 2.0, 2.0],
            "invB": [1.0, 1.0, 1.0, 1.0],
        },
        index=pd.Index(["p1", "p2", "p3", "p4"], name="person_id"),
    )
    ancestry = pd.Series(
        ["EUR", "EUR", "AFR", "AFR"],
        index=dosages.index,
        name="ANCESTRY",
    )

    summary = inv_freq.summarize_population_frequencies(dosages, ancestry)

    eur_invA = summary[(summary["Inversion"] == "invA") & (summary["Population"] == "EUR")].iloc[0]
    assert eur_invA["N"] == 2
    assert eur_invA["Allele_Freq"] == pytest.approx(0.25)
    assert eur_invA["CI95_Lower"] == pytest.approx(0.0)
    assert eur_invA["CI95_Upper"] == pytest.approx(0.74, rel=1e-3)

    afr_invA = summary[(summary["Inversion"] == "invA") & (summary["Population"] == "AFR")].iloc[0]
    assert afr_invA["N"] == 2
    assert afr_invA["Allele_Freq"] == pytest.approx(1.0)
    assert afr_invA["CI95_Lower"] == pytest.approx(1.0)
    assert afr_invA["CI95_Upper"] == pytest.approx(1.0)

    all_invB = summary[(summary["Inversion"] == "invB") & (summary["Population"] == "ALL")].iloc[0]
    assert all_invB["N"] == 4
    assert all_invB["Allele_Freq"] == pytest.approx(0.5)
    assert all_invB["CI95_Lower"] == pytest.approx(0.5)
    assert all_invB["CI95_Upper"] == pytest.approx(0.5)


def test_load_all_inversion_dosages_coerces_and_deduplicates(tmp_path):
    input_path = tmp_path / "dosages.tsv"
    pd.DataFrame(
        {
            "SampleID": ["s1", "s1", "s2"],
            "invA": [0.1, 0.2, ""],
            "invB": ["1.0", "1.0", "not_a_number"],
        }
    ).to_csv(input_path, sep="\t", index=False)

    loaded = inv_freq.load_all_inversion_dosages(str(input_path))

    assert list(loaded.index) == ["s1", "s2"]
    assert loaded.index.name == "person_id"
    assert loaded.loc["s1", "invA"] == pytest.approx(0.1)
    assert np.isnan(loaded.loc["s2", "invA"])
    assert loaded.loc["s1", "invB"] == pytest.approx(1.0)
    assert np.isnan(loaded.loc["s2", "invB"])
