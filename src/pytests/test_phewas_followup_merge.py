import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from phewas import run


def build_stage1_df():
    return pd.DataFrame(
        {
            "Phenotype": ["phe1", "phe2"],
            "Inversion": ["inv1", "inv2"],
            "P_Source": ["stage1_a", "stage1_b"],
            "Other": [1, 2],
        }
    )


def test_merge_followup_results_renames_overlapping_columns():
    stage1_df = build_stage1_df()
    follow_df = pd.DataFrame(
        {
            "Phenotype": ["phe1", "phe2"],
            "Inversion": ["inv1", "inv2"],
            "P_Source": ["stage2_a", "stage2_b"],
            "Beta": [0.1, 0.2],
        }
    )

    merged = run._merge_followup_results(stage1_df, follow_df)

    # Stage-1 values remain intact
    assert list(merged["P_Source"]) == ["stage1_a", "stage1_b"]

    # Follow-up metrics are available under a Stage2-prefixed column
    assert "Stage2_P_Source" in merged.columns
    assert list(merged["Stage2_P_Source"]) == ["stage2_a", "stage2_b"]
    assert list(merged["Beta"]) == [0.1, 0.2]


def test_merge_followup_results_handles_existing_stage2_prefix():
    stage1_df = build_stage1_df()
    follow_df = pd.DataFrame(
        {
            "Phenotype": ["phe1", "phe2"],
            "Inversion": ["inv1", "inv2"],
            "P_Source": ["stage2_primary", "stage2_secondary"],
            "Stage2_P_Source": ["preexisting", "values"],
        }
    )

    merged = run._merge_followup_results(stage1_df, follow_df)

    # Original stage-1 column is untouched
    assert list(merged["P_Source"]) == ["stage1_a", "stage1_b"]

    # The follow-up "P_Source" column is renamed to a unique Stage2-prefixed name
    dynamic_cols = [c for c in merged.columns if c.startswith("Stage2_P_Source")]
    assert len(dynamic_cols) == 2
    assert "Stage2_P_Source" in dynamic_cols
    dynamic_cols.remove("Stage2_P_Source")
    # Remaining column stores the renamed overlap (e.g., Stage2_P_Source_2)
    renamed_col = dynamic_cols[0]
    assert list(merged[renamed_col]) == ["stage2_primary", "stage2_secondary"]
    # The pre-existing Stage2-prefixed column is preserved as-is
    assert list(merged["Stage2_P_Source"]) == ["preexisting", "values"]
