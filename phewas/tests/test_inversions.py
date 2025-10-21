from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phewas import iox


def test_load_inversions_deduplicates_person_ids(tmp_path):
    target = 'chr_test-1-INV-1'
    inversion_records = pd.DataFrame(
        {
            'SampleID': ['p1', 'p1', 'p2'],
            target: [0.1, 0.9, 0.3],
        }
    )
    inversion_path = tmp_path / 'inversions.tsv'
    inversion_records.to_csv(inversion_path, sep='\t', index=False)

    with pytest.warns(UserWarning, match='Duplicate person_id values encountered'):
        inversion_df = iox.load_inversions(target, str(inversion_path))

    assert list(inversion_df.index) == ['p1', 'p2']
    assert inversion_df.index.is_unique
    assert inversion_df.loc['p1', target] == pytest.approx(0.1)

    covariates = pd.DataFrame(
        {'AGE': [37.0, 52.0]},
        index=pd.Index(['p1', 'p2'], name='person_id'),
    )
    joined = covariates.join(inversion_df, how='inner')
    assert joined.index.is_unique
    assert joined.loc['p1', target] == pytest.approx(0.1)
