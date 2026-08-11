import pandas as pd
import pytest

from scripts.freeze_investigator_vegetation_reference_contract import boolean_series


def test_boolean_series_accepts_csv_boolean_strings():
    values = pd.Series(["True", "false", " TRUE "])
    assert boolean_series(values).tolist() == [True, False, True]


def test_boolean_series_rejects_unknown_values():
    with pytest.raises(ValueError, match="non-boolean"):
        boolean_series(pd.Series(["true", "unknown"]))
