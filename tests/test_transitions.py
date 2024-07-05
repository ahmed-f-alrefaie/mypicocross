import pytest
import pandas as pd
import numpy as np
from astropy import units as u


@pytest.fixture
def exomol_transitions(tmp_path):
    d = tmp_path / "exomol_file"
    d.mkdir()
    p = d / "test.trans"

    p.write_text(
        """       2        1 6.4700e-22
       2        5 1.4010e-20
      5       3 3.5010e-26
      3       4 2.3110e-24
       7        6 1.3020e-19
"""
    )

    return p


def test_read_exomol_transitions(exomol_transitions):
    from picocross.transitions import read_exomol_transitions_dataframe

    df = read_exomol_transitions_dataframe(exomol_transitions)

    assert len(df) == 5

    assert list(df.columns) == ["UpperId", "LowerId", "Afi"]


def test_merge_states_transitions(exomol_transitions):
    from picocross.transitions import (
        read_exomol_transitions_dataframe,
        merge_transitions_states,
    )
    import pandas as pd

    transitions = read_exomol_transitions_dataframe(exomol_transitions)

    states = pd.DataFrame(
        {
            "Energy": [
                0.0,
                1594.873096,
                3151.677108,
                3657.155752,
                4666.724999,
                5235.220005,
                6000.000000,
            ],
            "g_total": [1, 1, 1, 1, 1, 1, 1],
            "J": [0, 1, 2, 0, 3, 1, 2],
        },
        index=[1, 2, 3, 4, 5, 6, 7],
    )

    new_df = merge_transitions_states(states, transitions)

    assert len(new_df) == 5

    transitions = new_df["vfi"].values

    energies = states["Energy"].values

    assert transitions[0] == energies[1] - energies[0]
    assert transitions[1] == energies[1] - energies[4]


def test_transition_intensities():
    from picocross.transitions import transition_intensities

    trans_df = pd.DataFrame(
        {
            "Afi": np.random.rand(5),
            "Energy_lower": np.random.rand(5),
            "vfi": np.random.rand(5),
            "g_total_upper": np.random.rand(5),
        }
    )

    temperature = 300 * u.K

    partition_function = 1 * u.dimensionless_unscaled

    transition_wn, absolute_intensity = transition_intensities(
        trans_df, temperature, partition_function
    )

    absolute_intensity.to(u.cm)
