import pytest


@pytest.fixture
def exomol_state(tmp_path):
    d = tmp_path / "exomol_file"
    d.mkdir()
    p = d / "test.states"

    p.write_text(
        """           1     0.000000      1       0         Inf +  1          1 p A1  0  0  0  0  0
           2  1594.873096      1       0  4.1203e-02 +  1          2 p A1  0  1  0  0  0
           3  3151.677108      1       0  2.0601e-02 +  1          3 p A1  0  2  0  0  0
           4  3657.155752      1       0  1.4963e-01 +  1          4 p A1  1  0  0  0  0
           5  4666.724999      1       0  1.3749e-02 +  1          5 p A1  0  3  0  0  0
           6  5235.220005      1       0  3.3154e-02 +  1          6 p A1  1  1  0  0  0
"""
    )

    return p


def test_exomol_read_dataframe(exomol_state):
    from picocross.states import read_exomol_states_dataframe

    df = read_exomol_states_dataframe(exomol_state)

    assert len(df) == 6


def test_nopath():
    """Test if fails when no path is given."""
    import uuid

    bad_path = f"/my/cool/path/here/{uuid.uuid4()}"
    from picocross.states import read_exomol_states_dataframe

    with pytest.raises(FileNotFoundError):
        read_exomol_states_dataframe(bad_path)
