"""States reading and functions."""

import astropy.units as u
import numpy as np
import astropy.constants as const
import pandas as pd
import numpy.typing as npt


def partition(
    energy: u.Quantity, g_total: u.Quantity, temperature: u.Quantity
) -> u.Quantity:
    """Partition function."""
    c2 = const.h * const.c / const.k_B

    if not isinstance(energy, u.Quantity):
        raise ValueError("Energy must be a Quantity object.")

    if not isinstance(temperature, u.Quantity):
        raise ValueError("Temperature must be a Quantity object.")

    return np.sum(g_total * np.exp(-c2 * energy / temperature))


def read_exomol_states_dataframe(filename: str) -> pd.DataFrame:
    """Read ExoMol states file.

    Args:
        filename: ExoMol states file.

    """
    from pathlib import Path

    p = Path(filename)

    if p.is_file() is False:
        raise FileNotFoundError(f"{filename} not found")

    states = pd.read_csv(
        p,
        sep="\s+",
        usecols=[0, 1, 2, 3],
        index_col=0,
        names=["ID", "Energy", "g_total", "J"],
    )

    return states


def convert_exomol_states_dataframe(
    df: pd.DataFrame,
) -> tuple[u.Quantity, u.Quantity, npt.NDArray[np.int64]]:
    energy = df["Energy"].values << (1 / u.cm)
    g_total = df["g_total"].values << u.dimensionless_unscaled
    J = df["J"].values.astype(np.int64)
    return energy, g_total, J


class ExomolStates:

    def __init__(self, filename: str):
        self.filename = filename
        self.states_df = read_exomol_states_dataframe(filename)
        (self.energy, self.g_total, J) = convert_exomol_states_dataframe(self.states_df)

    def Q(self, temperature: u.Quantity) -> u.Quantity:
        return partition(self.energy, self.g_total, temperature)

    @property
    def df(self):
        return self.states_df
