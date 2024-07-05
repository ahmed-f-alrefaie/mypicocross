import pandas as pd
from astropy import units as u
from astropy import constants as const
import numpy as np
from .states import ExomolStates


def read_exomol_transitions_dataframe(filename: str, **kwargs) -> pd.DataFrame:
    """Read ExoMol transitions file.

    Args:
        filename: ExoMol transitions file.

    """
    from pathlib import Path

    p = Path(filename)
    if p.is_file() is False:
        raise FileNotFoundError(f"{filename} not found")

    transitions = pd.read_csv(
        p, sep="\s+", usecols=[0, 1, 2], names=["UpperId", "LowerId", "Afi"], **kwargs
    )

    return transitions


def read_transitions_iter(directory: str, chunksize=1000) -> pd.DataFrame:
    """Read all transitions in a directory."""
    import pathlib

    p = pathlib.Path(directory)

    if not p.is_dir():
        raise FileNotFoundError(f"{directory} not found or is not directory")

    transitions_files = p.glob("*.trans")

    for f in transitions_files:
        for df in read_exomol_transitions_dataframe(f, chunksize=chunksize):
            yield df


def merge_transitions_states(
    states: pd.DataFrame, transition: pd.DataFrame
) -> pd.DataFrame:
    """Merge transitions and states dataframes."""

    new_df = transition.merge(
        states,
        left_on="UpperId",
        right_index=True,
    )

    new_df = new_df.merge(
        states,
        left_on="LowerId",
        right_index=True,
        suffixes=(
            "_upper",
            "_lower",
        ),
    )

    new_df["vfi"] = new_df["Energy_upper"] - new_df["Energy_lower"]

    return new_df


def transition_intensities(
    transitions: pd.DataFrame, temperature: u.Quantity, partition_function: u.Quantity
) -> tuple[u.Quantity, u.Quantity]:
    """Calculate transition intensities.

    Args:
        transitions: DataFrame with transitions.
        temperature: Temperature.

    Returns:
        A tuple with the transition wavenumbers and absolute intensities.

    """
    Afi = transitions["Afi"].values << (1 / u.s)
    vfi = transitions["vfi"].values << (1 / u.cm)
    Ei = transitions["Energy_lower"].values << (1 / u.cm)
    gtotf = transitions["g_total_upper"].values

    c2 = const.h * const.c / const.k_B

    intensity_factor = gtotf * Afi / (8 * np.pi * const.c * vfi**2)

    intensity_energy = np.exp(-c2 * Ei / temperature)

    intensity_transition = np.exp(-c2 * vfi / temperature)

    return (
        vfi,
        (
            intensity_factor
            * intensity_energy
            * (1 - intensity_transition)
            / partition_function
        ).to(u.cm),
    )


class ExomolTransitions:

    def __init__(self, directory: str, states: ExomolStates):
        """Read ExoMol transitions files."""
        self.state = states
        self.directory = directory

    def iterate(self, chunksize=1000):
        """Iterate over transitions."""
        for df in read_transitions_iter(self.directory, chunksize=chunksize):
            yield merge_transitions_states(self.state.df, df)

    def iterate_transitions(self, temperature, chunksize=1000):
        """Iterate over transitions."""
        Q = self.state.Q(temperature)
        for df in self.iterate(chunksize=chunksize):
            yield transition_intensities(df, temperature, Q)
