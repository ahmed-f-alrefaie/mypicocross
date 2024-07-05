import numba
import numpy as np
from astropy import units as u


def doppler_profile(
    grid: u.Quantity, vfi: u.Quantity, intens: u.Quantity, broading: u.Quantity
) -> u.Quantity:
    """Doppler profile using numpy."""
    ln2 = np.log(2)
    ln2sqpi = np.sqrt(ln2 / np.pi)
    doppl = np.square(grid[None, :] - vfi[:, None]) * ln2 / broading**2
    return np.sum(intens[:, None] * np.exp(-doppl) / broading * ln2sqpi, axis=0)


@numba.njit(parallel=True)
def _doppler_profile_numba(grid, vfi, intens, broading):
    ln2 = np.log(2)
    ln2sqpi = np.sqrt(ln2 / np.pi) / broading
    intens = np.zeros(grid.shape)
    num_states = vfi.shape[0]
    num_grid = grid.shape[0]
    for i in numba.prange(num_states):
        for j in range(num_grid):
            intens[j] += (
                intens[i]
                * np.exp(-np.square(grid[j] - vfi[i]) * ln2 / broading**2)
                * ln2sqpi
            )

    return intens


def doppler_profile_numba(
    grid: u.Quantity, vfi: u.Quantity, intens: u.Quantity, broading: u.Quantity
) -> u.Quantity:
    """Doppler profile using numba."""
    grid = grid.to(1 / u.cm).value
    vfi = vfi.to(1 / u.cm).value
    intens = intens.to(u.cm).value
    broading = broading.to(1 / u.cm).value

    result = _doppler_profile_numba(grid, vfi, intens, broading)

    return result << u.cm**2
