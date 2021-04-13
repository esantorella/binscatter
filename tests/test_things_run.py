from typing import Callable, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy import sparse as sps

import binscatter  # noqa: F401


@pytest.fixture
def data() -> pd.DataFrame:
    n_obs = 100
    np.random.seed(0)
    data = pd.DataFrame({"experience": np.random.poisson(4, n_obs) + 1})
    data["Tenure"] = data["experience"] + np.random.normal(0, 1, n_obs)
    data["Wage"] = data["experience"] + data["Tenure"] + np.random.normal(0, 1, n_obs)
    return data


@pytest.mark.parametrize("recenter_y", [False, True])
@pytest.mark.parametrize("x_convert_fn", [np.asarray, list, lambda x: x])
@pytest.mark.parametrize("y_convert_fn", [np.asarray, list, lambda x: x])
@pytest.mark.parametrize(
    "control_convert_fn",
    [
        np.asarray,
        lambda x: list(x.values[:, 0]),
        lambda x: x,
        sps.coo_matrix,
        sps.csc_matrix,
        sps.csr_matrix,
    ],
)
@pytest.mark.parametrize("bins", [None, [slice(0, 50), slice(50, 100)]])
def test_fig_runs(
    data: pd.DataFrame,
    recenter_y: bool,
    x_convert_fn: Callable[[pd.Series], npt.ArrayLike],
    y_convert_fn: Callable[[pd.Series], npt.ArrayLike],
    control_convert_fn: Callable[[pd.DataFrame], npt.ArrayLike],
    bins: Optional[List[slice]],
):

    fig, ax = plt.subplots()
    ax.binscatter(
        x_convert_fn(data["Tenure"]),
        y_convert_fn(data["Wage"]),
        controls=control_convert_fn(data[["experience"]]),
        recenter_y=recenter_y,
        bins=bins,
    )


@pytest.mark.parametrize("recenter_y", [False, True])
@pytest.mark.parametrize("x_convert_fn", [np.asarray, list, lambda x: x])
@pytest.mark.parametrize("y_convert_fn", [np.asarray, list, lambda x: x])
@pytest.mark.parametrize("bins", [None, [slice(0, 50), slice(50, 100)]])
def test_fig_runs_no_controls(
    data: pd.DataFrame,
    recenter_y: bool,
    x_convert_fn: Callable[[pd.Series], npt.ArrayLike],
    y_convert_fn: Callable[[pd.Series], npt.ArrayLike],
    bins: Optional[List[slice]],
):

    fig, ax = plt.subplots()
    ax.binscatter(
        x_convert_fn(data["Tenure"]),
        y_convert_fn(data["Wage"]),
        recenter_y=recenter_y,
        bins=bins,
    )


def test_readme_example_runs() -> None:

    # Create fake data
    n_obs = 1000
    data = pd.DataFrame({"experience": np.random.poisson(4, n_obs) + 1})
    data["Tenure"] = data["experience"] + np.random.normal(0, 1, n_obs)
    data["Wage"] = data["experience"] + data["Tenure"] + np.random.normal(0, 1, n_obs)
    fig, axes = plt.subplots(2, sharex=True, sharey=True)

    # Binned scatter plot of Wage vs Tenure
    axes[0].binscatter(data["Tenure"], data["Wage"])

    # Binned scatter plot that partials out the effect of experience
    axes[1].binscatter(
        data["Tenure"],
        data["Wage"],
        controls=data["experience"],
        recenter_x=True,
        recenter_y=True,
    )
    axes[1].set_xlabel("Tenure (residualized, recentered)")
    axes[1].set_ylabel("Wage (residualized, recentered)")

    plt.tight_layout()
    plt.savefig("readme_example")
    plt.close("all")
