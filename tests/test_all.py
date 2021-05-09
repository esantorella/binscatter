from typing import Callable, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy import sparse as sps

import binscatter  # noqa: F401
from binscatter.binscatter import _get_bins, get_binscatter_objects


@pytest.fixture
def data() -> pd.DataFrame:
    n_obs = 100
    np.random.seed(0)
    data = pd.DataFrame({"experience": np.random.poisson(4, n_obs) + 1})
    data["Tenure"] = data["experience"] + np.random.normal(0, 1, n_obs)
    data["Wage"] = data["experience"] + data["Tenure"] + np.random.normal(0, 1, n_obs)
    return data


def test_get_bins():
    n_elements = 5
    n_bins = 2
    bins = _get_bins(n_elements, n_bins)
    assert len(bins) == n_bins
    assert all(isinstance(elt, slice) for elt in bins)
    assert all(elt.start < elt.stop for elt in bins)
    assert all(elt.step is None for elt in bins)
    test_arr = np.arange(n_elements * 2)
    sliced = [test_arr[bin] for bin in bins]
    # every element must be in sliced exactly once
    assert sum(len(elt) for elt in sliced) == n_elements
    elts_found = set()
    for data in sliced:
        assert len(set(data) & elts_found) == 0
        elts_found.update(set(data))
    assert elts_found == set(test_arr[:n_elements])


def test_get_binscatter_objects_no_controls():
    np.random.seed(0)
    n_bins = 2
    n_elts = 6
    x = np.random.random(n_elts)
    y = np.random.random(n_elts)

    x_means, y_means, intercept, coef = get_binscatter_objects(
        y,
        x,
        controls=None,
        n_bins=n_bins,
        recenter_x=False,
        recenter_y=False,
        bins=None,
    )
    for list_ in [x_means, y_means]:
        assert isinstance(list_, list)
        assert isinstance(list_[0], float)
        assert len(list_) == n_bins

    assert (np.diff(x_means) > 0).all()
    assert isinstance(intercept, float)
    predictions = np.dot(x, coef) + intercept
    residuals = y - predictions
    np.testing.assert_almost_equal(residuals.sum(), 0)
    np.testing.assert_almost_equal(residuals.dot(x), 0)

    bin_predictions = np.dot(x_means, coef) + intercept
    bin_resids = np.array(y_means) - bin_predictions
    np.testing.assert_almost_equal(bin_resids.sum(), 0)


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
    plt.close("all")


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
    plt.close("all")


@pytest.mark.parametrize("fit_reg", [True, None, False])
def test_readme_example_runs(fit_reg) -> None:

    # Create fake data
    n_obs = 1000
    data = pd.DataFrame({"experience": np.random.poisson(4, n_obs) + 1})
    data["Tenure"] = data["experience"] + np.random.normal(0, 1, n_obs)
    data["Wage"] = data["experience"] + data["Tenure"] + np.random.normal(0, 1, n_obs)
    fig, axes = plt.subplots(2, sharex=True, sharey=True)

    # Binned scatter plot of Wage vs Tenure
    axes[0].binscatter(data["Tenure"], data["Wage"], fit_reg=fit_reg)

    # Binned scatter plot that partials out the effect of experience
    axes[1].binscatter(
        data["Tenure"],
        data["Wage"],
        controls=data["experience"],
        recenter_x=True,
        recenter_y=True,
        fit_reg=fit_reg,
    )
    axes[1].set_xlabel("Tenure (residualized, recentered)")
    axes[1].set_ylabel("Wage (residualized, recentered)")

    plt.tight_layout()
    if fit_reg:
        plt.savefig("readme_example")
    plt.close("all")
