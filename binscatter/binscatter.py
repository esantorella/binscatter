"""
Inspired by Stata's binscatter, described fully by Michael Stepner at
https://michaelstepner.com/binscatter/.
"""
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse as sps
from sklearn import linear_model


def get_binscatter_objects(
    y: np.ndarray,
    x: np.ndarray,
    controls: Optional[np.ndarray],
    n_bins: int,
    recenter_x: bool,
    recenter_y: bool,
    bins: Optional[Iterable],
) -> Tuple[List[float], List[float], float, float]:
    """
    Returns mean x and mean y within each bin, and coefficients if residualizing.
    Parameters are essentially the same as in binscatter.
    """
    # Check if data is sorted

    if controls is None:
        if np.any(np.diff(x) < 0):
            argsort = np.argsort(x)
            x = x[argsort]
            y = y[argsort]
        x_data = x
        y_data = y
    else:
        # Residualize
        if controls.ndim == 1:
            controls = controls[:, None]

        demeaning_y_reg = linear_model.LinearRegression().fit(controls, y)
        y_data = y - demeaning_y_reg.predict(controls)

        demeaning_x_reg = linear_model.LinearRegression().fit(controls, x)
        x_data = x - demeaning_x_reg.predict(controls)
        argsort = np.argsort(x_data)
        x_data = x_data[argsort]
        y_data = y_data[argsort]

        if recenter_y:
            y_data += np.mean(y)
        if recenter_x:
            x_data += np.mean(x)

    if x_data.ndim == 1:
        x_data = x_data[:, None]
    reg = linear_model.LinearRegression().fit(x_data, y_data)
    if bins is None:
        bin_edges = np.linspace(0, len(y), n_bins + 1).astype(int)
        assert len(bin_edges) == n_bins + 1
        bins = [
            slice(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)
        ]
        assert len(bins) == n_bins

    x_means = [np.mean(x_data[bin_]) for bin_ in bins]
    y_means = [np.mean(y_data[bin_]) for bin_ in bins]

    return x_means, y_means, reg.intercept_, reg.coef_[0]


def binscatter(
    self,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    controls: Optional[
        Union[pd.DataFrame, pd.SparseDataFrame, np.ndarray, sps.spmatrix]
    ] = None,
    n_bins=20,
    line_kwargs: Optional[Dict] = None,
    scatter_kwargs: Optional[Dict] = None,
    recenter_x: bool = False,
    recenter_y: bool = True,
    # TODO: make 'bins' consistent with functions in other libraries, as in pd.cut
    bins: Optional[Iterable[slice]] = None,
) -> Tuple[List[float], List[float], float, float]:
    """
    :param self: matplotlib.axes.Axes object.
        i.e., fig, axes = plt.subplots(3)
              axes[0].binscatter(x, y)

    :param y: Numpy ArrayLike, such as numpy.ndarray or pandas.Series; must be 1d
    :param x: Numpy ArrayLike, such as numpy.ndarray or pandas.Series
    :param controls: Optional; whatever can be passed to
        sklearn.linear_model.LinearRegression, such as Numpy array or sparse matrix
    :param n_bins: int, default 20
    :param line_kwargs: keyword arguments passed to the line in the
    :param scatter_kwargs: dict
    :param recenter_y: If true, recenter y-tilde so its mean is the mean of y
    :param recenter_x: If true, recenter y-tilde so its mean is the mean of y
    :param bins: Indices of each bin. By default, if you leave 'bins' as None,
        binscatter constructs equal sized bins;
        if you don't like that, use this parameter to construct your own.
    :return:
    """
    if line_kwargs is None:
        line_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if controls is not None:
        if isinstance(controls, pd.SparseDataFrame) or isinstance(
            controls, pd.SparseSeries
        ):
            controls = controls.to_coo()
        elif isinstance(controls, pd.DataFrame) or isinstance(controls, pd.Series):
            controls = controls.values
        assert isinstance(controls, np.ndarray) or sps.issparse(controls)

    x_means, y_means, intercept, coef = get_binscatter_objects(
        np.asarray(y), np.asarray(x), controls, n_bins, recenter_x, recenter_y, bins
    )

    self.scatter(x_means, y_means, **scatter_kwargs)
    x_range = np.array(self.get_xlim())
    self.plot(
        x_range,
        intercept + x_range * coef,
        label="beta=" + str(round(coef, 3)),
        **line_kwargs
    )
    # If series were passed, might be able to label
    if hasattr(x, "name"):
        self.set_xlabel(x.name)
    if hasattr(y, "name"):
        self.set_ylabel(y.name)
    return x_means, y_means, intercept, coef


matplotlib.axes.Axes.binscatter = binscatter
