"""Monkey-patch Matplotlib to add an 'ax.binscatter' method."""
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import numpy.typing as npt
from sklearn import linear_model


def _get_bins(n_elements: int, n_bins: int) -> List[slice]:
    bin_edges = np.linspace(0, n_elements, n_bins + 1).astype(int)
    bins = [slice(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    return bins


def get_binscatter_objects(
    y: np.ndarray,
    x: np.ndarray,
    controls,
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
        if np.ndim(controls) == 1:
            controls = np.expand_dims(controls, 1)

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
        bins = _get_bins(len(y), n_bins)

    x_means = [np.mean(x_data[bin_]) for bin_ in bins]
    y_means = [np.mean(y_data[bin_]) for bin_ in bins]

    return x_means, y_means, reg.intercept_, reg.coef_[0]


def binscatter(
    self,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    controls=None,
    n_bins=20,
    line_kwargs: Optional[Dict] = None,
    scatter_kwargs: Optional[Dict] = None,
    recenter_x: bool = False,
    recenter_y: bool = True,
    # TODO: make 'bins' consistent with functions in other libraries, as in pd.cut
    bins: Optional[Iterable[slice]] = None,
    fit_reg: Optional[bool] = True,
) -> Tuple[List[float], List[float], float, float]:
    """
    :param self: matplotlib.axes.Axes object.
        i.e., fig, axes = plt.subplots(3)
              axes[0].binscatter(x, y)

    :param y: Numpy ArrayLike, such as numpy.ndarray or pandas.Series; must be 1d
    :param x: Numpy ArrayLike, such as numpy.ndarray or pandas.Series
    :param controls: Optional, {array-like, sparse matrix}; whatever can be passed to
        sklearn.linear_model.LinearRegression
    :param n_bins: int, default 20
    :param line_kwargs: keyword arguments passed to the line in the
    :param scatter_kwargs: dict
    :param recenter_y: If true, recenter y-tilde so its mean is the mean of y
    :param recenter_x: If true, recenter y-tilde so its mean is the mean of y
    :param bins: Indices of each bin. By default, if you leave 'bins' as None,
        binscatter constructs equal sized bins;
        if you don't like that, use this parameter to construct your own.
    :param fit_reg: Whether to plot a regression line.
    """
    if line_kwargs is None:
        line_kwargs = {}
    elif not fit_reg:
        warnings.warn("Both fit_reg=False and non-None line_kwargs were passed.")
    if scatter_kwargs is None:
        scatter_kwargs = {}

    x_means, y_means, intercept, coef = get_binscatter_objects(
        np.asarray(y), np.asarray(x), controls, n_bins, recenter_x, recenter_y, bins
    )

    self.scatter(x_means, y_means, **scatter_kwargs)
    x_range = np.array(self.get_xlim())
    if fit_reg:
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
