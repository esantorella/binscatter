import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn import linear_model


def _get_bins(n_elements: int, n_bins: int) -> List[slice]:
    """
    Returns a list of slice objects representing bins of equal size.

    Parameters
    ----------
    n_elements : int
        The number of elements to divide into bins.
    n_bins : int
        The number of bins to create.

    Returns
    -------
    bins : list of slice objects
        The bins, represented as slice objects.
    """

    bin_edges = np.linspace(0, n_elements, n_bins + 1).astype(int)
    return [slice(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]


def get_binscatter_objects(y: np.ndarray, x: np.ndarray, controls, n_bins: int, recenter_x: bool, recenter_y: bool,
                           bins: Optional[Iterable]) -> Tuple[List[float], List[float], float, float]:
    """
    Computes mean x and y values within bins, and optionally residuals.

    Parameters
    ----------
    y : numpy.ndarray
        The dependent variable to bin.
    x : numpy.ndarray
        The independent variable to bin.
    controls : array-like or sparse matrix or None, default=None
        Control variables to use for residualization. If provided, residuals will be computed
        and used to plot a regression line.
    n_bins : int
        The number of bins to use in the plot.
    recenter_x : bool
        Whether to recenter residualized x by adding the mean of the original x.
    recenter_y : bool
        Whether to recenter the dependent variable y by adding its mean.
    bins : iterable of slice objects or None, default=None
        The bins to use in the plot. If None, equal-sized bins will be used.

    Returns
    -------
    x_means : list of floats
        The mean x values within each bin.
    y_means : list of floats
        The mean y values within each bin.
    intercept : float
        The intercept of the regression line, if residuals were computed.
    coef : float
        The slope of the regression line, if residuals were computed.

    """

    if controls is None:
        if np.any(np.diff(x) < 0):
            x, y = np.sort([x, y], axis = 1)
        x_data, y_data = x, y
    else:
        if np.ndim(controls) == 1:
            controls = np.expand_dims(controls, 1)

        demeaning_y_reg = linear_model.LinearRegression().fit(controls, y)
        y_data = y - demeaning_y_reg.predict(controls)

        demeaning_x_reg = linear_model.LinearRegression().fit(controls, x)
        x_data = x - demeaning_x_reg.predict(controls)
        x_data, y_data = np.sort([x_data, y_data], axis = 1)

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


def binscatter(self, x: npt.ArrayLike, y: npt.ArrayLike, controls = None, n_bins = 20,
               line_kwargs: Optional[Dict] = None, scatter_kwargs: Optional[Dict] = None, recenter_x: bool = False,
               recenter_y: bool = True, bins: Optional[Iterable[slice]] = None, fit_reg: Optional[bool] = True) -> \
Tuple[List[float], List[float], float, float]:
    """
    Plots a binned scatter plot with optional regression line.

    Parameters
    ----------
    self : matplotlib.axes.Axes object
        The plot to which to add the binned scatter plot.
    y : array-like
        The dependent variable to plot.
    x : array-like
        The independent variable to plot.
    controls : array-like or sparse matrix, default=None
        Control variables to use for residualization. If provided, residuals will be plotted
        against residualized x.
    n_bins : int, default=20
        The number of bins to use in the plot.
    line_kwargs : dict or None, default=None
        Keyword arguments to pass to the regression line plot.
    scatter_kwargs : dict or None, default=None
        Keyword arguments to pass to the scatter plot.
    recenter_x : bool, default=False
        Whether to recenter residualized x by adding the mean of the original x.
    recenter_y : bool, default=True
        Whether to recenter the dependent variable y by adding its mean.
    bins : iterable of slice objects or None, default=None
        The bins to use in the plot. If None, equal-sized bins will be used.
    fit_reg : bool or None, default=True
        Whether to plot a regression line. If None, no regression line will be plotted.

    Returns
    -------
    x_means : list of floats
        The mean x values within each bin.
    y_means : list of floats
        The mean y values within each bin.
    intercept : float
        The intercept of the regression line.
    coef : float
        The slope of the regression line.
    """

    if line_kwargs is None:
        line_kwargs = {}
    elif not fit_reg:
        warnings.warn("Both fit_reg=False and non-None line_kwargs were passed.")
    if scatter_kwargs is None:
        scatter_kwargs = {}

    x_means, y_means, intercept, coef = get_binscatter_objects(np.asarray(y), np.asarray(x), controls, n_bins,
        recenter_x, recenter_y, bins)

    self.scatter(x_means, y_means, **scatter_kwargs)
    x_range = np.array(self.get_xlim())
    if fit_reg:
        self.plot(x_range, intercept + x_range * coef, label = "beta=" + str(round(coef, 3)), **line_kwargs)

    if hasattr(x, "name"):
        self.set_xlabel(x.name)
    if hasattr(y, "name"):
        self.set_ylabel(y.name)
    return x_means, y_means, intercept, coef
