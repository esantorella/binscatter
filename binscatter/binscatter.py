"""
Inspired by Stata's binscatter, described fully by Michael Stepner at
https://michaelstepner.com/binscatter/.
"""
from typing import Iterable, Union
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy import sparse as sps


def get_binscatter_objects(y: np.ndarray, x: np.ndarray, controls, weights: Union[None, np.ndarray],
                           n_bins: int, recenter_x: bool, recenter_y: bool, bins: Iterable):
    """
    Returns mean x and mean y within each bin, and coefficients if residualizing.
    Parameters are essentially the same as in binscatter.
    """
    def _weighted_mean(a: np.ndarray, w: Union[None, np.ndarray]):
        if w is None:
            return a.mean()
        return np.squeeze(a).dot(w) / w.sum()

    # Check if data is sorted. If not, sort it.
    if controls is None:
        if np.any(np.diff(x) < 0):
            argsort = np.argsort(x)
            x = x[argsort]
            y = y[argsort]
            if weights is not None:
                weights = weights[argsort]
        x_data = x
        y_data = y
    else:
        # Residualize
        if controls.ndim == 1:
            controls = controls[:, None]

        demeaning_y_reg = linear_model.LinearRegression().fit(controls, y, sample_weight=weights)
        y_data = y - demeaning_y_reg.predict(controls)

        demeaning_x_reg = linear_model.LinearRegression().fit(controls, x, sample_weight=weights)
        x_data = x - demeaning_x_reg.predict(controls)
        argsort = np.argsort(x_data)
        x_data = x_data[argsort]
        y_data = y_data[argsort]
        if weights is not None:
            weights = weights[argsort]

        if recenter_y:
            y_data += _weighted_mean(y, weights)
        if recenter_x:
            x_data += _weighted_mean(x, weights)

    if x_data.ndim == 1:
        x_data = x_data[:, None]
    reg = linear_model.LinearRegression().fit(x_data, y_data, sample_weight=weights)
    if bins is None:
        if weights is None:
            bin_edges = np.linspace(0, len(y), n_bins + 1).astype(int)
        else:
            cum_weight = np.cumsum(weights) / weights.sum()
            cum_weight_per_bin = np.linspace(0, 1, n_bins + 1)
            # Caution: Memory-inefficient in large data sets
            # Constructs a matrix of size (n rows) x (n bins)
            bin_edges = np.argmin(np.abs(cum_weight[:, None] - cum_weight_per_bin[None, :]), 0)

        assert len(bin_edges) == n_bins + 1
        bins = [slice(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
        assert len(bins) == n_bins

    x_means = [_weighted_mean(x_data[bin_], None if weights is None else weights[bin_])
               for bin_ in bins]
    y_means = [_weighted_mean(y_data[bin_], None if weights is None else weights[bin_])
               for bin_ in bins]

    return x_means, y_means, reg.intercept_, reg.coef_[0]


def binscatter(self: matplotlib.axes.Axes,
               x: Iterable,
               y: Iterable,
               controls=None,
               weights: Iterable=None,
               n_bins=20,
               line_kwargs: dict=None,
               scatter_kwargs: dict=None,
               recenter_x=False,
               recenter_y=True,
               bins: Iterable=None):
    """
    :param self: matplotlib.axes.Axes object.
        i.e., fig, axes = plt.subplots(3)
              axes[0].binscatter(x, y)

    :param y: Something that can be converted to 1d numpy array, e.g. Numpy array, Pandas Series, or list
    :param x: Something that can be converted to 1d numpy array, e.g. Numpy array, Pandas Series, or list
    :param controls: numpy array or sparse matrix
    :param weights: See x, y
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
        if isinstance(controls, pd.SparseDataFrame) or isinstance(controls, pd.SparseSeries):
            controls = controls.to_coo()
        elif isinstance(controls, pd.DataFrame) or isinstance(controls, pd.Series):
            controls = controls.values
        if not (isinstance(controls, np.ndarray) or sps.issparse(controls)):
            raise TypeError("""Controls must be a Pandas DataFrame, Pandas
            SparseDataFrame, Numpy array, or Scipy Sparse matrix""")
    if weights is not None:
        weights = np.asarray(weights)

    x_means, y_means, intercept, coef = get_binscatter_objects(np.asarray(y), np.asarray(x),
                                                               controls, weights,
                                                               n_bins, recenter_x,
                                                               recenter_y, bins)

    self.scatter(x_means, y_means, **scatter_kwargs)
    x_range = np.array(self.get_xlim())
    self.plot(x_range, intercept + x_range * coef, label='beta=' + str(round(coef, 3)), **line_kwargs)
    # If pd.Series was passed, might be able to label
    if hasattr(x, 'name'):
        self.set_xlabel(x.name)
    if hasattr(y, 'name'):
        self.set_ylabel(y.name)

    return x_means, y_means, intercept, coef

matplotlib.axes.Axes.binscatter = binscatter


def main():
    n_obs = 1000
    data = pd.DataFrame({'experience': np.random.poisson(4, n_obs) + 1})
    data['tenure'] = data['experience'] + np.random.normal(0, 1, n_obs)
    data['wage'] = data['experience'] + data['tenure'] + np.random.normal(0, 1, n_obs)
    data['weight'] = np.random.uniform(1, 2, len(data))

    fig, axes = plt.subplots(1, 2)
    axes[0].binscatter(data['wage'], data['tenure'], weights=data['weight'])
    axes[0].legend()
    axes[0].set_ylabel('Wage')
    axes[0].set_ylabel('Tenure')
    axes[0].set_title('No controls')
    axes[1].binscatter(data['wage'], data['tenure'], controls=data['experience'], weights=data['weight'])
    axes[1].set_xlabel('Tenure (residualized)')
    axes[1].set_ylabel('Wage (residualized, recentered)')
    axes[1].legend()
    axes[1].set_title('Controlling for experience')
    plt.savefig('test')
    plt.close('all')

    # Make y more interpretable
    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].binscatter(data['wage'], data['tenure'])
    axes[0].legend()
    axes[0].set_ylabel('Wage')
    axes[0].set_ylabel('Tenure')
    axes[0].set_title('No controls')
    axes[1].binscatter(data['wage'], data['tenure'], controls=data['experience'], recenter_y=True)
    axes[1].set_xlabel('Tenure (residualized, recentered)')
    # axes[1].set_ylabel('Wage (residualized, recentered)')
    axes[1].legend()
    axes[1].set_title('Controlling for experience')
    plt.savefig('test2')
    plt.close('all')
    return

if __name__ == '__main__':
    main()
