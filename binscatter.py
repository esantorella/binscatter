"""
Inspired by Stata's binscatter, described fully by Michael Stepner at
https://michaelstepner.com/binscatter/.
"""
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy import sparse as sps


def get_binscatter_objects(y, x, controls, n_bins, recenter_x, recenter_y, bins):
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
        bins = [slice(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
        assert len(bins) == n_bins

    x_means = [np.mean(x_data[bin_]) for bin_ in bins]
    y_means = [np.mean(y_data[bin_]) for bin_ in bins]

    return x_means, y_means, reg.intercept_, reg.coef_[0]


def binscatter(self, y, x, controls=None, n_bins=20,
               line_kwargs=None, scatter_kwargs=None, recenter_x=False,
               recenter_y=True, bins=None):
    """
    :param self: matplotlib.axes.Axes object.
        i.e., fig, axes = plt.subplots(3)
              axes[0].binscatter(y, x)

    :param y: 1d numpy array or pandas series
    :param x: 1d numpy array or Pandas Series
    :param controls: numpy array or sparse matrix
    :param n_bins: int, default 20
    :param line_kwargs: dict
    :param scatter_kwargs: dict
    :param recenter_y: If true, recenter y-tilde so its mean is the mean of y
    :param recenter_x: If true, recenter y-tilde so its mean is the mean of y
    :param bins: Indices of each bin, if you don't like the default
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
        assert isinstance(controls, np.ndarray) or sps.issparse(controls)

    x_means, y_means, intercept, coef = get_binscatter_objects(np.asarray(y), np.asarray(x),
                                                               controls, n_bins, recenter_x,
                                                               recenter_y, bins)

    self.scatter(x_means, y_means, **scatter_kwargs)
    x_range = np.array(self.get_xlim())
    self.plot(x_range, intercept + x_range * coef, label='beta=' + str(round(coef, 3)), **line_kwargs)
    # If series were passed, might be able to label
    try:
        self.set_xlabel(x.name)
    except AttributeError:
        pass
    try:
        self.set_ylabel(y.name)
    except AttributeError:
        pass
    return coef

matplotlib.axes.Axes.binscatter = binscatter


def main():
    n_obs = 1000
    data = pd.DataFrame({'experience': np.random.poisson(4, n_obs) + 1})
    data['tenure'] = data['experience'] + np.random.normal(0, 1, n_obs)
    data['wage'] = data['experience'] + data['tenure'] + np.random.normal(0, 1, n_obs)

    fig, axes = plt.subplots(1, 2)
    axes[0].binscatter(data['wage'], data['tenure'])
    axes[0].legend()
    axes[0].set_ylabel('Wage')
    axes[0].set_ylabel('Tenure')
    axes[0].set_title('No controls')
    axes[1].binscatter(data['wage'], data['tenure'], controls=data['experience'])
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
