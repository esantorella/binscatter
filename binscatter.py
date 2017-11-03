"""
Inspired by Stata's binscatter, described fully by Michael Stepner at
https://michaelstepner.com/binscatter/.
"""
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import linear_model
import warnings


def get_binscatter_objects(data, y, x, controls, n_bins, recenter_x):
    """
    Returns x_means, y_means, coefficients.
    :param data:
    :param y:
    :param x:
    :param controls:
    :param n_bins:
    :return:
    """
    # Check if data is sorted
    if np.any(np.diff(data[x]) < 0):
        warnings.warn('Data was not sorted. Going to sort.')
        data = data.sort_values(x).reset_index(drop=True)
    if controls is None:
        x_data = data[x]
        y_data = data[y]
    else:
        # Residualize
        assert set(controls).issubset(set(data.columns))
        demeaning_y_reg = linear_model.LinearRegression().fit(data[controls], data[y])
        y_data = data[y] - demeaning_y_reg.predict(data[controls]) + data[y].mean()
        demeaning_x_reg = linear_model.LinearRegression().fit(data[controls], data[x])
        x_data = data[x] - demeaning_x_reg.predict(data[controls])
        if recenter_x:
            x_data += data[x].mean()

    reg = linear_model.LinearRegression().fit(x_data[:, None], y_data)
    bin_edges = np.linspace(0, len(data), n_bins + 1).astype(int)
    assert len(bin_edges) == n_bins + 1
    bins = [slice(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    assert len(bins) == n_bins
    x_means = [np.mean(x_data[bin_]) for bin_ in bins]
    y_means = [np.mean(y_data[bin_]) for bin_ in bins]

    return x_means, y_means, reg.intercept_, reg.coef_[0]


def binscatter(self, data, y, x, controls=None, n_bins=20,
               line_kwargs=None, scatter_kwargs=None, recenter_x=False):
    """

    :param self: matplotlib.axes.Axes object
    :param data: pandas DataFrame
    :param y: string, referring to a column in data
    :param x:
    :param controls: list of strings; all should be columns in data
    :param n_bins: int, default 20
    :param line_kwargs: dict
    :param scatter_kwargs: dict
    :param recenter_x: If true, recenter x the same way y was recentered
    :return:
    """
    if line_kwargs is None:
        line_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}

    x_means, y_means, intercept, coef = get_binscatter_objects(data, y, x, controls, n_bins,
                                                               recenter_x)

    self.scatter(x_means, y_means, **scatter_kwargs)
    x_range = np.array(self.get_xlim())
    self.plot(x_range, intercept + x_range * coef, label='beta=' + str(coef), **line_kwargs)
    return self

matplotlib.axes.Axes.binscatter = binscatter

if __name__ == '__main__':
    n_obs = 1000
    data = pd.DataFrame({'experience': np.random.poisson(4, n_obs) + 1})
    data['tenure'] = data['experience'] + np.random.normal(0, 1, n_obs)
    data['wage'] = data['experience'] + data['tenure'] + np.random.normal(0, 1, n_obs)

    fig, axes = plt.subplots(2)
    axes[0].binscatter(data, 'wage', 'tenure')
    axes[0].legend()
    axes[0].set_ylabel('Wage')
    axes[0].set_ylabel('Tenure')
    axes[0].set_title('No controls')
    axes[1].binscatter(data, 'wage', 'tenure', controls=['experience'])
    axes[1].set_xlabel('Tenure (residualized)')
    axes[1].set_ylabel('Wage (residualized, recentered)')
    axes[1].legend()
    axes[1].set_title('Controlling for experience')
    plt.savefig('test')
    plt.close('all')

    # Make x more interpretable
    fig, axes = plt.subplots(2, sharex=True, sharey=True)
    axes[0].binscatter(data, 'wage', 'tenure')
    axes[0].legend()
    axes[0].set_ylabel('Wage')
    axes[0].set_ylabel('Tenure')
    axes[0].set_title('No controls')
    axes[1].binscatter(data, 'wage', 'tenure', controls=['experience'], recenter_x=True)
    axes[1].set_xlabel('Tenure (residualized, recentered)')
    axes[1].set_ylabel('Wage (residualized, recentered)')
    axes[1].legend()
    axes[1].set_title('Controlling for experience')
    plt.savefig('test2')
    plt.close('all')
