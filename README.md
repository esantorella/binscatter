`binscatter` is inspired by Stata's binscatter, described fully by Michael Stepner
[here](https://michaelstepner.com/binscatter/). You can use it in essentially
the same way you use Matplotlib functions like `plot` and `scatter`.
A more extensive description of this package is [here](http://esantorella.com/2017/11/03/binscatter/).

## Getting started

1. _Copy and paste_: Binscatter's meaningful code consists of consists of just one file.
You can copy `binscatter/binscatter.py` into the directory the rest of your code is in.

2. _Install via pip_: To make it easier to use `binscatter` in multiple projects and directories,
open a terminal and
   - git clone https://github.com/esantorella/binscatter.git
   - cd binscatter
   - pip install .

## Usage

```
import binscatter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Create fake data
n_obs = 1000
data = pd.DataFrame({'experience': np.random.poisson(4, n_obs) + 1})
data['Tenure'] = data['experience'] + np.random.normal(0, 1, n_obs)
data['Wage'] = data['experience'] + data['Tenure'] + np.random.normal(0, 1, n_obs)
fig, axes = plt.subplots(2, sharex=True, sharey=True)

# Binned scatter plot of Wage vs Tenure
axes[0].binscatter(data['Tenure'], data['Wage'])

# Binned scatter plot that partials out the effect of experience
axes[1].binscatter(data['Tenure'], data['Wage'], controls=data['experience'],
                   recenter_x=True, recenter_y=True)
axes[1].set_xlabel('Tenure (residualized, recentered)')
axes[1].set_ylabel('Wage (residualized, recentered)')

plt.tight_layout()
plt.show()
```
