Inspired by Stata's binscatter, described fully by Michael Stepner at
https://michaelstepner.com/binscatter/.
Better description at http://esantorella.com/2017/11/03/binscatter/.

Binscatter consists of just one file. The easiest way to use binscatter is to 
copy the file into the directory the rest of your code is in and "import binscatter."

Binscatter has dependencies scipy, numpy, matplotlib, pandas, and sklearn.

Example:

```
# Create fake data
n_obs = 1000
data = pd.DataFrame({'experience': np.random.poisson(4, n_obs) + 1})
data['tenure'] = data['experience'] + np.random.normal(0, 1, n_obs)
data['wage'] = data['experience'] + data['tenure'] + np.random.normal(0, 1, n_obs)
fig, axes = plt.subplots(2)

# Binned scatter plot of wage vs tenure
axes[0].binscatter(data, 'wage', 'tenure')
axes[0].set_ylabel('Wage')
axes[0].set_ylabel('Tenure')

# Binned scatter plot that partials out the effect of experience
axes[1].binscatter(data, 'wage', 'tenure', controls=['experience'])
axes[1].set_xlabel('Tenure (residualized)')
axes[1].set_ylabel('Wage (residualized, recentered)')
plt.show()
```
