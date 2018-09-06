from setuptools import setup

setup(
    packages=['binscatter'],

    install_requires=[
        'pandas', 'numpy', 'matplotlib', 'sklearn', 'scipy'
    ],
    zip_safe=False,
    entry_points={},

    name='binscatter',
    version='1.0.0',
    description='binned scatter plots with or without covariates',
    long_description='',

    url='https://github.com/esantorella/binscatter',
    author='Elizabeth Santorella',
    license='MIT',
    platforms=[''],
    classifiers=[]
)
