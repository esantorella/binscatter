import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import binscatter  # noqa: F401


def main():
    n_obs = 1000
    data = pd.DataFrame({"experience": np.random.poisson(4, n_obs) + 1})
    data["Tenure"] = data["experience"] + np.random.normal(0, 1, n_obs)
    data["Wage"] = data["experience"] + data["Tenure"] + np.random.normal(0, 1, n_obs)

    fig, axes = plt.subplots(1, 2)
    axes[0].binscatter(data["Tenure"], data["Wage"])
    axes[0].legend()
    axes[0].set_title("No controls")
    axes[1].binscatter(data["Tenure"], data["Wage"], controls=data["experience"])
    axes[1].set_xlabel("Tenure (residualized)")
    axes[1].set_ylabel("Wage (residualized, recentered)")
    axes[1].legend()
    axes[1].set_title("Controlling for experience")
    plt.tight_layout()
    plt.savefig("test")
    plt.close("all")

    # Make y more interpretable
    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].binscatter(data["Tenure"], data["Wage"])
    axes[0].legend()
    axes[0].set_title("No controls")
    axes[1].binscatter(
        data["Tenure"], data["Wage"], controls=data["experience"], recenter_y=True
    )
    axes[1].set_xlabel("Tenure (residualized)")
    axes[1].set_ylabel("Wage (residualized, recentered)")
    axes[1].legend()
    axes[1].set_title("Controlling for experience")
    plt.savefig("test2")
    plt.close("all")


def readme_example() -> None:

    # Create fake data
    n_obs = 1000
    data = pd.DataFrame({"experience": np.random.poisson(4, n_obs) + 1})
    data["Tenure"] = data["experience"] + np.random.normal(0, 1, n_obs)
    data["Wage"] = data["experience"] + data["Tenure"] + np.random.normal(0, 1, n_obs)
    fig, axes = plt.subplots(2, sharex=True, sharey=True)

    # Binned scatter plot of Wage vs Tenure
    axes[0].binscatter(data["Tenure"], data["Wage"])

    # Binned scatter plot that partials out the effect of experience
    axes[1].binscatter(
        data["Tenure"],
        data["Wage"],
        controls=data["experience"],
        recenter_x=True,
        recenter_y=True,
    )
    axes[1].set_xlabel("Tenure (residualized, recentered)")
    axes[1].set_ylabel("Wage (residualized, recentered)")

    plt.tight_layout()
    plt.savefig("readme_example")
    plt.close("all")


if __name__ == "__main__":
    main()
    readme_example()
