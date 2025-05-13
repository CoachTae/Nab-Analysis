import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def test_gaussian_fit_from_histogram(hist_dict, plot=True):
    """
    Tests whether histogram data resembles a Gaussian using multiple statistical tests.

    Parameters:
        hist_dict (dict): Keys are energy values (x), values are counts (y)
        plot (bool): Whether to show a histogram with fitted Gaussian

    Returns:
        dict: p-values for Shapiro-Wilk and Anderson-Darling tests
    """

    # Step 1: Expand histogram into raw sample list
    energies = []
    for energy, count in hist_dict.items():
        energies.extend([energy] * count)
    data = np.array(energies)

    # Step 2: Fit Gaussian to sample
    mu, std = np.mean(data), np.std(data)

    # Step 3: Run goodness-of-fit tests
    shapiro_stat, shapiro_p = stats.shapiro(data)
    ad_stat, ad_critical_values, ad_significance_levels = stats.anderson(data, dist='norm')

    # Step 4: Optional plot
    if plot:
        plt.hist(data, bins=100, density=True, alpha=0.6, label='Data')
        x = np.linspace(min(data), max(data), 1000)
        plt.plot(x, stats.norm.pdf(x, mu, std), 'r--', label='Fitted Gaussian')
        plt.legend()
        plt.title("Histogram with Gaussian Fit")
        plt.xlabel("Energy")
        plt.ylabel("Normalized Count")
        plt.show()

    return {
        "Shapiro-Wilk p": shapiro_p,
        "Anderson-Darling stat": ad_stat,
        "Anderson Critical Values": ad_critical_values,
        "Anderson Significance Levels": ad_significance_levels
    }
