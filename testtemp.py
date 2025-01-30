import numpy as np
import scipy.stats as sats
import enum
import matplotlib.pyplot as plt
import streamlit as st

i = 1j  # setting large imagnery number


class OptionType(enum.Enum):
    Call = 1.0
    Put = -1.0


# bs
def bs_call_put_option_price(cp, S_0, K, sigma, tau, r):
    K = np.array(K).reshape([len(K), 1])
    d1 = ((np.log(S_0 / K) + (r + 0.5 * sigma ** 2) * tau)) / float(sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if cp == OptionType.Call:
        value = (sats.norm.cdf(d1) * S_0) - (K * np.exp(-r * tau) * sats.norm.cdf(d2))
    elif cp == OptionType.Put:
        value = (K * np.exp(-r * tau) * sats.norm.cdf(-d2)) - (sats.norm.cdf(-d1) * S_0)
    return value


# generating gbm using euiler discritisation
def GeneratePathsGBMEuler(NoOfPaths, NoOfSteps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    # eulier apprximation s1
    S1 = np.zeros([NoOfPaths, NoOfSteps + 1])
    S1[:, 0] = S_0
    time = np.zeros([NoOfSteps + 1])
    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])  # normalized your bitchs
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        S1[:, i + 1] = S1[:, i] + r * S1[:, i] * dt + sigma * S1[:, i] * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt
    paths = {'time': time, 'S': S1}
    return paths


def BS_Cash_Or_Nothing_Price(cp, S_0, K, sigma, tau, r):
    K = np.array(K).reshape(len(K), 1)
    d1 = ((np.log(S_0 / K) + (r + 0.5 * sigma ** 2) * tau)) / float(sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if cp == OptionType.Call:
        value = K * np.exp(-r * tau) * sats.norm.cdf(d2)
    elif cp == OptionType.Put:
        value = K * np.exp(-r * tau) * (1.0 - sats.norm.cdf(d2))
    return value


def GeneratePathsGBMMilstein(NoOfPaths, NoOfSteps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    # eulier apprximation s1
    S1 = np.zeros([NoOfPaths, NoOfSteps + 1])
    S1[:, 0] = S_0
    time = np.zeros([NoOfSteps + 1])
    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])  # normalized your bitchs
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        S1[:, i + 1] = S1[:, i] + r * S1[:, i] * dt + sigma * S1[:, i] * (
                    W[:, i + 1] - W[:, i]) + 0.5 * sigma ** 2.0 * S1[:, i] * ((W[:, i + 1] - W[:, i]) ** 2 - dt)
        time[i + 1] = time[i] + dt
    paths = {'time': time, 'S': S1}
    return paths


def europianPriceFromMCPaths(cp, S, K, T, r):
    if cp == OptionType.Call:
        return np.exp(-r * T) * np.mean(np.maximum(S - K, 0.0))
    elif cp == OptionType.Put:
        return np.exp(-r * T) * np.mean(np.maximum(K - S, 0.0))


def cashorNothingPriceFromMCPaths(cp, S, K, T, r):
    if cp == OptionType.Call:
        return np.exp(-r * T) * K * np.mean((S > K))
    elif cp == OptionType.Put:
        return np.exp(-r * T) * K * np.mean((S <= K))


def plot_paths(NoOfSteps, T, r, sigma, S_0):
    """Improved path plotting function"""
    NoOfPaths = 10  # Direct integer input

    np.random.seed(1)
    PathsEuler = GeneratePathsGBMEuler(NoOfPaths, NoOfSteps, T, r, sigma, S_0)
    PathsMilstein = GeneratePathsGBMMilstein(NoOfPaths, NoOfSteps, T, r, sigma, S_0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Euler paths
    ax1.plot(PathsEuler["time"], PathsEuler["S"].T, linewidth=0.5)
    ax1.set_title("Euler Scheme Paths")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Asset Price")

    # Milstein paths
    ax2.plot(PathsMilstein["time"], PathsMilstein["S"].T, linewidth=0.5)
    ax2.set_title("Milstein Scheme Paths")
    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.show()


def plot_results(x_values, y_values_list, labels, title, xlabel, ylabel, figure_num, exact_price=None):
    """
    Efficient plotting function for multiple series with common formatting
    """
    plt.figure(figure_num)
    for y_values, label in zip(y_values_list, labels):
        plt.plot(x_values, y_values, label=label)

    if exact_price is not None:
        plt.axhline(y=exact_price, color="r", linestyle="--", label="Exact Price")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.tight_layout()


def mainCalculation():
    # Parameter setup
    cp = OptionType.Call
    T = 1
    r = 0.06
    sigma = 0.3
    S_0 = 5
    K = [S_0]
    NoOfSteps = 1000
    NoOfPathsV = [100, 1000, 5000, 10000]

    # European option pricing
    print("\nEUROPEAN OPTION PRICING")
    exactPrice = bs_call_put_option_price(cp, S_0, K, sigma, T, r)[0]
    print(f"Exact BS price: {float(exactPrice):.4f}")

    # Monte Carlo simulation
    pricesEuler, pricesMilstein = [], []
    for N in NoOfPathsV:
        # Path generation
        PathsEuler = GeneratePathsGBMEuler(N, NoOfSteps, T, r, sigma, S_0)
        PathsMilstein = GeneratePathsGBMMilstein(N, NoOfSteps, T, r, sigma, S_0)

        # Price calculation
        priceEuler = europianPriceFromMCPaths(cp, PathsEuler["S"][:, -1], K, T, r)
        priceMilstein = europianPriceFromMCPaths(cp, PathsMilstein["S"][:, -1], K, T, r)

        pricesEuler.append(priceEuler)
        pricesMilstein.append(priceMilstein)

        print(f"N={N:5d}: Euler={float(priceEuler):.4f} (Error: {float(priceEuler - exactPrice):.4f})"
              f" | Milstein={float(priceMilstein):.4f} (Error: {float(priceMilstein - exactPrice):.4f})")

    # European option plots
    plot_results(NoOfPathsV, [pricesEuler, pricesMilstein],
                 ["Euler", "Milstein"],
                 "European Option Price Convergence",
                 "Number of Paths", "Price", 1, exactPrice)

    # Error analysis
    errorsEuler = np.abs(pricesEuler - exactPrice)
    errorsMilstein = np.abs(pricesMilstein - exactPrice)

    plot_results(NoOfPathsV, [errorsEuler, errorsMilstein],
                 ["Euler Error", "Milstein Error"],
                 "European Option Pricing Errors",
                 "Number of Paths", "Absolute Error", 2)

    # Cash-or-Nothing pricing
    print("\nCASH-OR-NOTHING OPTION PRICING")
    exactCash = BS_Cash_Or_Nothing_Price(cp, S_0, K, sigma, T, r)[0]
    print(f"Exact cash-or-nothing price: {float(exactCash):.4f}")

    cashEuler, cashMilstein = [], []
    for N in NoOfPathsV:
        PathsEuler = GeneratePathsGBMEuler(N, NoOfSteps, T, r, sigma, S_0)
        PathsMilstein = GeneratePathsGBMMilstein(N, NoOfSteps, T, r, sigma, S_0)

        priceEuler = cashorNothingPriceFromMCPaths(cp, PathsEuler["S"][:, -1], K[0], T, r)
        priceMilstein = cashorNothingPriceFromMCPaths(cp, PathsMilstein["S"][:, -1], K[0], T, r)

        cashEuler.append(priceEuler)
        cashMilstein.append(priceMilstein)

        print(f"N={N:5d}: Euler={float(priceEuler):.4f} (Error: {float(priceEuler - exactCash):.4f})"
              f" | Milstein={float(priceMilstein):.4f} (Error: {float(priceMilstein - exactCash):.4f})")

    # Cash-or-Nothing plots
    plot_results(NoOfPathsV, [cashEuler, cashMilstein],
                 ["Euler", "Milstein"],
                 "Cash-or-Nothing Option Price Convergence",
                 "Number of Paths", "Price", 3, exactCash)

    # Error analysis
    errorsCashEuler = np.abs(cashEuler - exactCash)
    errorsCashMilstein = np.abs(cashMilstein - exactCash)

    plot_results(NoOfPathsV, [errorsCashEuler, errorsCashMilstein],
                 ["Euler Error", "Milstein Error"],
                 "Cash-or-Nothing Option Pricing Errors",
                 "Number of Paths", "Absolute Error", 4)

    # Show path plots
    plot_paths(NoOfSteps, T, r, sigma, S_0)


# Run main calculation
if __name__ == "__main__":
    mainCalculation()