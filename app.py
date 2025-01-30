import numpy as np
import scipy.stats as stats
import enum
import matplotlib.pyplot as plt
import streamlit as st

# Set up the title and description
st.title("Option Pricing using Monte Carlo Simulation")
st.write("This app calculates the price of European and Cash-or-Nothing options using the Euler and Milstein schemes.")

# Option type enumeration
class OptionType(enum.Enum):
    Call = 1.0
    Put = -1.0

# Black-Scholes formula for call/put option price
def bs_call_put_option_price(cp, S_0, K, sigma, tau, r):
    K = np.array(K).reshape([len(K), 1])
    d1 = ((np.log(S_0 / K) + (r + 0.5 * sigma**2) * tau)) / float(sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if cp == OptionType.Call:
        value = (stats.norm.cdf(d1) * S_0) - (K * np.exp(-r * tau) * stats.norm.cdf(d2))
    elif cp == OptionType.Put:
        value = (K * np.exp(-r * tau) * stats.norm.cdf(-d2)) - (stats.norm.cdf(-d1) * S_0)
    return value
def BS_Cash_Or_Nothing_Price(cp,S_0,K,sigma,tau,r):
    K=np.array(K).reshape(len(K),1)
    d1=((np.log(S_0/K)+(r+0.5*sigma**2)*tau))/float(sigma*np.sqrt(tau))
    d2=d1-sigma*np.sqrt(tau)
    if cp==OptionType.Call:
        value=K*np.exp(-r*tau)*stats.norm.cdf(d2)
    elif cp==OptionType.Put:
        value=K*np.exp(-r*tau)*(1.0-stats.norm.cdf(d2))
    return value
# Generate paths using Euler discretization
@st.cache_data
def GeneratePathsGBMEuler(NoOfPaths, NoOfSteps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    S1 = np.zeros([NoOfPaths, NoOfSteps + 1])
    S1[:, 0] = S_0
    time = np.zeros([NoOfSteps + 1])
    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])  # Normalization
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        S1[:, i + 1] = S1[:, i] + r * S1[:, i] * dt + sigma * S1[:, i] * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt
    paths = {"time": time, "S": S1}
    return paths

# Generate paths using Milstein discretization
@st.cache_data
def GeneratePathsGBMMilstein(NoOfPaths, NoOfSteps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    S1 = np.zeros([NoOfPaths, NoOfSteps + 1])
    S1[:, 0] = S_0
    time = np.zeros([NoOfSteps + 1])
    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])  # Normalization
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        S1[:, i + 1] = S1[:, i] + r * S1[:, i] * dt + sigma * S1[:, i] * (W[:, i + 1] - W[:, i]) \
                       + 0.5 * sigma**2.0 * S1[:, i] * ((W[:, i + 1] - W[:, i])**2 - dt)
        time[i + 1] = time[i] + dt
    paths = {"time": time, "S": S1}
    return paths

# European option price from Monte Carlo paths
def europianPriceFromMCPaths(cp, S, K, T, r):
    if cp == OptionType.Call:
        return np.exp(-r * T) * np.mean(np.maximum(S - K, 0.0))
    elif cp == OptionType.Put:
        return np.exp(-r * T) * np.mean(np.maximum(K - S, 0.0))

# Cash-or-nothing option price from Monte Carlo paths
def cashorNothingPriceFromMCPaths(cp, S, K, T, r):
    if cp == OptionType.Call:
        return np.exp(-r * T) * K * np.mean((S > K))
    elif cp == OptionType.Put:
        return np.exp(-r * T) * K * np.mean((S <= K))

# Plotting function
def plot_results(option_type, PathsEuler, PathsMilstein, NoOfPathsV, pricesEuler, pricesMilstein, exactPrice,num_paths_to_plot):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Simulated Paths
    ax1.plot(PathsEuler["time"], PathsEuler["S"][:num_paths_to_plot, :].T, linewidth=0.5)
    ax1.set_title(f"First {num_paths_to_plot} Paths (Euler)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Asset Price")
    ax1.grid()

    ax2.plot(PathsMilstein["time"], PathsMilstein["S"][:num_paths_to_plot, :].T, linewidth=0.5)
    ax2.set_title(f"First {num_paths_to_plot} Paths (Milstein)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Asset Price")
    ax2.grid()
    st.pyplot(fig1)

    # Price Convergence
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(NoOfPathsV, pricesEuler, 'b-o', label="Euler Scheme")
    ax.plot(NoOfPathsV, pricesMilstein, 'g--s', label="Milstein Scheme")
    ax.axhline(y=exactPrice, color="r", linestyle="-", label="Exact Price")
    ax.set_title(f"Price Convergence ({option_type} Option)")
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Option Price")
    ax.legend()
    ax.grid()
    st.pyplot(fig2)

    # Error Comparison
    errorsEuler = np.abs(np.array(pricesEuler) - exactPrice)
    errorsMilstein = np.abs(np.array(pricesMilstein) - exactPrice)

    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(NoOfPathsV, errorsEuler, 'b-o', label="Euler Scheme Error")
    ax.plot(NoOfPathsV, errorsMilstein, 'g--s', label="Milstein Scheme Error")
    ax.set_title(f"Error Comparison ({option_type} Option)")
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Absolute Error")
    ax.legend()
    ax.grid()
    st.pyplot(fig3)

# Main function
def main():
    # Input parameters
    st.sidebar.header("Input Parameters")
    S_0 = st.sidebar.number_input("Initial Stock Price (S_0)", value=5.0)
    K_input = st.sidebar.text_input("Strike Prices (K) - Comma Separated", value="5.0")
    T = st.sidebar.number_input("Time to Maturity (T)", value=1.0)
    r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.06)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.3)
    NoOfSteps = st.sidebar.number_input("Number of Time Steps", value=1000)

    NoOfPathsV = [100, 1000, 5000, 10000]
    if st.sidebar.checkbox("Custom Number of Paths"):
        morepaths = st.sidebar.text_input("Number of Paths - Comma Separated", value="100,1000,5000,10000")
        extrapaths = [int(n.strip()) for n in morepaths.split(",")]
        NoOfPathsV.extend(extrapaths)

    num_paths_to_plot = st.sidebar.slider("Number of Paths to Plot", 1, 500, 20,
                                        help="Reduce this number for faster plotting")

    try:
        K = np.array([float(k.strip()) for k in K_input.split(",")])
    except:
        st.error("Invalid input for strike prices. Please enter comma-separated numbers.")
        return

    col2,col3=st.columns(2)
    with col2:
        cp = st.radio("Select Call or Put", [OptionType.Call, OptionType.Put])
    with col3:
        option_type = st.radio("Select Option Type", ["European", "Cash-or-Nothing"])

    # Initialize session state
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False

    if st.button("Calculate"):
        st.session_state.calculated = True

        st.session_state.option_type = option_type

        if option_type == "European":
            st.session_state.exactPrice = bs_call_put_option_price(cp, S_0, K, sigma, T, r)[0]
            st.session_state.pricesEuler = []
            st.session_state.pricesMilstein = []

            for noOfpathtemp in NoOfPathsV:
                PathsEuler = GeneratePathsGBMEuler(noOfpathtemp, NoOfSteps, T, r, sigma, S_0)
                PathsMilstein = GeneratePathsGBMMilstein(noOfpathtemp, NoOfSteps, T, r, sigma, S_0)
                st.session_state.pricesEuler.append(europianPriceFromMCPaths(cp, PathsEuler["S"][:, -1], K, T, r))
                st.session_state.pricesMilstein.append(europianPriceFromMCPaths(cp, PathsMilstein["S"][:, -1], K, T, r))

            # Store last paths for plotting
            st.session_state.PathsEuler = PathsEuler
            st.session_state.PathsMilstein = PathsMilstein

        elif option_type == "Cash-or-Nothing":
            st.session_state.exactPrice = BS_Cash_Or_Nothing_Price(cp, S_0, K, sigma, T, r)
            st.session_state.pricesEuler = []
            st.session_state.pricesMilstein = []

            for noOfpathtemp in NoOfPathsV:
                PathsEuler = GeneratePathsGBMEuler(noOfpathtemp, NoOfSteps, T, r, sigma, S_0)
                PathsMilstein = GeneratePathsGBMMilstein(noOfpathtemp, NoOfSteps, T, r, sigma, S_0)
                st.session_state.pricesEuler.append(cashorNothingPriceFromMCPaths(cp, PathsEuler["S"][:, -1], K, T, r))
                st.session_state.pricesMilstein.append(
                    cashorNothingPriceFromMCPaths(cp, PathsMilstein["S"][:, -1], K, T, r))

            # Store last paths for plotting
            st.session_state.PathsEuler = PathsEuler
            st.session_state.PathsMilstein = PathsMilstein

    if st.session_state.calculated:
        st.header(f"{st.session_state.option_type} Option Results")
        st.markdown(f"### Exact option price = {float(st.session_state.exactPrice):.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Euler Scheme Prices")
            for i, n in enumerate(NoOfPathsV):
                st.write(f"N = {n}: {float(st.session_state.pricesEuler[i]):.4f}")

        with col2:
            st.markdown("### Milstein Scheme Prices")
            for i, n in enumerate(NoOfPathsV):
                st.write(f"N = {n}: {float(st.session_state.pricesMilstein[i]):.4f}")

        if st.button("Generate Plots"):
            plot_results(
                st.session_state.option_type,
                st.session_state.PathsEuler,
                st.session_state.PathsMilstein,
                NoOfPathsV,
                st.session_state.pricesEuler,
                st.session_state.pricesMilstein,
                st.session_state.exactPrice,
                num_paths_to_plot
            )


# Run the app
if __name__ == "__main__":
    main()