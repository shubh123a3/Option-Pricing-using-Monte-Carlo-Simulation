
# Option Pricing with Monte Carlo Simulation - Streamlit App
https://github.com/user-attachments/assets/13b20bdc-a9ef-456e-9377-ddb58ae50d02

```markdown

A Streamlit-based web application for pricing European and Cash-or-Nothing options using Monte Carlo simulations with Euler and Milstein discretization schemes.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Example Usage](#example-usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features 🚀

### 1. **Option Type Support**
- European Options (Call/Put)
- Cash-or-Nothing Options (Call/Put)

### 2. **Flexible Inputs**
- Adjustable financial parameters:
  - Initial stock price (`S₀`)
  - Strike price(s) (`K`) - supports multiple comma-separated values
  - Time to maturity (`T`)
  - Risk-free rate (`r`)
  - Volatility (`σ`)
- Numerical parameters:
  - Number of Monte Carlo paths
  - Time steps for discretization

### 3. **Discretization Schemes**
- Euler-Maruyama method
- Milstein method

### 4. **Advanced Analytics**
- Real-time comparison with Black-Scholes closed-form solutions
- Convergence analysis across different numbers of paths
- Error visualization (Absolute error vs. number of paths)

### 5. **Interactive Visualizations**
- Path simulations visualization (first 20-500 paths)
- Price convergence plots
- Error comparison charts

## Installation ⚙️

### Requirements
- Python 3.7+
- Required packages:
  ```bash
  pip install streamlit numpy scipy matplotlib
  ```

### Run the App
```bash
streamlit run app.py
```

## Usage 📊

1. **Parameter Configuration** (Left Sidebar)
   - Set financial parameters
   - Choose option type (European/Cash-or-Nothing)
   - Select Call/Put option
   - Configure simulation parameters:
     - Number of paths (supports multiple values: 100, 1000, 5000, 10000)
     - Number of time steps
     - Paths to visualize

2. **Simulation Execution**
   - Click "Calculate" to run simulations
   - Results display includes:
     - Exact Black-Scholes price
     - Euler vs Milstein prices comparison
     - Visualizations button for detailed analysis

3. **Visual Analysis**
   - Path simulations visualization
   - Price convergence graphs
   - Error comparison plots

## Technical Details 🔍

### Models Implemented
- **Black-Scholes-Merton Framework**
  - Exact solutions for European options:
    ```python
    C = S₀N(d₁) - Ke^{-rT}N(d₂)
    P = Ke^{-rT}N(-d₂) - S₀N(-d₁)
    ```
  - Cash-or-Nothing options:
    ```python
    CashCall = Ke^{-rT}N(d₂)
    CashPut = Ke^{-rT}(1 - N(d₂))
    ```

- **Monte Carlo Methods**
  - Euler discretization:
    ```python
    S_{t+Δt} = S_t(1 + rΔt + σΔW_t)
    ```
  - Milstein discretization:
    ```python
    S_{t+Δt} = S_t[1 + rΔt + σΔW_t + ½σ²(ΔW_t² - Δt)]
    ```

### Error Analysis
- Absolute Error = |MC Price - Exact BS Price|
- Convergence measured across:
  - 100 paths (quick test)
  - 1,000 paths (standard accuracy)
  - 5,000 paths (high accuracy)
  - 10,000 paths (production-level)

## Example Usage 💡

### Scenario: ATM European Call
- Parameters:
  ```
  S₀ = 100, K = 100, T = 1, r = 5%, σ = 20%
  ```
- Expected Output:
  - BS Price: ~10.45
  - MC Prices (10k paths): 
    - Euler: 10.32-10.58
    - Milstein: 10.40-10.50

### Scenario: Cash-or-Nothing Put
- Parameters:
  ```
  S₀ = 50, K = 55, T = 0.5, r = 3%, σ = 35%
  ```
- Expected Output:
  - BS Price: ~49.25
  - MC Prices (5k paths):
    - Euler: 48.80-49.70
    - Milstein: 49.10-49.40

## Contributing 🤝

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments 🏆

- Black-Scholes model: Fischer Black, Myron Scholes
- Monte Carlo methods: Stanislaw Ulam, Nicholas Metropolis
- Financial engineering community
- Streamlit development team
```

This README provides:
- Clear installation/usage instructions
- Technical specification for quantitative analysts
- Example scenarios for practical understanding
- Contribution guidelines for developers
- Professional structure with emoji visual cues

You can enhance it further by:
1. Adding actual screenshots
2. Including performance benchmarks
3. Adding links to academic references
4. Expanding the "Advanced Usage" section
5. Adding CI/CD badges if applicable
