# Statistical Arbitrage via Continuous-Time Ornstein–Uhlenbeck Modeling

**Production-grade quantitative research framework for pairs trading on cointegrated equities, featuring exact Gaussian Maximum Likelihood calibration, discretization bias analysis, and out-of-sample execution under realistic market frictions.**

---

## Overview

Heuristic pairs trading strategies (such as rolling Bollinger Bands) treat price spread deviations through static, discrete approximations. They typically overlook continuous-time transition dynamics, mean-reversion half-lives, finite-step discretization bias, and trade-coherent hedge ratio locking.

This repository provides an end-to-end quantitative research framework that formulates statistical arbitrage through continuous-time stochastic calculus. Focusing on dual-class equity pairs sharing identical corporate fundamentals (Alphabet Class A `GOOGL` and Class C `GOOG`), the system:
1. Conducts econometric cointegration pre-testing (Engle–Granger and Augmented Dickey–Fuller tests).
2. Quantifies discrete vs. continuous calibration discrepancy (AR(1) OLS vs. exact Gaussian Maximum Likelihood Estimation).
3. Executes a market-neutral state machine strategy with hysteresis entry/exit thresholds and stop-loss bounds.
4. Accounts for institutional execution frictions, including exchange transaction fees, bid-ask slippage, and short-borrow financing costs.

---

## Mathematical Formulation

### The Ornstein–Uhlenbeck (OU) SDE
We model the log-price spread $X_t \in \mathbb{R}$ as an Ornstein–Uhlenbeck diffusion process:
$$dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t$$

where:
- $\theta > 0$ is the annualized mean-reversion speed,
- $\mu \in \mathbb{R}$ is the long-term equilibrium mean spread,
- $\sigma > 0$ is the continuous diffusion volatility,
- $W_t$ is a standard one-dimensional Brownian motion on a filtered probability space $(\Omega, \mathcal{F}, \{\mathcal{F}_t\}_{t \ge 0}, \mathbb{P})$.

### Exact Transition Law and Stationary Moments
Applying Itô's Lemma to the integrating factor $Y_t = e^{\theta t}(X_t - \mu)$ yields the exact continuous-time solution over a time step $\Delta t$:
$$X_{t+\Delta t} = \mu + (X_t - \mu)e^{-\theta \Delta t} + \sigma \int_t^{t+\Delta t} e^{-\theta(t+\Delta t - s)}\,dW_s$$

By Itô's Isometry, the conditional distribution of $X_{t+\Delta t}$ given $X_t$ is strictly Gaussian:
$$X_{t+\Delta t} \mid X_t \sim \mathcal{N}\left(\mu + (X_t - \mu)e^{-\theta \Delta t},\; \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta \Delta t}\right)\right)$$

As $\Delta t \to \infty$, the process converges ergodically to its stationary distribution:
$$X_\infty \sim \mathcal{N}\left(\mu,\; \sigma_{\text{eq}}^2 = \frac{\sigma^2}{2\theta}\right)$$

The characteristic **mean-reversion half-life** represents the expected time required for a spread deviation to decay by half:
$$t_{1/2} = \frac{\ln(2)}{\theta}$$

---

## Econometric Pre-Testing & Model Calibration

### 1. Cointegration Pre-Testing
Before deploying the mean-reversion model, cointegration is established via the Engle–Granger two-step procedure:
$$\ln(\text{GOOGL}_t) = \alpha + \beta \ln(\text{GOOG}_t) + \varepsilon_t$$

Stationarity of the spread residuals $\varepsilon_t$ is confirmed via the Augmented Dickey–Fuller (ADF) test ($p$-value $< 0.01$).

### 2. Discretization Bias: Gaussian MLE vs. AR(1) OLS
Discrete implementations frequently approximate the OU drift via an AR(1) regression:
$$X_{k} = a + b X_{k-1} + \eta_k$$

Under naive Euler discretization, the continuous parameters are mapped as $\hat{\theta}_{\text{Euler}} = -\frac{b - 1}{\Delta t}$. However, the exact mapping requires:
$$\hat{\theta}_{\text{exact}} = -\frac{\ln(b)}{\Delta t}$$

For coarse sampling or high mean-reversion speeds, Euler AR(1) regression systematically underestimates the true mean-reversion speed. Our calibration module implements closed-form Gaussian Maximum Likelihood Estimation (MLE) based on the exact conditional density, recovering unbiased parameters regardless of sampling step size.

---

## Trading Architecture & Execution Model

The trading engine implements a market-neutral state machine designed to minimize unnecessary portfolio churn:

- **Standardized Dimensionless Z-Score:**
  $$Z_t = \frac{X_t - \hat{\mu}_t}{\hat{\sigma}_{\text{eq}, t}}$$
- **Hysteresis Entry / Exit Logic:**
  - **Enter Short Spread:** When $Z_t \ge +1.25$ (Short `GOOGL`, Long $\beta \times$`GOOG`).
  - **Enter Long Spread:** When $Z_t \le -1.25$ (Long `GOOGL`, Short $\beta \times$`GOOG`).
  - **Exit / Mean-Reversion:** When $|Z_t| \le 0.25$ (capturing the restorative swing without waiting for exact zero crossover).
  - **Stop-Loss Protection:** Immediate position liquidation if $|Z_t| \ge 3.00$ to prevent adverse trend divergence.
- **Trade-Coherent Hedge Locking:** The hedge ratio $\beta$ is locked upon position entry and maintained through exit, preventing spurious portfolio rebalancing driven by rolling estimation noise while a trade is active.
- **Friction Modeling:** Deducts round-trip transaction costs (5 bps baseline, stress-tested up to 50 bps) and short-borrow financing fees (25 bps annualized).

---

## Empirical Out-of-Sample Performance

The model was calibrated strictly on in-sample data (**2019–2022**, 1,008 trading days) and backtested out-of-sample (**2023–2024**, 502 trading days).

### Out-of-Sample Performance Metrics (2023–2024)

| Metric | OU Pairs Strategy (Unlevered) | OU Pairs Strategy (5x Levered) | Benchmark: Buy & Hold GOOGL |
| :--- | :---: | :---: | :---: |
| **Total Net Return** | **+2.49%** | **+12.44%** | +113.18% |
| **Compound Annual Growth Rate (CAGR)** | **1.24%** | **6.04%** | 46.23% |
| **Annualized Volatility** | **1.49%** | **7.46%** | 29.19% |
| **Sharpe Ratio (Rf = 0%)** | **0.83** | **0.83** | 1.45 |
| **Sortino Ratio** | **1.21** | **1.21** | 2.17 |
| **Maximum Drawdown** | **-1.36%** | **-6.80%** | **-22.14%** |
| **Calmar Ratio** | **0.91** | **0.91** | 2.09 |
| **Win Rate (Hit Rate)** | **94.4%** | **94.4%** | 56.8% |
| **Profit Factor** | **39.17** | **39.17** | 1.29 |
| **Total Completed Cycles** | 18 trades | 18 trades | Buy-and-Hold |
| **Average Holding Duration** | 20.3 trading days | 20.3 trading days | 501 trading days |

### Risk-Adjusted Assessment
- **Drawdown Compression:** The market-neutral OU strategy compressed maximum drawdown to just **-1.36%** (unlevered) and **-6.80%** (5x leverage), compared to **-22.14%** for passive single-stock exposure.
- **Execution Consistency:** Achieved a **94.4% win rate** across 18 completed cycles with a **39.17 profit factor**, demonstrating that statistical mean-reversion signals reliably monetized liquidity imbalances net of transaction costs.
- **Capital Efficiency:** Because the strategy operates market-neutral, margin capital can be deployed into short-term cash yielding risk-free return alongside the statistical arbitrage spread alpha.

---

## Visual Diagnostics & Sensitivity Analysis

The experimental framework generates high-resolution figures in `figures/`:

1. **`fig1_sde_simulations.png`**: Analytical vs. numerical path trajectories, finite-step variance bias, and strong convergence analysis.
2. **`fig2_calibration_diagnostics.png`**: Cointegration spread time series, ADF residual stationarity tests, and Gaussian MLE density overlays.
3. **`fig3_backtest_performance.png`**: Out-of-sample cumulative equity curves, underwater drawdown profiles, and active spread positioning.
4. **`fig4_sensitivity_analysis.png`**: Parameter stability surfaces ($Z_{\text{entry}}$ vs. $Z_{\text{exit}}$) across varying transaction fee tiers (0 to 50 bps).

---

## Repository Structure

```text
.
├── src/                          # Modular Python quantitative library
│   ├── __init__.py               # Package namespace and API exports
│   ├── models.py                 # SDE simulation engines (exact & Euler) and moments
│   ├── calibration.py            # Gaussian MLE and OLS estimators
│   ├── data_loader.py            # Market data ingestion, missing-value audit, SQLite caching
│   ├── strategy.py               # Signal generation and state-machine trading logic
│   └── backtest.py               # Performance attribution, friction models, and metrics
├── data/                         # Historical datasets and experiment output
│   ├── market_prices_goog.csv    # Daily price data for GOOGL/GOOG
│   ├── market_prices.csv         # Multi-asset pricing dataset
│   └── experiment_summary.json   # Machine-readable output metrics
├── figures/                      # High-resolution output figures (300 DPI)
│   ├── fig1_sde_simulations.png
│   ├── fig2_calibration_diagnostics.png
│   ├── fig3_backtest_performance.png
│   └── fig4_sensitivity_analysis.png
├── report/                       # Research paper
│   ├── report.tex                # Two-column LaTeX manuscript
│   └── report.pdf                # Compiled publication-ready PDF report
├── project_walkthrough.ipynb     # Interactive research notebook
├── run_experiments.py            # Reproducible end-to-end execution script
├── requirements.txt              # Pinned Python package dependencies
├── .gitignore                    # Git ignore configuration
└── README.md                     # Technical documentation
```

---

## Quickstart & Reproducibility

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Reproduce All Experiments
To execute the full simulation, calibration, and backtest pipeline:
```bash
python run_experiments.py
```
This runs the full quantitative workflow, generates `data/experiment_summary.json`, and updates all diagnostic figures in `figures/`.

### 3. Interactive Research Notebook
Launch the Jupyter notebook for a guided walkthrough of the mathematical derivations, simulations, and backtest diagnostics:
```bash
jupyter notebook project_walkthrough.ipynb
```

---

## Technical Paper & References

For full mathematical proofs, Itô expansions, and econometric derivations:
- **Research Paper:** [`report/report.pdf`](report/report.pdf) (LaTeX source: [`report/report.tex`](report/report.tex))

---

## Authors

- **Kavin Kashyap Albal**
- **Sanjan Ganesan**
- **Atharva Sandeep Agarwala**
- **Shaurya Manjkhola**
