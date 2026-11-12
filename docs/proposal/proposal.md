# MAC F244: Stochastic Calculus and Application to Finance — Project Proposal

**Course:** MAC F244 (Semester I, AY 2026–2027)  
**Institution:** Department of Mathematics, BITS Pilani  

---

## 1. Project Title and a Brief Description of the Problem

**Project Title:** *Development and Evaluation of Algorithmic Trading Strategies using Mean-Reverting Stochastic Processes: Ornstein–Uhlenbeck and Cox–Ingersoll–Ross Models*

### Problem Description
Standard asset pricing models conventionally rely on Geometric Brownian Motion (GBM), which assumes log-normal diffusion with variance that scales linearly and without bound over time ($\text{Var}(\ln S_t) = \sigma^2 t$). While suitable for broad market equity indices over long horizons, this formulation fails for a wide range of financial state variables that exhibit strong economic restorative forces pulling them toward an equilibrium level. Such phenomena are central to:
- **Cointegrated Equity and Commodity Spreads:** Linear combinations of non-stationary asset prices that share a common stochastic trend.
- **Sovereign Interest Rates:** Yield curve levels bounded by monetary policy targets, macroeconomic fundamentals, and central bank reaction functions.
- **Implied Volatility Indices:** Market-implied volatility surfaces that spike during crises but invariably revert to long-term economic baseline levels.

Applying heuristic, discrete trading rules (such as uncalibrated Bollinger Bands) ignores continuous-time transition dynamics, mean-reversion half-lives, heteroskedastic volatility scaling, and estimation bias. Conversely, formulating algorithmic trading systems through continuous-time mean-reverting Stochastic Differential Equations (SDEs) introduces three fundamental mathematical and computational challenges:
1. **Discretization Regularity near Boundaries:** Standard Euler–Maruyama schemes often fail or generate negative values when applied to square-root diffusion processes near zero boundaries where the diffusion coefficient is not globally Lipschitz.
2. **Consistent and Stable Parameter Calibration:** Continuous-time drift and volatility parameters must be inferred accurately from discrete market data via Maximum Likelihood Estimation (MLE) without suffering from numerical overflow in special function evaluations (e.g., modified Bessel functions).
3. **Empirical Strategy Robustness:** Out-of-sample trading alpha calibrated from continuous SDEs must be demonstrated to survive realistic market frictions, including execution delays, bid-ask slippage, and short-borrow fees.

---

## 2. Mathematical Model or Stochastic Process to be Investigated

### 2.1 Ornstein–Uhlenbeck (OU) Process
The Ornstein–Uhlenbeck process models real-valued, mean-reverting spreads $X_t \in \mathbb{R}$:
$$dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t, \qquad \theta > 0,\; \mu \in \mathbb{R},\; \sigma > 0,$$
where $\theta$ represents the annualized rate of mean reversion, $\mu$ is the long-term equilibrium mean, $\sigma$ is the diffusion volatility parameter, and $W_t$ is a standard one-dimensional Brownian motion on a filtered probability space $(\Omega, \mathcal{F}, \{\mathcal{F}_t\}_{t \ge 0}, \mathbb{P})$.

Applying Itô's Lemma to the integrating factor transformation $Y_t = e^{\theta t}(X_t - \mu)$ yields:
$$dY_t = \theta e^{\theta t}(X_t - \mu)\,dt + e^{\theta t}dX_t = \sigma e^{\theta t}\,dW_t.$$
Integrating from $t$ to $t + \Delta t$ gives the exact continuous-time solution:
$$X_{t+\Delta t} = \mu + (X_t - \mu)e^{-\theta \Delta t} + \sigma \int_t^{t+\Delta t} e^{-\theta(t+\Delta t - s)}\,dW_s.$$
By Itô's Isometry, the conditional distribution of $X_{t+\Delta t}$ given $X_t$ is Gaussian:
$$X_{t+\Delta t} \mid X_t \sim \mathcal{N}\!\left(\mu + (X_t - \mu)e^{-\theta \Delta t},\; \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta \Delta t}\right)\right).$$
As $\Delta t \to \infty$, the process converges to its stationary distribution $\mathcal{N}(\mu, \sigma_{\text{eq}}^2)$ with equilibrium variance $\sigma_{\text{eq}}^2 = \sigma^2 / (2\theta)$ and mean-reversion half-life $t_{1/2} = \ln(2)/\theta$.

### 2.2 Cox–Ingersoll–Ross (CIR) Process
To model strictly non-negative financial variables such as interest rates and volatility $r_t \in \mathbb{R}_+$, we examine the CIR square-root diffusion process:
$$dr_t = \kappa(\theta - r_t)\,dt + \xi\sqrt{r_t}\,dW_t, \qquad \kappa > 0,\; \theta > 0,\; \xi > 0,$$
where $\kappa$ is the speed of adjustment, $\theta$ is the long-term mean rate, and $\xi$ is the volatility parameter. The state-dependent diffusion coefficient $\xi\sqrt{r_t}$ introduces empirical heteroskedasticity and collapses to zero as $r_t \to 0$.

The boundary at zero is non-attainable (yielding strictly positive paths $\mathbb{P}(r_t > 0 \; \forall t \ge 0) = 1$) if and only if the **Feller condition** is satisfied:
$$2\kappa\theta > \xi^2.$$
The exact transition density of the CIR process follows a scaled non-central chi-squared distribution:
$$2c\,r_{t+\Delta t} \mid r_t \sim \chi'^2_d(\lambda_t), \qquad c = \frac{2\kappa}{\xi^2(1 - e^{-\kappa\Delta t})}, \quad d = \frac{4\kappa\theta}{\xi^2}, \quad \lambda_t = 2 c r_t e^{-\kappa\Delta t},$$
where $d$ represents the degrees of freedom, $\lambda_t$ is the non-centrality parameter, and $c$ is the scaling constant. The process converges to an ergodic stationary Gamma distribution $\text{Gamma}\bigl(\alpha = \frac{2\kappa\theta}{\xi^2}, \beta = \frac{2\kappa}{\xi^2}\bigr)$.

---

## 3. Proposed Computational Methodology

1. **Numerical SDE Simulation and Discretization Analysis:**
   - Simulate exact OU paths using the Gaussian transition density alongside Euler–Maruyama discretization, quantifying finite-step stationary variance bias $V_{\text{EM}} = \frac{\sigma^2}{2\theta - \theta^2\Delta t} > \sigma_{\text{eq}}^2$ and evaluating strong convergence rates.
   - Simulate exact CIR paths using non-central chi-squared sampling alongside boundary-preserving schemes (Full Truncation, Reflection, Absorption, and Truncated Milstein), testing weak distribution convergence under both Feller-satisfied ($2\kappa\theta > \xi^2$) and boundary-violating ($2\kappa\theta \le \xi^2$) conditions.
2. **Cointegration Pre-Testing and Parameter Estimation:**
   - Ingest historical daily market data via Yahoo Finance (`yfinance`) and perform Engle–Granger two-step cointegration and Augmented Dickey–Fuller (ADF) unit-root stationarity tests under strict causal walk-forward splits.
   - Derive closed-form Maximum Likelihood Estimators (MLE) for the OU process and compare them with discrete AR(1) Ordinary Least Squares (OLS) regression to evaluate discretization bias.
   - Implement exact CIR MLE using Weighted Conditional Least Squares (CLS) initialization and exponentially scaled modified Bessel functions $I_q^e(z) = e^{-z}I_q(z)$ to eliminate double-precision numerical overflow.
3. **Algorithmic Trading, Multi-Strategy Allocation, and Backtesting:**
   - **OU Statistical Arbitrage Strategy:** Construct standardized rolling Z-scores $Z_t = (X_t - \hat{\mu}_t)/\hat{\sigma}_{\text{eq},t}$ on cointegrated ETF pairs with trade-coherent hedge locking, hysteresis entry/exit thresholds, and stop-loss rules.
   - **CIR Yield-Curve Regime Strategy:** Model benchmark 10-Year Treasury yields to construct state-adjusted scores $Z_t^{\text{CIR}} = (r_t - \hat{\theta}_t)/\sqrt{\frac{\hat{\xi}_t^2}{2\hat{\kappa}_t} r_t}$, dynamically allocating duration exposure across Treasury ETFs.
   - **Convex Portfolio Optimization (`cvxpy`):** Combine equity-neutral OU pairs and fixed-income CIR strategies using a quadratic mean-variance portfolio program with turnover constraints.
   - **Friction Sensitivity Analysis:** Evaluate net Sharpe, Sortino, and Maximum Drawdown across transaction costs (0–50 bps), short-borrowing fees (25 bps/year), and execution lags ($T+0$ to $T+2$).

---

## 4. Expected Outcomes

1. **Discretization and Calibration Fidelity:** Demonstration that Exact $\chi'^2$ sampling and Full-Truncation Euler preserve the theoretical probability distribution of the CIR process near zero boundaries, whereas naive schemes introduce substantial boundary distortion. Verification that naive Euler AR(1) regression systematically underestimates the true mean-reversion speed $\theta$ relative to exact Gaussian MLE.
2. **Empirical Strategy Viability and Cointegration Sensitivity:** Demonstration that the continuous-time OU pairs strategy produces statistically significant positive risk-adjusted returns net of transaction costs on cointegrated pairs, while showing performance breakdown on non-cointegrated pairs, establishing cointegration as a strict prerequisite.
3. **Fixed-Income Drawdown Protection and Diversification:** Verification that the heteroskedastic CIR regime strategy dynamically de-risks duration during yield spikes (such as the 2022 Fed tightening cycle), limiting portfolio drawdowns relative to passive bond benchmarks, and that blending uncorrelated OU and CIR returns via convex optimization enhances portfolio stability.

---

## 5. Names and ID Numbers of All Group Members

| S. No. | Student Name | BITS ID Number |
| :---: | :--- | :--- |
| 1 | Kavin Kashyap Albal | `2024ADPS0832G` |
| 2 | Sanjan Ganesan | `2024ADPS0833G` |
| 3 | Atharva Sandeep Agarwala | `2024ADPS0856G` |
| 4 | Shaurya Manjkhola | `2024ADPS0880G` |
