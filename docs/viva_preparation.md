# MAC F244 Presentation & Viva Preparation Guide
**Project:** *Development and Evaluation of Algorithmic Trading Strategies using Mean-Reverting Stochastic Processes: Ornstein–Uhlenbeck and Cox–Ingersoll–Ross Models*  
**Course:** MAC F244 (Stochastic Calculus and Application to Finance) — BITS Pilani  
**Format:** 15–20 Minute Group Viva & Mathematical Defense (5 Marks)

---

## Part I: Recommended 15–18 Minute Presentation Flow (Group of 4)

| Time | Speaker | Section | Key Talking Points |
| :--- | :--- | :--- | :--- |
| **0:00–4:00** | **Speaker 1** | **Motivation & Continuous-Time OU Theory** | • Why GBM fails for relative-value spreads and rates ($\text{Var}(\ln S_t) = \sigma^2 t$ grows unbounded).<br>• OU SDE ($dX_t = \theta(\mu - X_t)dt + \sigma dW_t$).<br>• Analytical derivation of strong solution using integrating factor $Y_t = e^{\theta t}(X_t - \mu)$ and Itô's Lemma.<br>• Stationary law $\mathcal{N}(\mu, \sigma^2/2\theta)$ and half-life $t_{1/2} = \ln(2)/\theta$. |
| **4:00–8:00** | **Speaker 2** | **CIR Square-Root Diffusion & Feller Condition** | • The fatal flaw of Vasicek/OU for rates: Gaussian noise produces negative rates ($\mathbb{P}(r_t < 0) > 0$).<br>• CIR SDE ($dr_t = \kappa(\theta - r_t)dt + \xi\sqrt{r_t}dW_t$) with state-dependent diffusion.<br>• **Feller Condition** ($2\kappa\theta > \xi^2$): Mathematical proof via Feller scale function ensuring strict positivity ($\mathbb{P}(r_t > 0) = 1$).<br>• Numerical discretization: Euler-Maruyama boundary fixes (Full Truncation vs. naive Euler). |
| **8:00–12:00** | **Speaker 3** | **Calibration: AR(1) OLS & Weighted Least Squares** | • **OU Calibration:** Exact AR(1) OLS mapping ($\phi = e^{-\theta\Delta t}$) vs. Gaussian MLE.<br>• Downward bias of naive Euler regression ($\hat{\theta}_{\text{EM}} = (1-\hat{\phi})/\Delta t$).<br>• **CIR Calibration:** Euler Weighted Least Squares (WLS). Dividing by $\sqrt{r_{t-1}}$ homoskedasticizes the error variance, turning CIR calibration into standard 2-variable linear regression.<br>• Monte Carlo $O(N^{-1/2})$ consistency verification. |
| **12:00–16:00** | **Speaker 4** | **Empirical Strategies & Benchmark Comparison** | • **Pillar 1 (Equities):** Cointegration of Alphabet dual-class shares (`GOOGL`/`GOOG`, $p = 0.0082$, $\beta = 0.988$). Market-neutral OU pairs strategy delivers **94.4% win rate**, Sharpe **0.83**, and max drawdown of only **-1.36%** vs. **-22.14%** for Buy & Hold `GOOGL`.<br>• **Pillar 2 (Fixed Income):** CIR dynamic regime on 10Y yields (`^TNX`) improves Sharpe ($0.25$ vs $0.17$) and cuts drawdown ($-9.01\%$ vs $-10.16\%$) over passive bond holding (`IEF`).<br>• Friction sensitivity (slippage & execution lag) and absence of overfitting (In-Sample vs. Out-of-Sample). |

---

## Part II: High-Yield Viva Questions & Answers

### Q1. Derive the exact solution, conditional mean, and conditional variance of the Ornstein–Uhlenbeck (OU) process using Itô's Lemma.

**Answer:**
Consider the OU SDE on a filtered probability space $(\Omega, \mathcal{F}, \{\mathcal{F}_t\}_{t \ge 0}, \mathbb{P})$:
$$dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t, \qquad \theta > 0,\; \sigma > 0.$$

**Step 1: Integrating Factor Transformation.**  
Define the centered process $\tilde{X}_t = X_t - \mu$. Then $d\tilde{X}_t = -\theta \tilde{X}_t\,dt + \sigma\,dW_t$.  
Define the transformation $f(t, x) = e^{\theta t}x$ and set $Y_t = f(t, \tilde{X}_t) = e^{\theta t}(X_t - \mu)$.  
The partial derivatives are:
$$\frac{\partial f}{\partial t} = \theta e^{\theta t}x, \qquad \frac{\partial f}{\partial x} = e^{\theta t}, \qquad \frac{\partial^2 f}{\partial x^2} = 0.$$

**Step 2: Apply Itô's Formula.**  
By Itô's Lemma:
$$dY_t = \frac{\partial f}{\partial t}\,dt + \frac{\partial f}{\partial x}\,d\tilde{X}_t + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}\,d\langle \tilde{X}\rangle_t.$$
Since $\frac{\partial^2 f}{\partial x^2} = 0$, the second-order correction term is zero:
$$dY_t = \theta e^{\theta t}\tilde{X}_t\,dt + e^{\theta t}\bigl(-\theta \tilde{X}_t\,dt + \sigma\,dW_t\bigr) = \sigma e^{\theta t}\,dW_t.$$

**Step 3: Integrate Over $[t, t+\Delta t]$.**  
$$e^{\theta(t+\Delta t)}(X_{t+\Delta t} - \mu) - e^{\theta t}(X_t - \mu) = \sigma \int_t^{t+\Delta t} e^{\theta s}\,dW_s.$$
Multiplying both sides by $e^{-\theta(t+\Delta t)}$ yields the exact strong solution:
$$X_{t+\Delta t} = \mu + (X_t - \mu)e^{-\theta \Delta t} + \sigma \int_t^{t+\Delta t} e^{-\theta(t+\Delta t - s)}\,dW_s.$$

**Step 4: Conditional Mean & Variance.**  
Since the integrand $g(s) = \sigma e^{-\theta(t+\Delta t - s)}$ is deterministic, the Itô integral has expectation zero:
$$\mathbb{E}[X_{t+\Delta t} \mid X_t] = \mu + (X_t - \mu)e^{-\theta \Delta t}.$$
By Itô's Isometry:
$$\text{Var}(X_{t+\Delta t} \mid X_t) = \int_t^{t+\Delta t} \sigma^2 e^{-2\theta(t+\Delta t - s)}\,ds = \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta\Delta t}\right).$$
As $\Delta t \to \infty$, the stationary distribution is Gaussian $\mathcal{N}\!\left(\mu, \frac{\sigma^2}{2\theta}\right)$, and the mean-reversion half-life is $t_{1/2} = \frac{\ln 2}{\theta}$.

---

### Q2. Why is the CIR model preferred over the Vasicek (OU) model for interest rates? What is the Feller condition?

**Answer:**
1. **The Flaw of Vasicek (OU) for Interest Rates:**  
   In 1977, Vasicek modeled interest rates with the OU process: $dr_t = \kappa(\theta - r_t)dt + \sigma dW_t$. Because Gaussian increments have unbounded support $(-\infty, \infty)$, the probability of nominal interest rates becoming negative is strictly positive:
   $$\mathbb{P}(r_{t+\Delta t} < 0 \mid r_t) = \Phi\!\left(-\frac{\mu + (r_t - \mu)e^{-\kappa\Delta t}}{\sqrt{\frac{\sigma^2}{2\kappa}(1 - e^{-2\kappa\Delta t})}}\right) > 0.$$
   When baseline interest rates are low (e.g. 1% to 2%), Vasicek routinely generates negative rates, which is economically unrealistic for standard nominal yields.

2. **The CIR Solution:**  
   In 1985, Cox, Ingersoll, and Ross introduced the square-root diffusion:
   $$dr_t = \kappa(\theta - r_t)\,dt + \xi\sqrt{r_t}\,dW_t.$$
   As $r_t \to 0$, the stochastic noise $\xi\sqrt{r_t} \to 0$, shutting off random fluctuations, while the drift $\kappa\theta > 0$ pushes the rate upward.

3. **The Feller Condition ($2\kappa\theta > \xi^2$):**  
   - **Scale function test:** Feller's scale function $s(x) = \int^x y^{-\frac{2\kappa\theta}{\xi^2}} e^{\frac{2\kappa}{\xi^2}y}dy$.  
   - The zero boundary is non-attainable ($\mathbb{P}(r_t > 0 \; \forall t \ge 0) = 1$) if and only if $\int_0^\varepsilon s(x)dx = \infty$, which holds when $2\kappa\theta \ge \xi^2$.
   - If $2\kappa\theta < \xi^2$, the process can touch zero in finite time, but zero acts as an instantaneously reflecting boundary ($r_t \ge 0$).

---

### Q3. How do you calibrate the OU model using AR(1) OLS, and why does naive Euler regression introduce bias?

**Answer:**
1. **Exact AR(1) Mapping:**  
   The exact conditional law $X_i \mid X_{i-1} \sim \mathcal{N}(\mu + (X_{i-1}-\mu)e^{-\theta\Delta t}, \sigma_\varepsilon^2)$ maps directly to a discrete AR(1) regression:
   $$X_i = \alpha + \phi X_{i-1} + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_\varepsilon^2).$$
   By matching terms:
   $$\phi = e^{-\theta\Delta t} \implies \hat{\theta} = -\frac{\ln\hat{\phi}}{\Delta t}, \qquad \alpha = \mu(1 - \phi) \implies \hat{\mu} = \frac{\hat{\alpha}}{1 - \hat{\phi}}.$$
   $$\sigma_\varepsilon^2 = \frac{\sigma^2}{2\theta}(1 - \phi^2) \implies \hat{\sigma} = s_\varepsilon\sqrt{\frac{2\hat{\theta}}{1 - \hat{\phi}^2}}.$$
   Because Gaussian AR(1) OLS and conditional Maximum Likelihood Estimation (MLE) share the exact same sufficient statistics, their parameter estimates $(\hat{\theta}, \hat{\mu})$ are algebraically identical.

2. **Downward Bias of Naive Euler Regression:**  
   If one instead discretizes naively via Euler: $\Delta X_i = \theta(\mu - X_{i-1})\Delta t + \sigma\sqrt{\Delta t}\varepsilon_i = a + b X_{i-1}$, the slope $b = -\theta\Delta t = \phi - 1$.  
   This implicitly assumes $\hat{\theta}_{\text{Euler}} = \frac{1 - \hat{\phi}}{\Delta t}$.  
   Since $1 - e^{-x} < x$ for all $x > 0$, approximating $-\ln(\phi)$ with $(1 - \phi)$ causes naive Euler regression to **systematically underestimate** the true mean-reversion speed $\theta$.

---

### Q4. How is the CIR model calibrated using Weighted Least Squares (WLS), and why is it preferred over exact Bessel function MLE in practice?

**Answer:**
1. **Weighted Least Squares (WLS) Formulation:**  
   Discretize the CIR SDE over $\Delta t$:
   $$r_i - r_{i-1} \approx \kappa(\theta - r_{i-1})\Delta t + \xi\sqrt{r_{i-1}}\sqrt{\Delta t}\,\eta_i, \quad \eta_i \sim \mathcal{N}(0, 1).$$
   Notice that the error term has variance $\text{Var}(\Delta r_i \mid r_{i-1}) = \xi^2 r_{i-1}\Delta t$ (heteroskedasticity).  
   Dividing both sides by $\sqrt{r_{i-1}}$ standardizes the variance (homoskedasticization):
   $$\frac{r_i - r_{i-1}}{\sqrt{r_{i-1}}} = \gamma_1\left(\frac{1}{\sqrt{r_{i-1}}}\right) + \gamma_2\sqrt{r_{i-1}} + \nu_i, \qquad \text{Var}(\nu_i) \approx \xi^2\Delta t.$$
   This is a standard linear regression $Y = \gamma_1 Z_1 + \gamma_2 Z_2 + \text{residual}$.
   From the fitted coefficients $\hat{\gamma}_1$ and $\hat{\gamma}_2$:
   $$\hat{\kappa} = -\frac{\hat{\gamma}_2}{\Delta t}, \qquad \hat{\theta} = -\frac{\hat{\gamma}_1}{\hat{\gamma}_2}, \qquad \hat{\xi} = \frac{s_\nu}{\sqrt{\Delta t}}.$$

2. **Why WLS is superior to Bessel MLE for Undergraduate Work:**  
   - The exact continuous transition density of CIR is a non-central chi-squared distribution whose PDF contains modified Bessel functions of the first kind $I_q(2\sqrt{u_i v_i})$.
   - At daily frequency ($\Delta t = 1/252$), the argument $2\sqrt{u_i v_i} > 1000$, causing $I_q(z) \sim e^z/\sqrt{2\pi z}$ to instantly overflow IEEE double precision (`inf`), requiring exponentially scaled Bessel tricks ($I_q^e(z) = e^{-z}I_q(z)$) and numerical gradient line-search solvers that frequently get trapped in local minima.
   - WLS is closed-form, numerically stable, has no convergence failures, and can be derived on a blackboard in 2 minutes.

---

### Q5. Why did you select Alphabet dual-class shares (`GOOGL` and `GOOG`) for the OU pairs strategy?

**Answer:**
1. **Economic Parity (The Core Requirement of Cointegration):**  
   Pairs trading fails if the two stocks have diverging fundamentals (as shown by non-cointegrated pairs where the spread drifts permanently, causing catastrophic stop-losses).  
   `GOOGL` (Class A) and `GOOG` (Class C) are dual-class shares of the **exact same company (Alphabet Inc.)**:
   - Class A has 1 vote per share; Class C has 0 votes (non-voting).
   - Both share classes have identical rights to cash flows, earnings per share, dividend declarations, and corporate liquidation proceeds.
2. **Natural Mean-Reverting Spread:**  
   Because the economic underlying is identical, their price ratio cannot diverge indefinitely. Any temporary price dislocation due to supply/demand order flow is an arbitrage opportunity that naturally reverts to the equilibrium voting premium.
3. **Statistical Verification:**  
   On our 2019–2022 training set:
   - Hedge ratio $\beta = 0.988 \approx 1.00$.
   - $R^2 = 0.9996$.
   - Augmented Dickey–Fuller (ADF) statistic $t = -3.490$ ($p = 0.0082 < 0.01$). Cointegration is established with $>99\%$ statistical confidence.

---

### Q6. Why did you choose Passive Buy-and-Hold as the benchmark for both models? What did the empirical comparison demonstrate?

**Answer:**
1. **Unified Assessment:**  
   Comparing an active quantitative strategy against passive Buy-and-Hold of the underlying asset answers the essential financial question: *"Does mathematical SDE modeling actually generate superior risk-adjusted performance compared to just holding the asset?"*

2. **Empirical Findings:**
   - **Pillar 1 (Equities: OU Pairs vs. Buy & Hold `GOOGL`):**  
     - Buy & Hold `GOOGL` captured huge equity market beta ($+113.2\%$), but exposed investors to brutal volatility ($29.19\%$) and a **$-22.14\%$ maximum drawdown**.
     - OU Pairs Trading is **market-neutral** (long `GOOGL`, short `GOOG`). Market rallies and crashes cancel out! It achieved an annualized volatility of only **$1.49\%$**, a **$94.4\%$ win rate**, and an extraordinary maximum drawdown of only **$-1.36\%$**, delivering a high Sharpe ratio of $0.83$ and Sortino of $1.21$ net of all transaction costs.
   - **Pillar 2 (Fixed Income: CIR Dynamic vs. Buy & Hold `IEF`):**  
     - Sovereign bonds suffered heavy turbulence during the 2023–2024 high-interest-rate environment.
     - The CIR dynamic regime strategy detected when yields were overshooting equilibrium and tactically shifted duration into short-term cash/bonds (`SHY`), cutting volatility from $8.14\%$ to $5.52\%$, reducing max drawdown from $-10.16\%$ to $-9.01\%$, and improving the Sharpe ratio from $0.17$ to $0.25$.

---

### Q7. How do transaction costs and execution lag affect the viability of the OU pairs strategy?

**Answer:**
1. **High Turnover Friction:**  
   Pairs trading executes two simultaneous legs (long one asset, short the other), meaning each round-trip trade incurs 4 executions. At $1.5\times$ leverage, portfolio turnover is significant.
2. **Transaction Cost Sensitivity:**  
   Our sensitivity analysis demonstrates that for mega-cap dual-class equities like Alphabet (where institutional bid-ask spreads are a fraction of a cent), a realistic cost of **5 bps one-way** leaves a healthy Sharpe ratio of **$0.83$**. The strategy remains profitable up to $15$ bps per leg before transaction drag absorbs the edge.
3. **Execution Lag ($T+1$):**  
   We enforce a causal $T+1$ execution protocol: signals generated at the close of day $t$ are filled at day $t+1$. This prevents look-ahead bias and simulates real-world algorithmic execution delay. Even with a $+1$ additional day of execution delay ($T+2$), the strategy retains positive alpha.
