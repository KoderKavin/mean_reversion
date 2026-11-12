#!/usr/bin/env python3
"""
Simulation, calibration, and backtesting experiments for OU and CIR processes.
MAC F244 Course Project: Algorithmic Trading with Mean-Reverting Processes.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tabulate import tabulate

from src import (
    CIRParams,
    CIRTradingConfig,
    OUParams,
    OUTradingConfig,
    calibrate_cir_cls,
    calibrate_cir_wls,
    calibrate_ou_mle,
    calibrate_ou_ols,
    cir_stationary_pdf,
    evaluate_backtest,
    fetch_market_data,
    generate_ou_signals,
    ou_analytical_moments,
    ou_log_likelihood,
    ou_stationary_pdf,
    run_cir_yield_strategy,
    run_ou_pairs_strategy,
    simulate_cir_euler,
    simulate_cir_exact,
    simulate_ou_euler,
    simulate_ou_exact,
    test_pair_cointegration,
)

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent
FIG_DIR = ROOT_DIR / "figures"
DATA_DIR = ROOT_DIR / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Plotting settings
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "crimson": "#B22222",
    "amber": "#D97706",
    "purple": "#5B2C6F",
    "slate": "#5D6D7E",
    "green": "#1E8449",
}


# =============================================================================
# 1. SDE Simulation and Discretization
# =============================================================================
def run_simulation_and_discretization_study() -> Dict[str, Any]:
    print("\n[1/4] Running SDE Simulation and Discretization Study...")

    # Set OU process parameters: dX_t = theta*(mu - X_t)dt + sigma*dW_t
    ou_p = OUParams(theta=6.0, mu=0.0, sigma=0.28)
    t_ou, paths_ou = simulate_ou_exact(ou_p, x0=0.32, T=2.0, dt=1.0 / 252.0, n_paths=6, seed=42)
    moments_ou = ou_analytical_moments(ou_p, x0=0.32, t=t_ou)

    # Set CIR process parameters: dr_t = kappa*(theta - r_t)dt + xi*sqrt(r_t)*dW_t
    # Regime 1: 2*kappa*theta > xi^2 (Feller condition holds, r_t > 0)
    cir_sat = CIRParams(kappa=3.5, theta=0.045, xi=0.15)
    # Regime 2: 2*kappa*theta < xi^2 (Feller condition violated, boundary touch)
    cir_vio = CIRParams(kappa=1.0, theta=0.030, xi=0.42)

    # Simulate CIR exact paths via non-central chi-squared transition law
    t_cir, paths_cir_sat = simulate_cir_exact(cir_sat, r0=0.075, T=2.0, dt=1.0 / 252.0, n_paths=4, seed=12)
    _, paths_cir_vio = simulate_cir_exact(cir_vio, r0=0.045, T=2.0, dt=1.0 / 252.0, n_paths=4, seed=24)

    # Test Euler-Maruyama discretization convergence across step sizes dt
    dt_list = np.array([1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4], dtype=np.float64)
    ou_strong_err = []
    ou_var_bias = []
    cir_w1_ft = []
    cir_w1_ref = []
    cir_w1_abs = []

    n_conv_paths = 6000
    T_conv = 1.0
    _, ref_exact_cir = simulate_cir_exact(cir_vio, r0=0.035, T=T_conv, dt=1 / 256, n_paths=n_conv_paths, seed=777)
    exact_cir_T = ref_exact_cir[:, -1]

    for idx_dt, dt_val in enumerate(dt_list):
        n_steps = int(round(T_conv / dt_val))
        rng = np.random.default_rng(500 + idx_dt)
        Z = rng.standard_normal((n_conv_paths, n_steps))

        # Compare exact OU simulation with Euler-Maruyama discretization
        _, ou_ex = simulate_ou_exact(ou_p, x0=0.25, T=T_conv, dt=dt_val, n_paths=n_conv_paths, normals=Z)
        _, ou_em = simulate_ou_euler(ou_p, x0=0.25, T=T_conv, dt=dt_val, n_paths=n_conv_paths, normals=Z)
        rmse_strong = float(np.sqrt(np.mean((ou_ex[:, -1] - ou_em[:, -1]) ** 2)))
        ou_strong_err.append(rmse_strong)
        ou_var_bias.append(abs(float(np.var(ou_em[:, -1])) - float(np.var(ou_ex[:, -1]))))

        # Test CIR Euler schemes: full truncation, reflection, and absorption
        _, c_ft = simulate_cir_euler(cir_vio, r0=0.035, T=T_conv, dt=dt_val, n_paths=n_conv_paths, scheme="full_truncation", normals=Z)
        _, c_rf = simulate_cir_euler(cir_vio, r0=0.035, T=T_conv, dt=dt_val, n_paths=n_conv_paths, scheme="reflection", normals=Z)
        _, c_ab = simulate_cir_euler(cir_vio, r0=0.035, T=T_conv, dt=dt_val, n_paths=n_conv_paths, scheme="absorption", normals=Z)

        # Compute Wasserstein-1 distance to exact distribution
        cir_w1_ft.append(float(stats.wasserstein_distance(exact_cir_T, c_ft[:, -1])))
        cir_w1_ref.append(float(stats.wasserstein_distance(exact_cir_T, c_rf[:, -1])))
        cir_w1_abs.append(float(stats.wasserstein_distance(exact_cir_T, c_ab[:, -1])))

    # Strong convergence rate slope
    ou_slope = float(np.polyfit(np.log(dt_list[:5]), np.log(ou_strong_err[:5]), 1)[0])

    # Plot Figure 1: Simulation and Discretization
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))

    # (a) OU sample paths and theoretical mean decay
    ax = axes[0, 0]
    for i in range(paths_ou.shape[0]):
        ax.plot(t_ou, paths_ou[i], lw=0.95, alpha=0.75)
    ax.plot(t_ou, moments_ou["mean"], color="black", lw=2.2, label=r"Conditional Mean $\mathbb{E}[X_t \mid X_0]$")
    ax.fill_between(
        t_ou,
        moments_ou["mean"] - 2.0 * moments_ou["std"],
        moments_ou["mean"] + 2.0 * moments_ou["std"],
        color="gray", alpha=0.18, label=r"$\pm 2\sigma$ Interval",
    )
    ax.axhline(ou_p.mu, color=PALETTE["crimson"], ls="--", lw=1.3, label=r"Mean $\mu=0$")
    ax.set_title("(a) Simulated OU Paths and Mean Decay")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Spread Level")
    ax.legend(loc="upper right", frameon=True)

    # (b) CIR sample paths under Feller-satisfied and violated regimes
    ax = axes[0, 1]
    for i in range(paths_cir_sat.shape[0]):
        ax.plot(t_cir, paths_cir_sat[i], color=PALETTE["teal"], lw=1.1, alpha=0.85, label="Feller Satisfied" if i == 0 else None)
    for i in range(paths_cir_vio.shape[0]):
        ax.plot(t_cir, paths_cir_vio[i], color=PALETTE["crimson"], lw=1.0, ls="--", alpha=0.75, label="Feller Violated" if i == 0 else None)
    ax.axhline(0.0, color="black", lw=1.1, ls=":")
    ax.set_title("(b) Simulated CIR Paths (Feller Regimes)")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Interest Rate")
    ax.legend(loc="upper right", frameon=True)

    # (c) Rate paths comparison: CIR vs Vasicek
    ax = axes[1, 0]
    vas_p = OUParams(theta=3.5, mu=0.045, sigma=0.04)
    _, vas_paths = simulate_ou_exact(vas_p, x0=0.015, T=2.0, dt=1.0 / 252.0, n_paths=5, seed=88)
    _, cir_low_paths = simulate_cir_exact(cir_sat, r0=0.015, T=2.0, dt=1.0 / 252.0, n_paths=5, seed=88)
    for i in range(vas_paths.shape[0]):
        ax.plot(t_ou, vas_paths[i], color=PALETTE["crimson"], lw=0.95, ls="--", alpha=0.75, label="Vasicek (OU)" if i == 0 else None)
    for i in range(cir_low_paths.shape[0]):
        ax.plot(t_ou, cir_low_paths[i], color=PALETTE["navy"], lw=1.2, alpha=0.85, label="CIR" if i == 0 else None)
    ax.axhline(0.0, color="black", lw=1.2, ls="--", label="Zero Boundary")
    ax.set_title("(c) Rate Paths: CIR vs. Vasicek")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Interest Rate")
    ax.legend(loc="upper right", frameon=True)

    # (d) Discretization error convergence vs step size
    ax = axes[1, 1]
    ax.loglog(dt_list, ou_strong_err, "o-", color=PALETTE["navy"], lw=1.8, label=f"OU Strong Error (slope={ou_slope:.2f})")
    ax.loglog(dt_list, cir_w1_ft, "s-", color=PALETTE["teal"], lw=1.8, label="CIR Full-Truncation")
    ax.loglog(dt_list, cir_w1_ref, "^--", color=PALETTE["amber"], lw=1.5, label="CIR Reflection")
    ax.loglog(dt_list, cir_w1_abs, "d:", color=PALETTE["crimson"], lw=1.5, label="CIR Absorption")
    ax.set_title(r"(d) Discretization Error vs. Step Size $\Delta t$")
    ax.set_xlabel(r"Step Size $\Delta t$ (Years, log scale)")
    ax.set_ylabel("Error Metric (log scale)")
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_sde_simulations.png")
    plt.close(fig)

    return {
        "ou_strong_slope": ou_slope,
        "dt_grid": dt_list.tolist(),
        "ou_strong_err": ou_strong_err,
        "ou_var_bias": ou_var_bias,
        "cir_w1_full_truncation": cir_w1_ft,
        "cir_w1_reflection": cir_w1_ref,
        "cir_w1_absorption": cir_w1_abs,
    }


# =============================================================================
# 2. Model Calibration and Diagnostics
# =============================================================================
def run_calibration_study(prices_df: pd.DataFrame, train_idx: int) -> Dict[str, Any]:
    print("\n[2/4] Running Model Calibration and Diagnostics...")

    dt = 1.0 / 252.0
    ou_true = OUParams(theta=12.0, mu=0.0, sigma=0.30)
    cir_true = CIRParams(kappa=4.0, theta=0.04, xi=0.14)

    # Monte Carlo simulation to evaluate parameter estimation error across sample sizes
    sample_sizes = [250, 500, 1000, 2000, 4000]
    n_rep = 200
    ou_mle_rmse_theta = []
    ou_euler_rmse_theta = []
    cir_wls_rmse_theta = []
    cir_wls_rmse_xi = []

    for n_obs in sample_sizes:
        T_sim = n_obs * dt
        _, ou_mc = simulate_ou_exact(ou_true, x0=0.0, T=T_sim, dt=dt, n_paths=n_rep, seed=1000 + n_obs)
        _, cir_mc = simulate_cir_exact(cir_true, r0=0.04, T=T_sim, dt=dt, n_paths=n_rep, seed=2000 + n_obs)

        th_mle_est = []
        th_eul_est = []
        cir_th_est = []
        cir_xi_est = []
        for m in range(n_rep):
            c_m = calibrate_ou_mle(ou_mc[m], dt=dt, compute_se=False)
            c_o = calibrate_ou_ols(ou_mc[m], dt=dt)
            th_mle_est.append(c_m.params.theta)
            th_eul_est.append(c_o.euler_theta)

            c_cir = calibrate_cir_cls(cir_mc[m], dt=dt)
            cir_th_est.append(c_cir.theta)
            cir_xi_est.append(c_cir.xi)

        ou_mle_rmse_theta.append(float(np.sqrt(np.mean((np.array(th_mle_est) - ou_true.theta) ** 2)) / ou_true.theta))
        ou_euler_rmse_theta.append(float(np.sqrt(np.mean((np.array(th_eul_est) - ou_true.theta) ** 2)) / ou_true.theta))
        cir_wls_rmse_theta.append(float(np.sqrt(np.mean((np.array(cir_th_est) - cir_true.theta) ** 2)) / cir_true.theta))
        cir_wls_rmse_xi.append(float(np.sqrt(np.mean((np.array(cir_xi_est) - cir_true.xi) ** 2)) / cir_true.xi))

    # Calibrate OU parameters on GOOGL/GOOG spread (training set 2019-2022)
    coint_goog_train = test_pair_cointegration(prices_df["GOOGL"].iloc[:train_idx], prices_df["GOOG"].iloc[:train_idx])
    spread_train = (
        np.log(prices_df["GOOGL"].iloc[:train_idx])
        - (coint_goog_train["alpha"] + coint_goog_train["beta"] * np.log(prices_df["GOOG"].iloc[:train_idx]))
    )
    ou_goog_mle = calibrate_ou_mle(spread_train.to_numpy(), dt=dt, compute_se=True)
    ou_goog_ols = calibrate_ou_ols(spread_train.to_numpy(), dt=dt)

    # Calibrate CIR parameters on 10Y Treasury yields (^TNX) via WLS
    tnx_dec_train = np.maximum(prices_df["^TNX"].iloc[:train_idx].to_numpy() / 100.0, 1e-4)
    cir_tnx_wls = calibrate_cir_wls(tnx_dec_train, dt=dt)

    # Calibration results summary table
    cal_table = [
        ["GOOGL/GOOG (OU)", "Exact MLE", f"{ou_goog_mle.params.theta:.3f}", f"{ou_goog_mle.params.mu:.4f}", f"{ou_goog_mle.params.sigma:.4f}", f"{ou_goog_mle.params.half_life*252:.1f}d", f"ll={ou_goog_mle.log_likelihood:.1f}"],
        ["GOOGL/GOOG (OU)", "AR(1) OLS", f"{ou_goog_ols.params.theta:.3f}", f"{ou_goog_ols.params.mu:.4f}", f"{ou_goog_ols.params.sigma:.4f}", f"{ou_goog_ols.params.half_life*252:.1f}d", f"R2={ou_goog_ols.r_squared:.4f}"],
        ["10Y Yield ^TNX", "Weighted Least Squares", f"{cir_tnx_wls.params.kappa:.3f}", f"{cir_tnx_wls.params.theta:.4f}", f"{cir_tnx_wls.params.xi:.4f}", f"{cir_tnx_wls.params.half_life*252:.1f}d", f"Feller={cir_tnx_wls.feller_ratio:.1f}"],
    ]
    print("\nIn-Sample Model Calibration (2019-2022 Training Period):")
    print(tabulate(cal_table, headers=["Asset / Model", "Method", "theta / kappa", "mu / theta", "sigma / xi", "Half-Life", "Fit / Diagnostic"], tablefmt="github"))

    # Plot Figure 2: Calibration Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))

    # (a) OU log-likelihood surface
    ax = axes[0, 0]
    _, path_ll = simulate_ou_exact(ou_true, x0=0.0, T=1000 * dt, dt=dt, n_paths=1, seed=314)
    sample_1d = path_ll[0]
    fit_1d = calibrate_ou_mle(sample_1d, dt=dt, compute_se=False)
    th_grid = np.linspace(5.0, 20.0, 50)
    sg_grid = np.linspace(0.24, 0.36, 50)
    LL_mat = np.zeros((len(sg_grid), len(th_grid)))
    for i_s, sg_v in enumerate(sg_grid):
        for j_t, th_v in enumerate(th_grid):
            LL_mat[i_s, j_t] = ou_log_likelihood(th_v, fit_1d.params.mu, sg_v, sample_1d, dt)

    cs = ax.contourf(th_grid, sg_grid, LL_mat - np.max(LL_mat), levels=np.linspace(-25, 0, 16), cmap="Blues")
    fig.colorbar(cs, ax=ax, label=r"$\ell(\theta, \sigma) - \ell_{\max}$")
    ax.plot(ou_true.theta, ou_true.sigma, "*", color=PALETTE["amber"], markersize=11, markeredgecolor="black", label=r"True $(\theta, \sigma)$")
    ax.plot(fit_1d.params.theta, fit_1d.params.sigma, "o", color=PALETTE["crimson"], markersize=7, label="Estimated")
    ax.set_title("(a) OU Log-Likelihood Surface")
    ax.set_xlabel(r"Mean-Reversion Speed $\theta$")
    ax.set_ylabel(r"Diffusion Volatility $\sigma$")
    ax.legend(loc="upper right", frameon=True)

    # (b) Parameter estimation error vs sample size
    ax = axes[0, 1]
    ax.loglog(sample_sizes, ou_mle_rmse_theta, "o-", color=PALETTE["navy"], lw=1.8, label=r"OU MLE ($\hat{\theta}$)")
    ax.loglog(sample_sizes, ou_euler_rmse_theta, "s--", color=PALETTE["crimson"], lw=1.6, label=r"OU Euler ($\hat{\theta}_{\mathrm{EM}}$)")
    ax.loglog(sample_sizes, cir_wls_rmse_theta, "^-", color=PALETTE["teal"], lw=1.8, label=r"CIR WLS ($\hat{\theta}$)")
    ax.loglog(sample_sizes, cir_wls_rmse_xi, "d-", color=PALETTE["purple"], lw=1.6, label=r"CIR WLS ($\hat{\xi}$)")
    ref_n = np.array(sample_sizes, dtype=float)
    ax.loglog(ref_n, 0.35 * (ref_n / ref_n[0]) ** (-0.5), ":", color="black", lw=1.2, label=r"$O(N^{-1/2})$ Rate")
    ax.set_title("(b) Estimation Error vs. Sample Size")
    ax.set_xlabel("Sample Size $N$")
    ax.set_ylabel("Relative RMSE")
    ax.legend(loc="upper right", frameon=True)

    # (c) Residual distribution of GOOGL/GOOG spread
    ax = axes[1, 0]
    sp_arr = spread_train.to_numpy()
    phi_emp = ou_goog_mle.phi
    mu_emp = ou_goog_mle.params.mu
    std_innov = (sp_arr[1:] - (mu_emp + (sp_arr[:-1] - mu_emp) * phi_emp)) / ou_goog_mle.sigma_eps
    z_x = np.linspace(-4.0, 4.0, 300)
    ax.hist(std_innov, bins=35, density=True, alpha=0.45, color=PALETTE["navy"], label="Residuals")
    ax.plot(z_x, stats.norm.pdf(z_x), color=PALETTE["crimson"], lw=2.0, label=r"Standard Normal $\mathcal{N}(0, 1)$")
    ax.set_title("(c) GOOGL/GOOG Spread Residuals")
    ax.set_xlabel("Standardized Residual")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=True)

    # (d) Volatility vs yield level (testing CIR square-root scaling)
    ax = axes[1, 1]
    tnx_full = np.maximum(prices_df["^TNX"].to_numpy() / 100.0, 1e-4)
    r_prev = tnx_full[:-1]
    r_diff = np.abs(tnx_full[1:] - r_prev)
    nbins = 12
    bins = np.quantile(r_prev, np.linspace(0.0, 1.0, nbins + 1))
    bin_centers = []
    bin_vol = []
    for b in range(nbins):
        mask = (r_prev >= bins[b]) & (r_prev <= bins[b + 1])
        if np.sum(mask) > 10:
            bin_centers.append(float(np.mean(r_prev[mask])))
            bin_vol.append(float(np.std(r_diff[mask], ddof=1) * np.sqrt(252.0)))
    bin_centers = np.array(bin_centers)
    bin_vol = np.array(bin_vol)
    ax.scatter(bin_centers * 100.0, bin_vol * 100.0, color=PALETTE["navy"], s=45, label="Realized Volatility")
    poly_fit = np.polyfit(np.sqrt(bin_centers), bin_vol, deg=1)
    x_fine = np.linspace(min(bin_centers), max(bin_centers), 100)
    ax.plot(x_fine * 100.0, (poly_fit[0] * np.sqrt(x_fine) + poly_fit[1]) * 100.0, color=PALETTE["teal"], lw=2.0, label=r"Fitted CIR: $\xi\sqrt{r}$")
    ax.axhline(float(np.mean(bin_vol) * 100.0), color=PALETTE["crimson"], ls="--", lw=1.2, label="Constant Volatility (OU)")
    ax.set_title("(d) 10Y Yield Volatility vs. Rate Level")
    ax.set_xlabel("10-Year Treasury Yield (%)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_calibration_diagnostics.png")
    plt.close(fig)

    return {
        "coint_goog_train": coint_goog_train,
        "ou_goog_mle": {
            "theta": ou_goog_mle.params.theta,
            "mu": ou_goog_mle.params.mu,
            "sigma": ou_goog_mle.params.sigma,
            "half_life_days": ou_goog_mle.params.half_life * 252.0,
            "log_likelihood": ou_goog_mle.log_likelihood,
        },
        "cir_tnx_wls": {
            "kappa": cir_tnx_wls.params.kappa,
            "theta": cir_tnx_wls.params.theta,
            "xi": cir_tnx_wls.params.xi,
            "feller_ratio": cir_tnx_wls.feller_ratio,
            "half_life_days": cir_tnx_wls.params.half_life * 252.0,
            "std_errors": cir_tnx_wls.std_errors,
        },
    }


# =============================================================================
# 3. Strategy Backtesting (2023-2024)
# =============================================================================
def run_empirical_backtests(prices_df: pd.DataFrame, eval_idx: int) -> tuple[Dict[str, Any], Any, Any]:
    print("\n[3/4] Running Out-of-Sample Backtests (2023-2024)...")

    # Strategy 1: OU pairs trading on GOOGL / GOOG
    ou_cfg = OUTradingConfig(
        z_entry=1.25, z_exit=0.20, z_stop=3.0, lookback=120,
        recalib_step=5, max_half_life_days=45.0, gross_leverage=1.5,
        tc_bps=5.0, borrow_bps_annual=25.0, execution_lag=1,
    )
    bt_goog_pair, diag_ou = run_ou_pairs_strategy(
        prices_df["GOOGL"], prices_df["GOOG"], ou_cfg, start_eval_idx=eval_idx, name="OU Pairs (GOOGL/GOOG)"
    )

    # Benchmark 1: Buy and hold GOOGL
    googl_rets = pd.DataFrame({"GOOGL": prices_df["GOOGL"].pct_change().fillna(0.0).iloc[eval_idx:]})
    googl_w = pd.DataFrame({"GOOGL": 1.0}, index=googl_rets.index)
    bt_googl_bh = evaluate_backtest(
        "Benchmark: Buy & Hold GOOGL",
        googl_rets, googl_w, pd.Series(1.0, index=googl_rets.index),
        tc_bps=0.0, borrow_bps_annual=0.0, execution_lag=1,
    )

    # Strategy 2: CIR dynamic yield strategy on IEF / SHY
    cir_cfg = CIRTradingConfig(
        z_entry=0.75, z_exit=0.10, z_stop=2.5, vol_stress_mult=1.15,
        lookback=180, recalib_step=10, tc_bps=3.0, borrow_bps_annual=0.0, execution_lag=1,
    )
    bt_cir, diag_cir = run_cir_yield_strategy(
        prices_df["^TNX"], prices_df["IEF"], prices_df["SHY"],
        cir_cfg, start_eval_idx=eval_idx, name="CIR Dynamic (IEF/SHY)"
    )

    # Benchmark 2: Buy and hold IEF
    ief_rets = pd.DataFrame({"IEF": prices_df["IEF"].pct_change().fillna(0.0).iloc[eval_idx:]})
    ief_w = pd.DataFrame({"IEF": 1.0}, index=ief_rets.index)
    bt_ief_bh = evaluate_backtest(
        "Benchmark: Buy & Hold IEF",
        ief_rets, ief_w, pd.Series(1.0, index=ief_rets.index),
        tc_bps=0.0, borrow_bps_annual=0.0, execution_lag=1,
    )

    all_bts = [bt_goog_pair, bt_googl_bh, bt_cir, bt_ief_bh]
    rows = []
    perf_dict = {}
    for bt in all_bts:
        d = {k: float(v) for k, v in bt.metrics.to_dict().items()}
        perf_dict[bt.name] = d
        rows.append([
            bt.name,
            f"{d['Total Return (%)']:.2f}",
            f"{d['CAGR (%)']:.2f}",
            f"{d['Ann. Vol (%)']:.2f}",
            f"{d['Sharpe Ratio']:.2f}",
            f"{d['Sortino Ratio']:.2f}",
            f"{d['Max Drawdown (%)']:.2f}",
            f"{d['Hit Rate (%)']:.1f}",
            f"{d['Profit Factor']:.2f}",
            f"{d['Cost Drag (%)']:.2f}",
        ])

    headers = ["Strategy", "Tot Ret%", "CAGR%", "Vol%", "Sharpe", "Sortino", "MaxDD%", "Win%", "ProfFac", "Cost%"]
    print("\nOut-of-Sample Performance (2023-2024):")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # Plot Figure 3: Backtest Performance
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8))

    # (a) GOOGL/GOOG spread Z-score and trading signals
    ax = axes[0, 0]
    ax.plot(diag_ou.index, diag_ou["z_score"], color=PALETTE["navy"], lw=1.1, label="Z-Score")
    ax.axhline(ou_cfg.z_entry, color=PALETTE["crimson"], ls="--", lw=1.1, label=rf"Entry Threshold $\pm {ou_cfg.z_entry}$")
    ax.axhline(-ou_cfg.z_entry, color=PALETTE["crimson"], ls="--", lw=1.1)
    ax.axhline(ou_cfg.z_exit, color=PALETTE["green"], ls=":", lw=0.9, label=rf"Exit Band $\pm {ou_cfg.z_exit}$")
    ax.axhline(-ou_cfg.z_exit, color=PALETTE["green"], ls=":", lw=0.9)
    long_pts = diag_ou[diag_ou["signal"] > 0]
    short_pts = diag_ou[diag_ou["signal"] < 0]
    ax.scatter(long_pts.index, long_pts["z_score"], marker="^", color=PALETTE["green"], s=28, label="Long GOOGL / Short GOOG", zorder=4)
    ax.scatter(short_pts.index, short_pts["z_score"], marker="v", color=PALETTE["crimson"], s=28, label="Short GOOGL / Long GOOG", zorder=4)
    ax.set_title("(a) GOOGL/GOOG Spread Z-Score and Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-Score")
    ax.legend(loc="upper right", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # (b) Cumulative returns of OU pairs strategy vs benchmark
    ax = axes[0, 1]
    norm_goog_bh = (1.0 + bt_googl_bh.net_returns).cumprod()
    norm_goog_pair = (1.0 + bt_goog_pair.net_returns).cumprod()
    ax.plot(norm_goog_pair.index, norm_goog_pair, color=PALETTE["navy"], lw=2.2, label=f"OU Pairs (Sharpe {bt_goog_pair.metrics.sharpe_ratio:.2f})")
    ax.plot(norm_goog_bh.index, norm_goog_bh, color=PALETTE["crimson"], lw=1.5, ls="--", label=f"Buy & Hold GOOGL (Sharpe {bt_googl_bh.metrics.sharpe_ratio:.2f})")
    ax.set_title("(b) Cumulative Returns: OU Pairs vs. Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth ($)")
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # (c) 10-Year Treasury yield vs CIR calibrated equilibrium mean
    ax = axes[1, 0]
    ax.plot(diag_cir.index, diag_cir["rate_decimal"] * 100.0, color=PALETTE["amber"], lw=1.4, label="10Y Treasury Yield (^TNX)")
    ax.plot(diag_cir.index, diag_cir["theta"] * 100.0, color=PALETTE["navy"], lw=1.6, ls="--", label="CIR Equilibrium Mean")
    ax.set_title("(c) 10-Year Yield and CIR Equilibrium Level")
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.legend(loc="upper right", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # (d) Cumulative returns of CIR strategy vs benchmark
    ax = axes[1, 1]
    norm_ief_bh = (1.0 + bt_ief_bh.net_returns).cumprod()
    norm_cir_strat = (1.0 + bt_cir.net_returns).cumprod()
    ax.plot(norm_cir_strat.index, norm_cir_strat, color=PALETTE["teal"], lw=2.2, label=f"CIR Dynamic IEF/SHY (Sharpe {bt_cir.metrics.sharpe_ratio:.2f})")
    ax.plot(norm_ief_bh.index, norm_ief_bh, color=PALETTE["crimson"], lw=1.5, ls="--", label=f"Buy & Hold IEF (Sharpe {bt_ief_bh.metrics.sharpe_ratio:.2f})")
    ax.set_title("(d) Cumulative Returns: CIR Strategy vs. Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth ($)")
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_backtest_performance.png")
    plt.close(fig)

    return perf_dict, bt_goog_pair, bt_cir


# =============================================================================
# 4. Sensitivity Analysis
# =============================================================================
def run_sensitivity_study(prices_df: pd.DataFrame, eval_idx: int) -> Dict[str, Any]:
    print("\n[4/4] Running Sensitivity Analysis...")

    # Grid search over entry and exit thresholds
    z_entries = np.linspace(0.75, 2.0, 6)
    z_exits = np.linspace(0.10, 0.40, 5)
    sharpe_grid = np.zeros((len(z_exits), len(z_entries)))

    for i_ex, ze in enumerate(z_exits):
        for j_en, zn in enumerate(z_entries):
            if zn <= ze:
                sharpe_grid[i_ex, j_en] = np.nan
                continue
            cfg = OUTradingConfig(z_entry=float(zn), z_exit=float(ze), z_stop=3.0, lookback=120, recalib_step=10, tc_bps=5.0)
            bt, _ = run_ou_pairs_strategy(prices_df["GOOGL"], prices_df["GOOG"], cfg, start_eval_idx=eval_idx, name="OU_Sens")
            sharpe_grid[i_ex, j_en] = bt.metrics.sharpe_ratio

    # In-sample vs out-of-sample comparison
    cfg_base = OUTradingConfig(z_entry=1.25, z_exit=0.20, z_stop=3.0, lookback=120, recalib_step=5, tc_bps=5.0, execution_lag=1)
    bt_is, _ = run_ou_pairs_strategy(prices_df["GOOGL"].iloc[:eval_idx], prices_df["GOOG"].iloc[:eval_idx], cfg_base, start_eval_idx=120, name="OU IS")
    bt_oos, _ = run_ou_pairs_strategy(prices_df["GOOGL"], prices_df["GOOG"], cfg_base, start_eval_idx=eval_idx, name="OU OOS")

    # Plot Figure 4: Sensitivity Diagnostics (1x2 layout)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # (a) Sensitivity to entry and exit thresholds
    ax = axes[0]
    sns.heatmap(
        sharpe_grid, annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=[f"{z:.2f}" for z in z_entries],
        yticklabels=[f"{z:.2f}" for z in z_exits],
        cbar_kws={"label": "Sharpe Ratio"}, ax=ax,
    )
    ax.set_title(r"(a) Sharpe Ratio by Thresholds $(z_{\mathrm{entry}}, z_{\mathrm{exit}})$")
    ax.set_xlabel(r"Entry Threshold $z_{\mathrm{entry}}$")
    ax.set_ylabel(r"Exit Threshold $z_{\mathrm{exit}}$")

    # (b) In-sample vs out-of-sample performance metrics
    ax = axes[1]
    metrics_names = ["Sharpe", "Sortino", "Win % / 100", "MaxDD % / 10"]
    is_vals = [bt_is.metrics.sharpe_ratio, bt_is.metrics.sortino_ratio, bt_is.metrics.hit_rate / 100.0, abs(bt_is.metrics.max_drawdown) * 10.0]
    oos_vals = [bt_oos.metrics.sharpe_ratio, bt_oos.metrics.sortino_ratio, bt_oos.metrics.hit_rate / 100.0, abs(bt_oos.metrics.max_drawdown) * 10.0]
    x_idx = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(x_idx - width/2, is_vals, width, label="In-Sample (2019-2022)", color=PALETTE["navy"])
    ax.bar(x_idx + width/2, oos_vals, width, label="Out-of-Sample (2023-2024)", color=PALETTE["teal"])
    ax.set_xticks(x_idx)
    ax.set_xticklabels(metrics_names)
    ax.set_title("(b) In-Sample vs. Out-of-Sample Performance")
    ax.set_ylabel("Metric Value")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_sensitivity_analysis.png")
    plt.close(fig)

    return {
        "z_entries": z_entries.tolist(),
        "z_exits": z_exits.tolist(),
        "is_metrics": {k: float(v) for k, v in bt_is.metrics.to_dict().items()},
        "oos_metrics": {k: float(v) for k, v in bt_oos.metrics.to_dict().items()},
    }


def main() -> None:
    print("=" * 70)
    print("MAC F244 Stochastic Calculus Project")
    print("OU and CIR Model Experiments: Simulation, Calibration, and Trading")
    print("=" * 70)

    tickers = ["GOOGL", "GOOG", "IEF", "SHY", "^TNX", "^VIX"]
    db_path = DATA_DIR / "market_data_goog.db"
    bundle = fetch_market_data(
        tickers=tickers,
        start="2019-01-01",
        end="2025-01-01",
        cache_path=DATA_DIR / "market_prices_goog.csv",
        db_path=db_path,
    )
    # Train/test split index (start of 2023)
    eval_idx = bundle.prices.index.get_loc(bundle.prices.loc["2023-01-01":].index[0])
    print(
        f"Data range: {bundle.start_date} to {bundle.end_date} ({bundle.n_trading_days} days) | "
        f"Train: {eval_idx} days (2019–2022) | Test: {bundle.n_trading_days - eval_idx} days (2023–2024)"
    )

    sim_res = run_simulation_and_discretization_study()
    cal_res = run_calibration_study(bundle.prices, train_idx=eval_idx)
    bt_res, bt_ou, bt_cir = run_empirical_backtests(bundle.prices, eval_idx=eval_idx)
    sens_res = run_sensitivity_study(bundle.prices, eval_idx=eval_idx)

    summary = {
        "dataset": {
            "tickers": tickers,
            "start_date": bundle.start_date,
            "end_date": bundle.end_date,
            "train_days": int(eval_idx),
            "test_days": int(bundle.n_trading_days - eval_idx),
            "oos_start_date": str(bundle.prices.index[eval_idx].date()),
        },
        "simulation": sim_res,
        "calibration": cal_res,
        "backtest": bt_res,
        "sensitivity": sens_res,
    }

    summary_path = DATA_DIR / "experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nExperiments completed successfully. Figures saved to figures/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
