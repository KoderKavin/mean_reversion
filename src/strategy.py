"""
strategy.py
-----------
Algorithmic trading strategies built on continuous-time mean-reverting SDEs:

1. Ornstein-Uhlenbeck (OU) Statistical Arbitrage / Pairs Strategy:
   - Standardized equilibrium Z-score: Z_t = (X_t - mu_hat) / sigma_eq_hat,
     where sigma_eq_hat = sigma_hat / sqrt(2 * theta_hat).
   - Trade-coherent hedge locking: rolling causal OLS + OU MLE updates parameters
     while flat; upon trade entry at t_0, the hedge ratio (alpha_{t_0}, beta_{t_0})
     and OU equilibrium reference (mu_{t_0}, sigma_eq,{t_0}) are locked until exit,
     ensuring Z_t tracks the true mark-to-market convergence of the held spread
     and eliminating spurious daily beta-rebalancing turnover.
   - Hysteresis state machine with entry, mean-reversion exit, half-life stationarity
     filter, and stop-loss cooldown protection against structural breaks.

2. Cox-Ingersoll-Ross (CIR) Volatility-Adjusted Yield-Curve Swing Strategy:
   - Calibrates dr_t = kappa*(theta - r_t)dt + xi*sqrt(r_t)dW_t on strictly positive
     benchmark Treasury yields (^TNX) over rolling causal windows.
   - State-dependent heteroskedastic standardization:
         sigma_CIR(r_t) = sqrt( (xi_hat^2 / (2 * kappa_hat)) * r_t )
         Z_t^CIR        = (r_t - theta_hat) / sigma_CIR(r_t)
   - Trades fixed-income duration ETFs (IEF 7-10Y vs. SHY 1-3Y defensive carry),
     incorporating inverse-square-root volatility sizing sqrt(theta_hat / r_t),
     Feller-ratio stress dampening, and yield-breakout regime filtering.

3. CVXPY Transaction-Cost-Aware Multi-Strategy Allocation:
   - Solves a rolling convex quadratic program allocating between the market-neutral
     OU Equity Pairs strategies and the CIR Yield-Curve strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestResult, evaluate_backtest
from .calibration import calibrate_cir_cls, calibrate_ou_mle, calibrate_ou_ols
from .data_loader import construct_rolling_spread


@dataclass
class OUTradingConfig:
    """Hyperparameters and friction settings for the OU Pairs Strategy."""

    z_entry: float = 1.50
    z_exit: float = 0.25
    z_stop: float = 3.25
    lookback: int = 180
    recalib_step: int = 5
    max_half_life_days: float = 65.0
    gross_leverage: float = 1.5
    tc_bps: float = 5.0
    borrow_bps_annual: float = 25.0
    execution_lag: int = 1


@dataclass
class CIRTradingConfig:
    """Hyperparameters and friction settings for the CIR Yield-Curve Strategy."""

    z_entry: float = 1.00
    z_exit: float = 0.15
    z_stop: float = 2.25
    vol_stress_mult: float = 1.15
    lookback: int = 252
    recalib_step: int = 10
    tc_bps: float = 3.0
    borrow_bps_annual: float = 0.0
    execution_lag: int = 1


def generate_ou_signals(
    z_scores: np.ndarray,
    z_entry: float = 1.5,
    z_exit: float = 0.25,
    z_stop: float = 3.25,
    half_lives_days: Optional[np.ndarray] = None,
    max_half_life_days: float = 65.0,
) -> np.ndarray:
    """
    Stateful hysteresis trading rule on standardized Z-score series Z_t:
      - Long (+1) when Z_t <= -z_entry; Short (-1) when Z_t >= +z_entry.
      - Exit (0) when |Z_t| <= z_exit (or overshoot across mean) or |Z_t| >= z_stop.
      - Cooldown after stop-loss: requires |Z_t| <= max(z_exit, 0.75 * z_entry)
        before re-entering to prevent whipsaw re-entry during structural divergence.
    """
    n = len(z_scores)
    signals = np.zeros(n, dtype=np.float64)
    curr_pos = 0.0
    stopped_out = False

    for t in range(n):
        z = z_scores[t]
        if not np.isfinite(z):
            signals[t] = 0.0
            curr_pos = 0.0
            continue

        hl_ok = True
        if half_lives_days is not None and np.isfinite(half_lives_days[t]):
            hl_ok = bool(half_lives_days[t] <= max_half_life_days)

        if stopped_out:
            if abs(z) <= max(z_exit, 0.75 * z_entry):
                stopped_out = False
            else:
                signals[t] = 0.0
                continue

        if curr_pos == 0.0:
            if hl_ok and (-z_stop < z <= -z_entry):
                curr_pos = 1.0
            elif hl_ok and (z_entry <= z < z_stop):
                curr_pos = -1.0
        elif curr_pos == 1.0:
            if z >= -z_exit:
                curr_pos = 0.0
            elif z <= -z_stop:
                curr_pos = 0.0
                stopped_out = True
        elif curr_pos == -1.0:
            if z <= z_exit:
                curr_pos = 0.0
            elif z >= z_stop:
                curr_pos = 0.0
                stopped_out = True

        signals[t] = curr_pos

    return signals


def run_ou_pairs_strategy(
    price_a: pd.Series,
    price_b: pd.Series,
    config: OUTradingConfig,
    start_eval_idx: Optional[int] = None,
    name: str = "OU Pairs Strategy",
) -> Tuple[BacktestResult, pd.DataFrame]:
    """
    End-to-end causal walk-forward OU pairs trading pipeline with trade-coherent
    hedge locking:
      1. Rolling OLS log-price cointegration spread X_t = ln P_A,t - (alpha_t + beta_t ln P_B,t).
      2. Rolling analytical MLE calibration of (theta_t, mu_t, sigma_t) on [t - W + 1, t].
      3. Equilibrium Z-score Z_t = (X_t - mu_t) / sigma_eq,t with sigma_eq,t = sigma_t / sqrt(2*theta_t).
      4. Upon entering a position, (alpha, beta, mu, sigma_eq) are locked for the active trade
         so exit triggers reflect genuine price convergence rather than rolling window drift.
    """
    dt = 1.0 / 252.0
    W = config.lookback
    spread_df = construct_rolling_spread(price_a, price_b, lookback=W, use_log=True)

    n = len(price_a)
    theta_arr = np.full(n, np.nan)
    mu_arr = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)
    sigma_eq_arr = np.full(n, np.nan)
    hl_days_arr = np.full(n, np.nan)
    z_arr = np.full(n, np.nan)
    active_beta_arr = np.full(n, np.nan)
    raw_signals = np.zeros(n, dtype=np.float64)

    log_a = spread_df["log_a"].to_numpy()
    log_b = spread_df["log_b"].to_numpy()
    alpha_vec = spread_df["alpha"].to_numpy()
    beta_vec = spread_df["beta"].to_numpy()

    last_params = None
    curr_pos = 0.0
    stopped_out = False
    locked_alpha = 0.0
    locked_beta = 1.0
    locked_mu = 0.0
    locked_scale = 1.0

    for t in range(W - 1, n):
        win_spread = log_a[t - W + 1 : t + 1] - (alpha_vec[t] + beta_vec[t] * log_b[t - W + 1 : t + 1])

        if last_params is None or (t - (W - 1)) % config.recalib_step == 0:
            cal = calibrate_ou_mle(win_spread, dt=dt, compute_se=False, theta_bounds=(0.1, 200.0))
            last_params = cal.params

        theta_arr[t] = last_params.theta
        mu_arr[t] = last_params.mu
        sigma_arr[t] = last_params.sigma
        hl_days = last_params.half_life * 252.0
        hl_days_arr[t] = hl_days

        emp_std = float(np.std(win_spread, ddof=1))
        rolling_scale = max(
            0.65 * min(last_params.sigma_eq, 3.0 * emp_std) + 0.35 * emp_std,
            1e-6,
        )
        sigma_eq_arr[t] = rolling_scale
        z_rolling = (win_spread[-1] - last_params.mu) / rolling_scale

        if curr_pos == 0.0:
            z_eval = z_rolling
            active_beta = beta_vec[t]
            if stopped_out and abs(z_rolling) <= max(config.z_exit, 0.75 * config.z_entry):
                stopped_out = False

            hl_ok = hl_days <= config.max_half_life_days and active_beta > 0.05
            if not stopped_out and hl_ok:
                if -config.z_stop < z_eval <= -config.z_entry:
                    curr_pos = 1.0
                    locked_alpha = alpha_vec[t]
                    locked_beta = active_beta
                    locked_mu = last_params.mu
                    locked_scale = rolling_scale
                elif config.z_entry <= z_eval < config.z_stop:
                    curr_pos = -1.0
                    locked_alpha = alpha_vec[t]
                    locked_beta = active_beta
                    locked_mu = last_params.mu
                    locked_scale = rolling_scale
        else:
            # Evaluate held trade against locked entry cointegration vector
            held_spread_t = log_a[t] - (locked_alpha + locked_beta * log_b[t])
            z_eval = (held_spread_t - locked_mu) / locked_scale
            active_beta = locked_beta

            if curr_pos == 1.0:
                if z_eval >= -config.z_exit:
                    curr_pos = 0.0
                elif z_eval <= -config.z_stop:
                    curr_pos = 0.0
                    stopped_out = True
            elif curr_pos == -1.0:
                if z_eval <= config.z_exit:
                    curr_pos = 0.0
                elif z_eval >= config.z_stop:
                    curr_pos = 0.0
                    stopped_out = True

        z_arr[t] = z_eval
        raw_signals[t] = curr_pos
        active_beta_arr[t] = locked_beta if curr_pos != 0.0 else beta_vec[t]

    safe_beta = np.nan_to_num(active_beta_arr, nan=1.0)
    norm = 1.0 + np.abs(safe_beta)
    w_a = config.gross_leverage * raw_signals / norm
    w_b = -config.gross_leverage * raw_signals * safe_beta / norm

    weights_df = pd.DataFrame({"A": w_a, "B": w_b}, index=price_a.index)
    asset_rets = pd.DataFrame(
        {"A": price_a.pct_change().fillna(0.0), "B": price_b.pct_change().fillna(0.0)},
        index=price_a.index,
    )
    sig_series = pd.Series(raw_signals, index=price_a.index, name="signal")

    diagnostics = pd.DataFrame(
        {
            "spread": spread_df["spread"],
            "beta": active_beta_arr,
            "theta": theta_arr,
            "mu": mu_arr,
            "sigma": sigma_arr,
            "sigma_eq": sigma_eq_arr,
            "half_life_days": hl_days_arr,
            "z_score": z_arr,
            "signal": raw_signals,
        },
        index=price_a.index,
    )

    eval_slice = slice(start_eval_idx, None) if start_eval_idx is not None else slice(W, None)
    bt_res = evaluate_backtest(
        name=name,
        asset_returns=asset_rets.iloc[eval_slice],
        target_weights=weights_df.iloc[eval_slice],
        signal_state=sig_series.iloc[eval_slice],
        tc_bps=config.tc_bps,
        borrow_bps_annual=config.borrow_bps_annual,
        execution_lag=config.execution_lag,
    )
    return bt_res, diagnostics.iloc[eval_slice]


def run_cir_yield_strategy(
    yield_series: pd.Series,
    long_bond_etf: pd.Series,
    short_bond_etf: pd.Series,
    config: CIRTradingConfig,
    start_eval_idx: Optional[int] = None,
    name: str = "CIR Yield-Curve Strategy",
) -> Tuple[BacktestResult, pd.DataFrame]:
    """
    Volatility-adjusted interest rate / yield-ETF regime-switching strategy driven
    by dual Cox-Ingersoll-Ross (CIR) square-root diffusions:
      1. Calibrates CIR on benchmark 10Y Treasury decimal yield r_t (^TNX / 100) AND
         on causal 21-day annualized realized yield volatility v_t (strictly > 0).
      2. Computes heteroskedastic CIR yield score:
             sigma_CIR(r_t) = sqrt( (xi_r^2 / (2 * kappa_r)) * r_t )
             Z_t^CIR        = (r_t - theta_eff,t) / sigma_eff,t
         and CIR rate-volatility stress indicator: v_t > vol_stress_mult * theta_v,t.
      3. Allocates dynamically between 7-10Y Treasury Bonds (`IEF`) and 1-3Y Treasury
         Carry (`SHY`) with regime-locked inverse-square-root volatility weights:
           * Bull Yield-Reversion Regime (+1): High yield below breakout stop
             (z_entry <= Z_t^CIR < z_stop) & contained rate vol -> Overweight IEF.
           * Neutral Carry Core (0): Equilibrium yield & calm rate vol -> 50% vol-scaled IEF + SHY.
           * Defensive Rate-Shock / Low-Yield Regime (-1): Depressed yield (Z_t^CIR <= -0.85*z_entry),
             breakout stop (Z_t^CIR >= z_stop), or CIR vol stress -> 100% SHY capital preservation.
    """
    dt = 1.0 / 252.0
    W = config.lookback
    r_arr = np.maximum(yield_series.to_numpy(dtype=np.float64) / 100.0, 1e-4)
    rv_arr = (
        pd.Series(r_arr, index=yield_series.index)
        .diff()
        .rolling(21)
        .std()
        .bfill()
        .clip(lower=1e-4)
        .to_numpy(dtype=np.float64)
        * np.sqrt(252.0)
    )
    n = len(r_arr)

    kappa_arr = np.full(n, np.nan)
    theta_arr = np.full(n, np.nan)
    xi_arr = np.full(n, np.nan)
    feller_arr = np.full(n, np.nan)
    z_cir_arr = np.full(n, np.nan)
    hl_days_arr = np.full(n, np.nan)
    rv_theta_arr = np.full(n, np.nan)
    duration_regime = np.zeros(n, dtype=np.float64)
    w_ief = np.zeros(n, dtype=np.float64)
    w_shy = np.ones(n, dtype=np.float64)

    cal_r = None
    cal_v = None
    curr_regime = 0
    locked_w_ief = 0.50

    for t in range(W - 1, n):
        win_r = r_arr[t - W + 1 : t + 1]
        win_v = rv_arr[t - W + 1 : t + 1]
        if cal_r is None or (t - (W - 1)) % config.recalib_step == 0:
            cal_r = calibrate_cir_cls(win_r, dt=dt)
            cal_v = calibrate_cir_cls(win_v, dt=dt)

        kappa_arr[t] = cal_r.kappa
        theta_arr[t] = cal_r.theta
        xi_arr[t] = cal_r.xi
        feller_arr[t] = cal_r.feller_ratio
        hl_days_arr[t] = cal_r.half_life * 252.0
        rv_theta_arr[t] = cal_v.theta

        cir_scale = np.sqrt((cal_r.xi ** 2) / (2.0 * cal_r.kappa) * r_arr[t])
        emp_scale = float(np.std(win_r, ddof=1))
        eff_scale = max(0.50 * min(cir_scale, 3.0 * emp_scale) + 0.50 * emp_scale, 1e-5)
        eff_theta = 0.50 * cal_r.theta + 0.50 * float(np.mean(win_r))
        z_cir = (r_arr[t] - eff_theta) / eff_scale
        z_cir_arr[t] = z_cir

        vol_stress = bool(rv_arr[t] > config.vol_stress_mult * cal_v.theta)
        low_yield_thresh = -0.85 * config.z_entry

        prev_regime = curr_regime
        if curr_regime == 0:
            if (config.z_entry <= z_cir < config.z_stop) and not vol_stress:
                curr_regime = 1
            elif (z_cir <= low_yield_thresh) or (z_cir >= config.z_stop) or vol_stress:
                curr_regime = -1
        elif curr_regime == 1:
            if (z_cir <= config.z_exit) or (z_cir >= config.z_stop) or vol_stress:
                curr_regime = -1 if (z_cir >= config.z_stop or vol_stress) else 0
        elif curr_regime == -1:
            if (-0.35 * config.z_entry <= z_cir < 0.80 * config.z_stop) and not vol_stress:
                curr_regime = 1 if z_cir >= config.z_entry else 0

        if curr_regime != prev_regime or t == W - 1:
            vol_factor = float(np.clip(np.sqrt(eff_theta / max(r_arr[t], 1e-4)), 0.70, 1.15))
            if cal_r.feller_ratio < 1.0:
                vol_factor *= 0.85
            if curr_regime == 1:
                locked_w_ief = vol_factor
            elif curr_regime == 0:
                locked_w_ief = 0.50 * vol_factor
            else:
                locked_w_ief = 0.0

        w_ief[t] = locked_w_ief
        w_shy[t] = max(0.0, 1.0 - locked_w_ief)
        duration_regime[t] = float(curr_regime)

    weights_df = pd.DataFrame({"IEF": w_ief, "SHY": w_shy}, index=yield_series.index)
    asset_rets = pd.DataFrame(
        {
            "IEF": long_bond_etf.pct_change().fillna(0.0),
            "SHY": short_bond_etf.pct_change().fillna(0.0),
        },
        index=yield_series.index,
    )
    sig_series = pd.Series(duration_regime, index=yield_series.index, name="duration_signal")

    diagnostics = pd.DataFrame(
        {
            "rate_decimal": r_arr,
            "rate_vol": rv_arr,
            "rate_vol_theta": rv_theta_arr,
            "kappa": kappa_arr,
            "theta": theta_arr,
            "xi": xi_arr,
            "feller_ratio": feller_arr,
            "half_life_days": hl_days_arr,
            "z_cir": z_cir_arr,
            "signal": duration_regime,
            "w_IEF": w_ief,
            "w_SHY": w_shy,
        },
        index=yield_series.index,
    )

    eval_slice = slice(start_eval_idx, None) if start_eval_idx is not None else slice(W, None)
    bt_res = evaluate_backtest(
        name=name,
        asset_returns=asset_rets.iloc[eval_slice],
        target_weights=weights_df.iloc[eval_slice],
        signal_state=sig_series.iloc[eval_slice],
        tc_bps=config.tc_bps,
        borrow_bps_annual=config.borrow_bps_annual,
        execution_lag=config.execution_lag,
    )
    return bt_res, diagnostics.iloc[eval_slice]

