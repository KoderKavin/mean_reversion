"""
backtest.py
-----------
Institutional performance attribution engine for continuous/discrete trading
strategies with realistic market frictions: proportional transaction costs
(commissions + bid-ask slippage), short-borrowing fees, and execution lag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Comprehensive risk-adjusted return, drawdown, and turnover statistics."""

    total_return: float
    cagr: float
    annual_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    profit_factor: float
    avg_holding_days: float
    num_trades: int
    annual_turnover: float
    total_cost_drag: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "Total Return (%)": 100.0 * self.total_return,
            "CAGR (%)": 100.0 * self.cagr,
            "Ann. Vol (%)": 100.0 * self.annual_vol,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Max Drawdown (%)": 100.0 * self.max_drawdown,
            "Calmar Ratio": self.calmar_ratio,
            "Hit Rate (%)": 100.0 * self.hit_rate,
            "Profit Factor": self.profit_factor,
            "Avg Hold (Days)": self.avg_holding_days,
            "Trades": float(self.num_trades),
            "Ann. Turnover (x)": self.annual_turnover,
            "Cost Drag (%)": 100.0 * self.total_cost_drag,
        }


@dataclass
class BacktestResult:
    """Full time-series output and summary metrics of a strategy backtest."""

    name: str
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    drawdown: pd.Series
    positions: pd.Series
    turnover: pd.Series
    metrics: BacktestMetrics


def _estimate_trade_stats(
    positions: np.ndarray,
    net_returns: np.ndarray,
) -> tuple[int, float, float, float]:
    """
    Segment active holding episodes (non-zero signal regimes) to compute exact
    trade count, average holding period (days), trade-level win rate, and profit factor.
    """
    n = len(positions)
    in_trade = False
    trade_pnls = []
    trade_lengths = []
    curr_pnl = 0.0
    curr_len = 0
    prev_sign = 0.0

    for i in range(n):
        sig = np.sign(positions[i]) if abs(positions[i]) > 1e-8 else 0.0
        if not in_trade:
            if sig != 0.0:
                in_trade = True
                prev_sign = sig
                curr_pnl = net_returns[i]
                curr_len = 1
        else:
            if sig == prev_sign:
                curr_pnl += net_returns[i]
                curr_len += 1
            else:
                trade_pnls.append(curr_pnl)
                trade_lengths.append(curr_len)
                if sig != 0.0:
                    in_trade = True
                    prev_sign = sig
                    curr_pnl = net_returns[i]
                    curr_len = 1
                else:
                    in_trade = False
                    prev_sign = 0.0

    if in_trade and curr_len > 0:
        trade_pnls.append(curr_pnl)
        trade_lengths.append(curr_len)

    num_trades = len(trade_pnls)
    if num_trades == 0:
        return 0, 0.0, 0.0, 0.0

    pnl_arr = np.asarray(trade_pnls, dtype=np.float64)
    avg_hold = float(np.mean(trade_lengths))
    hit_rate = float(np.mean(pnl_arr > 0.0))
    gains = float(np.sum(pnl_arr[pnl_arr > 0.0]))
    losses = float(-np.sum(pnl_arr[pnl_arr < 0.0]))
    profit_factor = gains / losses if losses > 1e-12 else (np.inf if gains > 0 else 0.0)
    return num_trades, avg_hold, hit_rate, profit_factor


def compute_performance_metrics(
    net_returns: pd.Series,
    gross_returns: Optional[pd.Series] = None,
    positions: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """Compute annualized return, risk, drawdown, and trade efficiency statistics."""
    r = net_returns.fillna(0.0).to_numpy(dtype=np.float64)
    n = len(r)
    if n == 0:
        raise ValueError("Empty return series passed to compute_performance_metrics.")

    equity = np.cumprod(1.0 + r)
    total_ret = float(equity[-1] - 1.0)
    years = n / float(periods_per_year)
    cagr = float(equity[-1] ** (1.0 / max(years, 1e-6)) - 1.0) if equity[-1] > 0 else -1.0

    rf_daily = rf_annual / float(periods_per_year)
    excess = r - rf_daily
    mean_ex = float(np.mean(excess))
    std_ret = float(np.std(r, ddof=1)) if n > 1 else 0.0
    ann_vol = std_ret * np.sqrt(periods_per_year)

    sharpe = (mean_ex / std_ret) * np.sqrt(periods_per_year) if std_ret > 1e-12 else 0.0

    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    sortino = (mean_ex / downside_dev) * np.sqrt(periods_per_year) if downside_dev > 1e-12 else 0.0

    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    max_dd = float(np.min(dd))
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    if positions is not None:
        pos_arr = positions.fillna(0.0).to_numpy(dtype=np.float64)
        num_trades, avg_hold, hit_rate, profit_factor = _estimate_trade_stats(pos_arr, r)
        # For continuously invested portfolios/benchmarks (1 continuous episode), report daily win rate & PF
        if num_trades <= 1:
            active = r[np.abs(r) > 1e-10]
            hit_rate = float(np.mean(active > 0.0)) if len(active) > 0 else 0.0
            gains = float(np.sum(active[active > 0.0]))
            losses = float(-np.sum(active[active < 0.0]))
            profit_factor = gains / losses if losses > 1e-12 else 0.0
    else:
        active = r[np.abs(r) > 1e-10]
        num_trades = 0
        avg_hold = 0.0
        hit_rate = float(np.mean(active > 0.0)) if len(active) > 0 else 0.0
        gains = float(np.sum(active[active > 0.0]))
        losses = float(-np.sum(active[active < 0.0]))
        profit_factor = gains / losses if losses > 1e-12 else 0.0

    ann_turnover = (
        float(turnover.fillna(0.0).mean() * periods_per_year)
        if turnover is not None
        else 0.0
    )

    if gross_returns is not None:
        gross_eq = float(np.prod(1.0 + gross_returns.fillna(0.0).to_numpy(dtype=np.float64)) - 1.0)
        cost_drag = max(gross_eq - total_ret, 0.0)
    else:
        cost_drag = 0.0

    return BacktestMetrics(
        total_return=total_ret,
        cagr=cagr,
        annual_vol=ann_vol,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=max_dd,
        calmar_ratio=float(calmar),
        hit_rate=hit_rate,
        profit_factor=float(min(profit_factor, 99.9)),
        avg_holding_days=avg_hold,
        num_trades=num_trades,
        annual_turnover=ann_turnover,
        total_cost_drag=cost_drag,
    )


def evaluate_backtest(
    name: str,
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    signal_state: pd.Series,
    tc_bps: float = 5.0,
    borrow_bps_annual: float = 30.0,
    execution_lag: int = 1,
    rf_annual: float = 0.0,
) -> BacktestResult:
    """
    Execute multi-leg portfolio accounting with strict execution delay:
        held_weights[t] = target_weights[t - execution_lag]
        gross_ret[t]    = sum_k held_weights[t, k] * asset_returns[t, k]
        turnover[t]     = sum_k |held_weights[t, k] - held_weights[t - 1, k]|
        short_notional  = sum_k max(-held_weights[t, k], 0)
        net_ret[t]      = gross_ret[t] - turnover[t] * (tc_bps * 1e-4)
                          - short_notional * (borrow_bps_annual * 1e-4 / 252)
    """
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0 (use >= 1 to prevent look-ahead bias).")

    aligned_rets = asset_returns.fillna(0.0)
    held_weights = target_weights.shift(execution_lag).fillna(0.0)
    lagged_signal = signal_state.shift(execution_lag).fillna(0.0)

    gross_ret = (held_weights * aligned_rets).sum(axis=1)
    weight_changes = held_weights.diff().fillna(held_weights.iloc[0])
    turnover = weight_changes.abs().sum(axis=1)

    short_notional = (-held_weights).clip(lower=0.0).sum(axis=1)
    tc_cost = turnover * (tc_bps * 1e-4)
    borrow_cost = short_notional * (borrow_bps_annual * 1e-4 / 252.0)

    net_ret = gross_ret - tc_cost - borrow_cost
    equity = (1.0 + net_ret).cumprod()
    gross_equity = (1.0 + gross_ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    metrics = compute_performance_metrics(
        net_returns=net_ret,
        gross_returns=gross_ret,
        positions=lagged_signal,
        turnover=turnover,
        rf_annual=rf_annual,
    )

    return BacktestResult(
        name=name,
        equity_curve=equity,
        gross_equity_curve=gross_equity,
        net_returns=net_ret,
        gross_returns=gross_ret,
        drawdown=drawdown,
        positions=lagged_signal,
        turnover=turnover,
        metrics=metrics,
    )
