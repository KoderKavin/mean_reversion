"""
MAC F244 Stochastic Calculus & Financial Mathematics Project
Topic 2: Mean-Reverting Stochastic Processes (Ornstein-Uhlenbeck & Cox-Ingersoll-Ross)

Package exposing SDE simulation, analytical MLE/OLS calibration,
real market data ingestion, algorithmic trading strategies, and backtesting.
"""

from .models import (
    OUParams,
    CIRParams,
    ou_analytical_moments,
    ou_transition_pdf,
    ou_stationary_pdf,
    simulate_ou_exact,
    simulate_ou_euler,
    cir_analytical_moments,
    cir_transition_pdf,
    cir_stationary_pdf,
    simulate_cir_exact,
    simulate_cir_euler,
)
from .calibration import (
    OUCalibrationResult,
    CIRCalibrationResult,
    calibrate_ou_mle,
    calibrate_ou_ols,
    ou_log_likelihood,
    calibrate_cir_mle,
    calibrate_cir_cls,
    calibrate_cir_wls,
    cir_log_likelihood,
)
from .data_loader import (
    MarketDataBundle,
    fetch_market_data,
    test_pair_cointegration,
    construct_cointegrated_spread,
    construct_rolling_spread,
)
from .strategy import (
    OUTradingConfig,
    CIRTradingConfig,
    generate_ou_signals,
    run_ou_pairs_strategy,
    run_cir_yield_strategy,
)
from .backtest import (
    BacktestMetrics,
    BacktestResult,
    evaluate_backtest,
    compute_performance_metrics,
)

__all__ = [
    "OUParams",
    "CIRParams",
    "ou_analytical_moments",
    "ou_transition_pdf",
    "ou_stationary_pdf",
    "simulate_ou_exact",
    "simulate_ou_euler",
    "cir_analytical_moments",
    "cir_transition_pdf",
    "cir_stationary_pdf",
    "simulate_cir_exact",
    "simulate_cir_euler",
    "OUCalibrationResult",
    "CIRCalibrationResult",
    "calibrate_ou_mle",
    "calibrate_ou_ols",
    "ou_log_likelihood",
    "calibrate_cir_mle",
    "calibrate_cir_cls",
    "calibrate_cir_wls",
    "cir_log_likelihood",
    "MarketDataBundle",
    "fetch_market_data",
    "test_pair_cointegration",
    "construct_cointegrated_spread",
    "construct_rolling_spread",
    "OUTradingConfig",
    "CIRTradingConfig",
    "generate_ou_signals",
    "run_ou_pairs_strategy",
    "run_cir_yield_strategy",
    "BacktestMetrics",
    "BacktestResult",
    "evaluate_backtest",
    "compute_performance_metrics",
]
