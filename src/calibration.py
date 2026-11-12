"""
calibration.py
--------------
Analytical and numerical parameter estimation routines for discrete-time
observations of Ornstein-Uhlenbeck (OU) and Cox-Ingersoll-Ross (CIR) processes.

Implemented Estimators:
1. OU Exact Analytical Maximum Likelihood Estimation (MLE) + Asymptotic SEs
2. OU Ordinary Least Squares (OLS) AR(1) Representation (Exact & Euler mappings)
3. CIR Weighted Conditional Least Squares (CLS) Initial Consistent Estimator
4. CIR Exact Maximum Likelihood Estimation (MLE) via Exponentially Scaled
   Modified Bessel Functions I_q^e(z) (Kladivko, 2007)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import statsmodels.api as sm
from scipy import optimize, special

from .models import CIRParams, OUParams


@dataclass
class OUCalibrationResult:
    """Container for Ornstein-Uhlenbeck parameter calibration diagnostics."""

    params: OUParams
    method: str
    log_likelihood: float
    aic: float
    bic: float
    phi: float
    sigma_eps: float
    std_errors: Dict[str, float]
    euler_theta: Optional[float] = None
    euler_sigma: Optional[float] = None
    r_squared: Optional[float] = None
    n_obs: int = 0


@dataclass
class CIRCalibrationResult:
    """Container for Cox-Ingersoll-Ross parameter calibration diagnostics."""

    params: CIRParams
    method: str
    log_likelihood: float
    aic: float
    bic: float
    feller_ratio: float
    feller_satisfied: bool
    std_errors: Dict[str, float]
    converged: bool = True
    n_obs: int = 0


# =============================================================================
# Ornstein-Uhlenbeck (OU) Calibration
# =============================================================================


def ou_log_likelihood(
    theta: float,
    mu: float,
    sigma: float,
    data: np.ndarray,
    dt: float,
) -> float:
    """
    Exact conditional Gaussian log-likelihood of discrete OU observations X_0, ..., X_N:
        ell(theta, mu, sigma) = -N/2 ln(2*pi*sigma_eps^2)
                                - 1/(2*sigma_eps^2) sum_{i=1}^N (X_i - mu - (X_{i-1}-mu)e^{-theta dt})^2
    where sigma_eps^2 = (sigma^2 / (2*theta)) * (1 - e^{-2*theta*dt}).
    """
    if theta <= 1e-10 or sigma <= 1e-10:
        return -1e15

    x_prev = data[:-1]
    x_next = data[1:]
    n = len(x_prev)

    phi = np.exp(-theta * dt)
    one_minus_phi2 = 1.0 - phi * phi
    if one_minus_phi2 <= 1e-15:
        sigma_eps2 = (sigma ** 2) * dt
    else:
        sigma_eps2 = (sigma ** 2) / (2.0 * theta) * one_minus_phi2

    if sigma_eps2 <= 1e-20 or not np.isfinite(sigma_eps2):
        return -1e15

    residuals = x_next - (mu + (x_prev - mu) * phi)
    rss = float(np.dot(residuals, residuals))
    ll = -0.5 * n * np.log(2.0 * np.pi * sigma_eps2) - 0.5 * rss / sigma_eps2
    return float(ll) if np.isfinite(ll) else -1e15


def _numerical_hessian_se_ou(
    theta: float,
    mu: float,
    sigma: float,
    data: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """Compute asymptotic standard errors via finite-difference observed Fisher Information."""
    p0 = np.array([theta, mu, sigma], dtype=np.float64)
    eps = np.maximum(1e-5 * np.abs(p0), 1e-6)
    k = 3
    hess = np.zeros((k, k), dtype=np.float64)

    def nll(p: np.ndarray) -> float:
        return -ou_log_likelihood(float(p[0]), float(p[1]), float(p[2]), data, dt)

    f0 = nll(p0)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = eps[i]
        hess[i, i] = (nll(p0 + ei) - 2.0 * f0 + nll(p0 - ei)) / (eps[i] ** 2)
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = eps[j]
            val = (
                nll(p0 + ei + ej)
                - nll(p0 + ei - ej)
                - nll(p0 - ei + ej)
                + nll(p0 - ei - ej)
            ) / (4.0 * eps[i] * eps[j])
            hess[i, j] = val
            hess[j, i] = val

    try:
        cov = np.linalg.pinv(hess)
        diag = np.maximum(np.diag(cov), 0.0)
        se = np.sqrt(diag)
        return {"theta": float(se[0]), "mu": float(se[1]), "sigma": float(se[2])}
    except np.linalg.LinAlgError:
        return {"theta": np.nan, "mu": np.nan, "sigma": np.nan}


def calibrate_ou_mle(
    data: np.ndarray,
    dt: float = 1.0 / 252.0,
    compute_se: bool = True,
    theta_bounds: tuple[float, float] = (1e-3, 250.0),
) -> OUCalibrationResult:
    """
    Analytical Maximum Likelihood Estimation (MLE) of OU parameters (theta, mu, sigma)
    from discrete equispaced observations X_0, X_1, ..., X_N.

    Closed-form first-order conditions (Tang & Chen, 2009; Franco, 2003):
        phi_hat = (N * S_xy - S_x * S_y) / (N * S_xx - S_x^2)
        mu_hat  = (S_y - phi_hat * S_x) / (N * (1 - phi_hat))
        theta_hat = -ln(phi_hat) / dt
        sigma_eps^2 = (1 / N) * sum (X_i - mu_hat - (X_{i-1} - mu_hat) * phi_hat)^2
        sigma_hat = sqrt( 2 * theta_hat * sigma_eps^2 / (1 - phi_hat^2) )

    Falls back cleanly to bounded L-BFGS-B optimization if finite-sample noise produces
    a boundary/unit-root phi_hat outside (0, 1).
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        raise ValueError("Need at least 5 finite observations to calibrate OU process.")

    x = arr[:-1]
    y = arr[1:]
    n = len(x)

    sx = float(np.sum(x))
    sy = float(np.sum(y))
    sxx = float(np.dot(x, x))
    sxy = float(np.dot(x, y))

    denom = n * sxx - sx * sx
    phi_hat = (n * sxy - sx * sy) / denom if abs(denom) > 1e-15 else 0.99

    phi_min = np.exp(-theta_bounds[1] * dt)
    phi_max = np.exp(-theta_bounds[0] * dt)

    if phi_min < phi_hat < phi_max:
        mu_hat = float((sy - phi_hat * sx) / (n * (1.0 - phi_hat)))
        theta_hat = float(-np.log(phi_hat) / dt)
        resid = y - (mu_hat + (x - mu_hat) * phi_hat)
        sigma_eps2 = float(np.dot(resid, resid)) / n
        sigma_eps = float(np.sqrt(max(sigma_eps2, 1e-18)))
        sigma_hat = float(np.sqrt(2.0 * theta_hat * sigma_eps2 / (1.0 - phi_hat * phi_hat)))
    else:
        # Bounded numerical MLE fallback for near-unit-root short windows
        mu_init = float(np.mean(arr))
        diff_std = float(np.std(np.diff(arr)) / np.sqrt(dt))
        res = optimize.minimize(
            fun=lambda p: -ou_log_likelihood(p[0], p[1], p[2], arr, dt),
            x0=[1.0, mu_init, max(diff_std, 1e-4)],
            bounds=[theta_bounds, (float(np.min(arr) - 5.0 * np.std(arr)), float(np.max(arr) + 5.0 * np.std(arr))), (1e-6, None)],
            method="L-BFGS-B",
        )
        theta_hat, mu_hat, sigma_hat = float(res.x[0]), float(res.x[1]), float(res.x[2])
        phi_hat = float(np.exp(-theta_hat * dt))
        sigma_eps = float(sigma_hat * np.sqrt((1.0 - phi_hat ** 2) / (2.0 * theta_hat)))

    params = OUParams(theta=theta_hat, mu=mu_hat, sigma=sigma_hat)
    ll = ou_log_likelihood(theta_hat, mu_hat, sigma_hat, arr, dt)
    k = 3
    aic = 2.0 * k - 2.0 * ll
    bic = k * np.log(n) - 2.0 * ll
    se = _numerical_hessian_se_ou(theta_hat, mu_hat, sigma_hat, arr, dt) if compute_se else {
        "theta": np.nan, "mu": np.nan, "sigma": np.nan
    }

    return OUCalibrationResult(
        params=params,
        method="Exact Analytical MLE",
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        phi=phi_hat,
        sigma_eps=sigma_eps,
        std_errors=se,
        n_obs=n,
    )


def calibrate_ou_ols(
    data: np.ndarray,
    dt: float = 1.0 / 252.0,
) -> OUCalibrationResult:
    """
    Calibrate OU parameters via AR(1) Ordinary Least Squares (OLS) regression:
        X_t = alpha + phi * X_{t-1} + eps_t,   eps_t ~ iid (0, s_eps^2).

    Computes both:
      1. Exact continuous-time mapping:
             theta = -ln(phi)/dt,  mu = alpha/(1 - phi),
             sigma = s_eps * sqrt(2*theta / (1 - phi^2))
      2. First-order Euler-Maruyama mapping:
             theta_Euler = (1 - phi)/dt,  sigma_Euler = s_eps / sqrt(dt)
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        raise ValueError("Need at least 5 finite observations for OU OLS calibration.")

    x = arr[:-1]
    y = arr[1:]
    n = len(x)

    X_design = sm.add_constant(x)
    ols_fit = sm.OLS(y, X_design).fit()

    alpha_hat = float(ols_fit.params[0])
    phi_hat = float(np.clip(ols_fit.params[1], 1e-6, 1.0 - 1e-6))
    # Unbiased residual standard error (ddof = 2)
    s_eps = float(np.sqrt(ols_fit.mse_resid))

    theta_hat = float(-np.log(phi_hat) / dt)
    mu_hat = float(alpha_hat / (1.0 - phi_hat))
    sigma_hat = float(s_eps * np.sqrt(2.0 * theta_hat / (1.0 - phi_hat * phi_hat)))

    euler_theta = float((1.0 - phi_hat) / dt)
    euler_sigma = float(s_eps / np.sqrt(dt))

    # Delta-method SEs from AR(1) OLS covariance
    se_phi = float(ols_fit.bse[1])
    se_alpha = float(ols_fit.bse[0])
    se_theta = se_phi / (phi_hat * dt)
    se_mu = np.sqrt((se_alpha / (1.0 - phi_hat)) ** 2 + (alpha_hat * se_phi / ((1.0 - phi_hat) ** 2)) ** 2)
    se_sigma = sigma_hat / np.sqrt(2.0 * max(n - 2, 1))

    params = OUParams(theta=theta_hat, mu=mu_hat, sigma=sigma_hat)
    ll = ou_log_likelihood(theta_hat, mu_hat, sigma_hat, arr, dt)
    k = 3

    return OUCalibrationResult(
        params=params,
        method="AR(1) OLS Regression",
        log_likelihood=ll,
        aic=2.0 * k - 2.0 * ll,
        bic=k * np.log(n) - 2.0 * ll,
        phi=phi_hat,
        sigma_eps=s_eps,
        std_errors={"theta": float(se_theta), "mu": float(se_mu), "sigma": float(se_sigma)},
        euler_theta=euler_theta,
        euler_sigma=euler_sigma,
        r_squared=float(ols_fit.rsquared),
        n_obs=n,
    )


# =============================================================================
# Cox-Ingersoll-Ross (CIR) Calibration
# =============================================================================


def cir_log_likelihood(
    kappa: float,
    theta: float,
    xi: float,
    data: np.ndarray,
    dt: float,
) -> float:
    """
    Exact log-likelihood of discrete CIR observations r_0, r_1, ..., r_N using the
    non-central chi-squared transition density expressed via the exponentially scaled
    modified Bessel function I_q^e(z) = e^{-z} I_q(z) (Kladivko, 2007):

        ln f(r_i | r_{i-1}) = ln(c) - (sqrt(u_i) - sqrt(v_i))^2
                              + (q / 2) * (ln(r_i / r_{i-1}) + kappa * dt)
                              + ln( I_q^e(2 * sqrt(u_i * v_i)) )

    where c = 2*kappa / (xi^2 * (1 - e^{-kappa*dt})), q = 2*kappa*theta / xi^2 - 1,
    u_i = c * r_{i-1} * e^{-kappa*dt}, and v_i = c * r_i.
    """
    if kappa <= 1e-8 or theta <= 1e-8 or xi <= 1e-8:
        return -1e15

    r_prev = data[:-1]
    r_next = data[1:]

    exp_k = np.exp(-kappa * dt)
    one_minus_exp = -np.expm1(-kappa * dt)
    if one_minus_exp <= 1e-15 or not np.isfinite(exp_k):
        return -1e15

    c = 2.0 * kappa / ((xi * xi) * one_minus_exp)
    q = 2.0 * kappa * theta / (xi * xi) - 1.0
    if not np.isfinite(c) or c <= 0.0 or q > 5000.0:
        return -1e15

    u = c * r_prev * exp_k
    v = c * r_next
    z = 2.0 * np.sqrt(u * v)

    bessel_scaled = special.ive(q, z)
    if np.any(bessel_scaled <= 0.0) or not np.all(np.isfinite(bessel_scaled)):
        return -1e15

    ll_terms = (
        np.log(c)
        - (np.sqrt(u) - np.sqrt(v)) ** 2
        + 0.5 * q * (np.log(r_next / r_prev) + kappa * dt)
        + np.log(bessel_scaled)
    )
    total_ll = float(np.sum(ll_terms))
    return total_ll if np.isfinite(total_ll) else -1e15


def calibrate_cir_wls(
    data: np.ndarray,
    dt: float = 1.0 / 252.0,
) -> CIRCalibrationResult:
    """
    Weighted Least Squares (WLS) / Conditional Least Squares estimator for CIR.
    Divides the discretized Euler increment by sqrt(r_{t-1}) to homoskedasticize errors:
        (r_i - r_{i-1}) / sqrt(r_{i-1}) = gamma_1 * (1 / sqrt(r_{i-1}))
                                          + gamma_2 * sqrt(r_{i-1}) + eta_i,
    where gamma_1 = (1 - e^{-kappa*dt}) * theta, gamma_2 = -(1 - e^{-kappa*dt}),
    and Var(eta_i) ≈ xi^2 * dt.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = np.maximum(arr[np.isfinite(arr)], 1e-8)
    n = len(arr) - 1

    r_prev = arr[:-1]
    r_next = arr[1:]
    sqrt_r = np.sqrt(r_prev)

    y = (r_next - r_prev) / sqrt_r
    X = np.column_stack([1.0 / sqrt_r, sqrt_r])
    ols_fit = sm.OLS(y, X).fit()

    gamma1 = float(ols_fit.params[0])
    gamma2 = float(ols_fit.params[1])

    one_minus_phi = float(np.clip(-gamma2, 1e-4, 0.95))
    kappa_hat = float(-np.log(1.0 - one_minus_phi) / dt)
    theta_hat = float(gamma1 / one_minus_phi) if gamma1 > 0.0 else float(np.mean(arr))
    theta_hat = float(np.clip(theta_hat, 0.1 * float(np.min(arr)), 5.0 * float(np.max(arr))))

    s_eps = float(np.sqrt(ols_fit.mse_resid))
    xi_hat = float(s_eps / np.sqrt(dt))

    # Delta method SEs
    se_g1 = float(ols_fit.bse[0])
    se_g2 = float(ols_fit.bse[1])
    se_kappa = float(se_g2 / ((1.0 - one_minus_phi) * dt))
    se_theta = float(np.sqrt((se_g1 / one_minus_phi) ** 2 + (gamma1 * se_g2 / (one_minus_phi ** 2)) ** 2))
    se_xi = float(xi_hat / np.sqrt(2.0 * max(n - 2, 1)))

    params = CIRParams(
        kappa=max(kappa_hat, 1e-3),
        theta=max(theta_hat, 1e-6),
        xi=max(xi_hat, 1e-4),
    )

    return CIRCalibrationResult(
        params=params,
        method="Weighted Least Squares (WLS)",
        log_likelihood=float(ols_fit.llf),
        aic=float(ols_fit.aic),
        bic=float(ols_fit.bic),
        feller_ratio=params.feller_ratio,
        feller_satisfied=params.feller_satisfied,
        std_errors={"kappa": se_kappa, "theta": se_theta, "xi": se_xi},
        converged=True,
        n_obs=n,
    )


def calibrate_cir_cls(
    data: np.ndarray,
    dt: float = 1.0 / 252.0,
) -> CIRParams:
    """Return CIRParams via Weighted Least Squares (WLS) calibration."""
    return calibrate_cir_wls(data, dt).params


def _numerical_hessian_se_cir(
    kappa: float,
    theta: float,
    xi: float,
    data: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """Compute asymptotic standard errors of CIR MLE via observed Fisher Information."""
    p0 = np.array([kappa, theta, xi], dtype=np.float64)
    eps = np.maximum(1e-4 * np.abs(p0), 1e-6)
    k = 3
    hess = np.zeros((k, k), dtype=np.float64)

    def nll(p: np.ndarray) -> float:
        return -cir_log_likelihood(float(p[0]), float(p[1]), float(p[2]), data, dt)

    f0 = nll(p0)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = eps[i]
        hess[i, i] = (nll(p0 + ei) - 2.0 * f0 + nll(p0 - ei)) / (eps[i] ** 2)
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = eps[j]
            val = (
                nll(p0 + ei + ej)
                - nll(p0 + ei - ej)
                - nll(p0 - ei + ej)
                + nll(p0 - ei - ej)
            ) / (4.0 * eps[i] * eps[j])
            hess[i, j] = val
            hess[j, i] = val

    try:
        cov = np.linalg.pinv(hess)
        diag = np.maximum(np.diag(cov), 0.0)
        se = np.sqrt(diag)
        return {"kappa": float(se[0]), "theta": float(se[1]), "xi": float(se[2])}
    except np.linalg.LinAlgError:
        return {"kappa": np.nan, "theta": np.nan, "xi": np.nan}


def calibrate_cir_mle(
    data: np.ndarray,
    dt: float = 1.0 / 252.0,
    compute_se: bool = True,
    kappa_bounds: tuple[float, float] = (1e-3, 150.0),
) -> CIRCalibrationResult:
    """
    Exact Maximum Likelihood Estimation (MLE) of CIR parameters (kappa, theta, xi).

    Two-stage algorithm (Kladivko, 2007):
      Stage 1: Weighted Conditional Least Squares (CLS) consistent initialization.
      Stage 2: Log-space L-BFGS-B optimization of exact Bessel transition log-likelihood,
               with Nelder-Mead simplex polish if gradient line-search stalls near q = -1.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        raise ValueError("Need at least 10 finite observations to calibrate CIR process.")
    if np.any(arr <= 0.0):
        raise ValueError("CIR calibration requires strictly positive observations (r_t > 0).")

    n = len(arr) - 1
    init_params = calibrate_cir_cls(arr, dt=dt)

    log_k_min, log_k_max = np.log(kappa_bounds[0]), np.log(kappa_bounds[1])
    log_th_min = np.log(max(0.05 * float(np.min(arr)), 1e-6))
    log_th_max = np.log(10.0 * float(np.max(arr)))
    log_xi_min, log_xi_max = np.log(1e-5), np.log(50.0 * max(init_params.xi, 1.0))

    x0_log = np.array([
        np.clip(np.log(init_params.kappa), log_k_min + 1e-3, log_k_max - 1e-3),
        np.clip(np.log(init_params.theta), log_th_min + 1e-3, log_th_max - 1e-3),
        np.clip(np.log(init_params.xi), log_xi_min + 1e-3, log_xi_max - 1e-3),
    ])

    def neg_ll_log(log_p: np.ndarray) -> float:
        k_val, th_val, xi_val = np.exp(log_p[0]), np.exp(log_p[1]), np.exp(log_p[2])
        val = -cir_log_likelihood(float(k_val), float(th_val), float(xi_val), arr, dt)
        return val

    res = optimize.minimize(
        fun=neg_ll_log,
        x0=x0_log,
        bounds=[(log_k_min, log_k_max), (log_th_min, log_th_max), (log_xi_min, log_xi_max)],
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )

    best_log_p = res.x
    best_nll = float(res.fun)
    converged = bool(res.success)

    # Compare against initial CLS likelihood and run simplex polish if needed
    cls_nll = neg_ll_log(x0_log)
    if not converged or best_nll > cls_nll:
        start = best_log_p if best_nll < cls_nll else x0_log
        res_nm = optimize.minimize(
            fun=lambda lp: (
                neg_ll_log(lp)
                if (log_k_min <= lp[0] <= log_k_max and log_th_min <= lp[1] <= log_th_max and log_xi_min <= lp[2] <= log_xi_max)
                else 1e15
            ),
            x0=start,
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-8, "maxiter": 800},
        )
        if float(res_nm.fun) < best_nll:
            best_log_p = res_nm.x
            best_nll = float(res_nm.fun)
            converged = bool(res_nm.success)

    kappa_hat, theta_hat, xi_hat = (
        float(np.exp(best_log_p[0])),
        float(np.exp(best_log_p[1])),
        float(np.exp(best_log_p[2])),
    )
    params = CIRParams(kappa=kappa_hat, theta=theta_hat, xi=xi_hat)
    ll = -best_nll
    k_num = 3
    aic = 2.0 * k_num - 2.0 * ll
    bic = k_num * np.log(n) - 2.0 * ll
    se = _numerical_hessian_se_cir(kappa_hat, theta_hat, xi_hat, arr, dt) if compute_se else {
        "kappa": np.nan, "theta": np.nan, "xi": np.nan
    }

    return CIRCalibrationResult(
        params=params,
        method="Exact Bessel MLE",
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        feller_ratio=params.feller_ratio,
        feller_satisfied=params.feller_satisfied,
        std_errors=se,
        converged=converged,
        n_obs=n,
    )
