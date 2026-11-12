"""
models.py
---------
Continuous-time stochastic differential equation (SDE) formulations, analytical
transition/stationary densities, and exact vs. discretized numerical simulation
schemes for the Ornstein-Uhlenbeck (OU) and Cox-Ingersoll-Ross (CIR) processes.

Mathematical Formulations:
1. Ornstein-Uhlenbeck (OU) Process (Uhlenbeck & Ornstein, 1930; Vasicek, 1977):
       dX_t = theta * (mu - X_t) dt + sigma dW_t,   theta > 0, sigma > 0
   Exact solution via integrating factor exp(theta * t) and Ito's Lemma:
       X_{t+dt} | X_t ~ N( mu + (X_t - mu) exp(-theta * dt),
                           (sigma^2 / (2 * theta)) * (1 - exp(-2 * theta * dt)) )

2. Cox-Ingersoll-Ross (CIR) Square-Root Diffusion (Cox, Ingersoll & Ross, 1985):
       dr_t = kappa * (theta - r_t) dt + xi * sqrt(r_t) dW_t,   kappa, theta, xi > 0
   Feller boundary non-attainment condition:
       2 * kappa * theta > xi^2   <=>   P(r_t > 0 for all t >= 0) = 1.
   Exact transition law via non-central chi-squared distribution:
       2 * c * r_{t+dt} | r_t ~ ncx2(df = d, nc = lambda_t),
   where c = 2 * kappa / (xi^2 * (1 - exp(-kappa * dt))),
         d = 4 * kappa * theta / xi^2,
         lambda_t = 2 * c * r_t * exp(-kappa * dt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy import stats, special


@dataclass(frozen=True)
class OUParams:
    """Parameters of the Ornstein-Uhlenbeck SDE dX_t = theta*(mu - X_t)dt + sigma*dW_t."""

    theta: float  # Mean-reversion speed (> 0, annualized)
    mu: float     # Long-run equilibrium level
    sigma: float  # Diffusion volatility (> 0, annualized)

    def __post_init__(self) -> None:
        if self.theta <= 0:
            raise ValueError(f"OU mean-reversion speed theta must be > 0, got {self.theta}.")
        if self.sigma <= 0:
            raise ValueError(f"OU volatility sigma must be > 0, got {self.sigma}.")

    @property
    def sigma_eq(self) -> float:
        """Asymptotic stationary standard deviation sigma / sqrt(2 * theta)."""
        return float(self.sigma / np.sqrt(2.0 * self.theta))

    @property
    def var_eq(self) -> float:
        """Asymptotic stationary variance sigma^2 / (2 * theta)."""
        return float((self.sigma ** 2) / (2.0 * self.theta))

    @property
    def half_life(self) -> float:
        """Expected half-life of mean reversion ln(2) / theta (in years if annualized)."""
        return float(np.log(2.0) / self.theta)


@dataclass(frozen=True)
class CIRParams:
    """Parameters of the Cox-Ingersoll-Ross SDE dr_t = kappa*(theta - r_t)dt + xi*sqrt(r_t)dW_t."""

    kappa: float  # Mean-reversion speed (> 0, annualized)
    theta: float  # Long-run equilibrium rate (> 0)
    xi: float     # Vol-of-vol / square-root diffusion coefficient (> 0, annualized)

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError(f"CIR kappa must be > 0, got {self.kappa}.")
        if self.theta <= 0:
            raise ValueError(f"CIR long-run mean theta must be > 0, got {self.theta}.")
        if self.xi <= 0:
            raise ValueError(f"CIR diffusion xi must be > 0, got {self.xi}.")

    @property
    def feller_lhs(self) -> float:
        """Left-hand side of Feller condition: 2 * kappa * theta."""
        return float(2.0 * self.kappa * self.theta)

    @property
    def feller_rhs(self) -> float:
        """Right-hand side of Feller condition: xi^2."""
        return float(self.xi ** 2)

    @property
    def feller_ratio(self) -> float:
        """Dimensionless Feller ratio 2 * kappa * theta / xi^2 (> 1 guarantees strict positivity)."""
        return float(2.0 * self.kappa * self.theta / (self.xi ** 2))

    @property
    def feller_satisfied(self) -> bool:
        """True iff 2 * kappa * theta > xi^2."""
        return bool(self.feller_lhs > self.feller_rhs)

    @property
    def var_eq(self) -> float:
        """Asymptotic Gamma stationary variance theta * xi^2 / (2 * kappa)."""
        return float(self.theta * (self.xi ** 2) / (2.0 * self.kappa))

    @property
    def sigma_eq(self) -> float:
        """Asymptotic stationary standard deviation sqrt(theta * xi^2 / (2 * kappa))."""
        return float(np.sqrt(self.var_eq))

    @property
    def half_life(self) -> float:
        """Expected half-life of mean reversion ln(2) / kappa."""
        return float(np.log(2.0) / self.kappa)


# =============================================================================
# Ornstein-Uhlenbeck (OU) Analytical & Simulation Routines
# =============================================================================


def ou_analytical_moments(
    params: OUParams,
    x0: float | np.ndarray,
    t: float | np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """
    Compute exact conditional mean and variance of X_t | X_0 = x0 under the OU SDE.

    Using Ito's Lemma on Y_t = e^{theta t}(X_t - mu):
        E[X_t | X_0]   = mu + (x0 - mu) * e^{-theta * t}
        Var(X_t | X_0) = (sigma^2 / (2 * theta)) * (1 - e^{-2 * theta * t})
    """
    decay = np.exp(-params.theta * np.asarray(t))
    mean = params.mu + (np.asarray(x0) - params.mu) * decay
    var = params.var_eq * (1.0 - np.exp(-2.0 * params.theta * np.asarray(t)))
    return {
        "mean": mean,
        "var": var,
        "std": np.sqrt(np.maximum(var, 0.0)),
        "sigma_eq": params.sigma_eq,
        "half_life": params.half_life,
    }


def ou_transition_pdf(
    x_next: np.ndarray,
    x_curr: float | np.ndarray,
    dt: float,
    params: OUParams,
) -> np.ndarray:
    """Exact Gaussian transition density f(x_{t+dt} | x_t) for the OU process."""
    moments = ou_analytical_moments(params, x_curr, dt)
    return stats.norm.pdf(x_next, loc=moments["mean"], scale=moments["std"])


def ou_stationary_pdf(x: np.ndarray, params: OUParams) -> np.ndarray:
    """Exact Gaussian stationary density N(mu, sigma^2 / (2*theta)) as t -> infinity."""
    return stats.norm.pdf(x, loc=params.mu, scale=params.sigma_eq)


def simulate_ou_exact(
    params: OUParams,
    x0: float,
    T: float,
    dt: float,
    n_paths: int = 1,
    seed: Optional[int] = 42,
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate exact sample paths of the OU process on [0, T] at step dt using the
    closed-form Gaussian transition law:
        X_{n+1} = mu + (X_n - mu) e^{-theta dt} + sigma_dt * Z_n,   Z_n ~ N(0, 1).

    Parameters
    ----------
    params : OUParams
        OU model parameters (theta, mu, sigma).
    x0 : float
        Initial state X_0.
    T : float
        Time horizon in years.
    dt : float
        Time step size in years.
    n_paths : int
        Number of Monte Carlo sample trajectories.
    seed : Optional[int]
        Random seed for reproducibility (ignored if `normals` is provided).
    normals : Optional[np.ndarray]
        Pre-drawn standard normal matrix of shape (n_paths, n_steps) for coupled
        strong-error comparisons against Euler-Maruyama.

    Returns
    -------
    t_grid : np.ndarray, shape (n_steps + 1,)
    paths : np.ndarray, shape (n_paths, n_steps + 1)
    """
    n_steps = int(round(T / dt))
    t_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)

    if normals is None:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, n_steps))
    else:
        if normals.shape != (n_paths, n_steps):
            raise ValueError(f"Expected normals shape {(n_paths, n_steps)}, got {normals.shape}.")
        Z = normals

    phi = np.exp(-params.theta * dt)
    sigma_dt = params.sigma * np.sqrt((1.0 - np.exp(-2.0 * params.theta * dt)) / (2.0 * params.theta))

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = x0
    drift_const = params.mu * (1.0 - phi)

    for step in range(n_steps):
        paths[:, step + 1] = drift_const + phi * paths[:, step] + sigma_dt * Z[:, step]

    return t_grid, paths


def simulate_ou_euler(
    params: OUParams,
    x0: float,
    T: float,
    dt: float,
    n_paths: int = 1,
    seed: Optional[int] = 42,
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate sample paths of the OU process via Euler-Maruyama discretization:
        X_{n+1} = X_n + theta * (mu - X_n) * dt + sigma * sqrt(dt) * Z_n.

    Note: Because OU has additive noise (d_x sigma = 0), Milstein coincides with
    Euler-Maruyama, achieving strong convergence order O(dt) (rather than O(dt^{1/2})),
    but retains a finite-step stationary variance bias:
        Var_EM = sigma^2 / (2*theta - theta^2 * dt)  vs.  Var_exact = sigma^2 / (2*theta),
    and becomes numerically unstable if theta * dt >= 2.
    """
    n_steps = int(round(T / dt))
    t_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)

    if normals is None:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, n_steps))
    else:
        if normals.shape != (n_paths, n_steps):
            raise ValueError(f"Expected normals shape {(n_paths, n_steps)}, got {normals.shape}.")
        Z = normals

    sqrt_dt = np.sqrt(dt)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = x0

    for step in range(n_steps):
        paths[:, step + 1] = (
            paths[:, step]
            + params.theta * (params.mu - paths[:, step]) * dt
            + params.sigma * sqrt_dt * Z[:, step]
        )

    return t_grid, paths


# =============================================================================
# Cox-Ingersoll-Ross (CIR) Analytical & Simulation Routines
# =============================================================================


def cir_analytical_moments(
    params: CIRParams,
    r0: float | np.ndarray,
    t: float | np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """
    Compute exact conditional mean and variance of r_t | r_0 under the CIR SDE:
        E[r_t | r_0]   = theta + (r_0 - theta) * e^{-kappa * t}
        Var(r_t | r_0) = r_0 * (xi^2 / kappa) * (e^{-kappa * t} - e^{-2 * kappa * t})
                         + (theta * xi^2 / (2 * kappa)) * (1 - e^{-kappa * t})^2
    """
    exp_kt = np.exp(-params.kappa * np.asarray(t))
    r0_arr = np.asarray(r0)
    mean = params.theta + (r0_arr - params.theta) * exp_kt
    var = (
        r0_arr * ((params.xi ** 2) / params.kappa) * (exp_kt - exp_kt ** 2)
        + params.var_eq * ((1.0 - exp_kt) ** 2)
    )
    return {
        "mean": mean,
        "var": var,
        "std": np.sqrt(np.maximum(var, 0.0)),
        "sigma_eq": params.sigma_eq,
        "feller_ratio": params.feller_ratio,
        "feller_satisfied": params.feller_satisfied,
    }


def cir_transition_pdf(
    r_next: np.ndarray,
    r_curr: float,
    dt: float,
    params: CIRParams,
) -> np.ndarray:
    """
    Exact non-central chi-squared transition density f(r_{t+dt} | r_t) for CIR,
    evaluated stably via the exponentially scaled modified Bessel function I_q^e.
    """
    r_next_arr = np.asarray(r_next, dtype=np.float64)
    pdf = np.zeros_like(r_next_arr)
    pos = r_next_arr > 0.0
    if not np.any(pos) or r_curr <= 0.0:
        return pdf

    exp_k = np.exp(-params.kappa * dt)
    c = 2.0 * params.kappa / ((params.xi ** 2) * (1.0 - exp_k))
    q = 2.0 * params.kappa * params.theta / (params.xi ** 2) - 1.0

    u = c * r_curr * exp_k
    v = c * r_next_arr[pos]
    z = 2.0 * np.sqrt(u * v)
    bessel_scaled = special.ive(q, z)

    # Using log I_q(z) = z + log(ive(q, z)), the exponent -u - v + z = -(sqrt(u) - sqrt(v))^2
    log_pdf = (
        np.log(c)
        - (np.sqrt(u) - np.sqrt(v)) ** 2
        + 0.5 * q * np.log(v / u)
        + np.log(np.maximum(bessel_scaled, 1e-300))
    )
    pdf[pos] = np.exp(log_pdf)
    return pdf


def cir_stationary_pdf(r: np.ndarray, params: CIRParams) -> np.ndarray:
    """
    Exact stationary Gamma density of the CIR process as t -> infinity:
        r_inf ~ Gamma(shape = alpha, rate = beta)
    with alpha = 2 * kappa * theta / xi^2 and beta = 2 * kappa / xi^2.
    Note: When Feller is violated (alpha < 1), the density diverges at r -> 0^+.
    """
    alpha = 2.0 * params.kappa * params.theta / (params.xi ** 2)
    beta = 2.0 * params.kappa / (params.xi ** 2)
    return stats.gamma.pdf(r, a=alpha, scale=1.0 / beta)


def simulate_cir_exact(
    params: CIRParams,
    r0: float,
    T: float,
    dt: float,
    n_paths: int = 1,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact simulation of the CIR process via the non-central chi-squared transition law
    (Glasserman, 2004, Section 3.4):
        r_{n+1} = Y / (2 * c),   Y ~ chi'^2_d(lambda_n),
    where:
        c        = 2 * kappa / (xi^2 * (1 - e^{-kappa * dt}))
        d        = 4 * kappa * theta / xi^2
        lambda_n = 2 * c * r_n * e^{-kappa * dt}.

    For d > 1, Glasserman's representation Y = (Z + sqrt(lambda_n))^2 + chi^2_{d-1}
    is O(1) per sample; numpy's `Generator.noncentral_chisquare(df, nonc)` handles
    all d > 0 (including Feller-violated 0 < d <= 2 via Poisson-mixture of central
    chi-squares when nonc > 0, and central gamma when nonc == 0).
    """
    if r0 < 0.0:
        raise ValueError(f"Initial rate r0 must be >= 0, got {r0}.")

    n_steps = int(round(T / dt))
    t_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)
    rng = np.random.default_rng(seed)

    exp_k = np.exp(-params.kappa * dt)
    c = 2.0 * params.kappa / ((params.xi ** 2) * (1.0 - exp_k))
    df = 4.0 * params.kappa * params.theta / (params.xi ** 2)
    inv_2c = 1.0 / (2.0 * c)
    nc_factor = 2.0 * c * exp_k

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = r0

    for step in range(n_steps):
        lam = nc_factor * paths[:, step]
        # Guardlam == 0 (if path touched 0 in Feller-violated limit): reduces to central chi-square
        pos_mask = lam > 0.0
        y = np.empty(n_paths, dtype=np.float64)
        if np.any(pos_mask):
            y[pos_mask] = rng.noncentral_chisquare(df=df, nonc=lam[pos_mask])
        if np.any(~pos_mask):
            y[~pos_mask] = rng.chisquare(df=df, size=int(np.sum(~pos_mask)))
        paths[:, step + 1] = inv_2c * y

    return t_grid, paths


CIRScheme = Literal["full_truncation", "reflection", "absorption", "partial_truncation", "milstein_truncated"]


def simulate_cir_euler(
    params: CIRParams,
    r0: float,
    T: float,
    dt: float,
    n_paths: int = 1,
    scheme: CIRScheme = "full_truncation",
    seed: Optional[int] = 42,
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretized simulation of the CIR square-root process with explicit boundary-handling
    schemes at r = 0 (Lord, Koekkoek & Van Dijk, 2010; Higham & Mao, 2005):

    - 'full_truncation' (Lord et al., 2010):
        tilde_r_{n+1} = tilde_r_n + kappa * (theta - tilde_r_n^+) * dt + xi * sqrt(tilde_r_n^+) * dW_n
        r_{n+1}       = tilde_r_{n+1}^+
      Achieves the lowest weak convergence bias among Euler-type schemes.

    - 'reflection' (Berkaoui, Bossy & Diop, 2008):
        r_{n+1} = | r_n + kappa * (theta - r_n) * dt + xi * sqrt(r_n) * dW_n |

    - 'absorption':
        r_{n+1} = max(0, r_n + kappa * (theta - r_n) * dt + xi * sqrt(r_n) * dW_n)

    - 'partial_truncation' (Deelstra & Delbaen, 1998):
        tilde_r_{n+1} = tilde_r_n + kappa * (theta - tilde_r_n) * dt + xi * sqrt(tilde_r_n^+) * dW_n
        r_{n+1}       = tilde_r_{n+1}^+

    - 'milstein_truncated' (Kahl & Jackel, 2006 / Higham & Mao, 2005):
        r_{n+1} = max(0, r_n + kappa*(theta - r_n)*dt + xi*sqrt(r_n)*dW_n + 0.25*xi^2*(dW_n^2 - dt))
    """
    if r0 < 0.0:
        raise ValueError(f"Initial rate r0 must be >= 0, got {r0}.")

    n_steps = int(round(T / dt))
    t_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)

    if normals is None:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, n_steps))
    else:
        if normals.shape != (n_paths, n_steps):
            raise ValueError(f"Expected normals shape {(n_paths, n_steps)}, got {normals.shape}.")
        Z = normals

    sqrt_dt = np.sqrt(dt)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = r0

    if scheme in ("full_truncation", "partial_truncation"):
        tilde_r = np.full(n_paths, r0, dtype=np.float64)
        for step in range(n_steps):
            r_pos = np.maximum(tilde_r, 0.0)
            drift_state = r_pos if scheme == "full_truncation" else tilde_r
            tilde_r = (
                tilde_r
                + params.kappa * (params.theta - drift_state) * dt
                + params.xi * np.sqrt(r_pos) * sqrt_dt * Z[:, step]
            )
            paths[:, step + 1] = np.maximum(tilde_r, 0.0)
    elif scheme == "reflection":
        for step in range(n_steps):
            r_curr = paths[:, step]
            paths[:, step + 1] = np.abs(
                r_curr
                + params.kappa * (params.theta - r_curr) * dt
                + params.xi * np.sqrt(r_curr) * sqrt_dt * Z[:, step]
            )
    elif scheme == "absorption":
        for step in range(n_steps):
            r_curr = paths[:, step]
            paths[:, step + 1] = np.maximum(
                0.0,
                r_curr
                + params.kappa * (params.theta - r_curr) * dt
                + params.xi * np.sqrt(r_curr) * sqrt_dt * Z[:, step],
            )
    elif scheme == "milstein_truncated":
        for step in range(n_steps):
            r_curr = paths[:, step]
            dW = sqrt_dt * Z[:, step]
            paths[:, step + 1] = np.maximum(
                0.0,
                r_curr
                + params.kappa * (params.theta - r_curr) * dt
                + params.xi * np.sqrt(r_curr) * dW
                + 0.25 * (params.xi ** 2) * (dW ** 2 - dt),
            )
    else:
        raise ValueError(f"Unknown CIR discretization scheme: {scheme!r}.")

    return t_grid, paths
