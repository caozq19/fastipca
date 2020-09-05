import logging

import numpy as np
import opt_einsum as oe
import pandas as pd
from scipy.sparse.linalg import svds

logging.basicConfig(
    format="%(levelname)s: %(name)s - %(asctime)s.%(msecs)03d %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

_logger = logging.getLogger(__name__)


def train(
    Z,
    R,
    n_factors=1,
    exog_factors=None,
    intercept=False,
    max_iter=1000,
    tol=1e-6,
    verbose=False,
):
    """Implements the IPCA algorithm by Kelly, Pruitt, Su (2017).

    Parameters
    ----------
    Z : pandas.DataFrame
        Panel of characteristics. Columns are individual characteristics.
        Index must be a multi-index with time as the first component
        and symbols as the second component.
        The time component must be sorted.

    R : pandas.Series
        Panel of returns to be explained. Index must agrees with Z index.

    n_factors : integer
        number of factors to calibrate possibly 0.

    exog_factors : pandas.DataFrame
        Matrix of pre-specified factors. Index must coincide with the first
        level of Z index..

    intercept: bool
        If true, a constant pre-specified factor equals to one is appended.

    max_iter: integer
        Maximum number of iterations to perform.

    tol : float
        MAE tolerance between iterations.

    verbose : bool
        If True, displays convergence info each iteration.

    Returns
    -------
    gamma : pandas.DataFrame
        Characteristics loadings.

    factors : pandas.DataFrame
        Factor estimates.

    Note
    ----
    The factor must be positive.
    """
    if verbose:
        _logger.setLevel(logging.INFO)
    else:
        _logger.setLevel(logging.WARN)

    assert n_factors >= 0
    assert max_iter >= 0

    assert Z.index.equals(R.index)
    assert Z.index.get_level_values(0).is_monotonic_increasing

    _logger.info("compute interactions and covariances")

    ix = Z.index.remove_unused_levels()
    r = 1 + np.nonzero(np.diff(ix.codes[0]))[0]
    r = np.array([0, *r.tolist(), ix.shape[0]])

    nobs = r[1:] - r[:-1]

    z = Z.values
    zr = z * R.values[:, None]
    Q = np.stack([zr[i:j].mean(0) for i, j in zip(r, r[1:])])

    W = np.stack([z[i:j].T @ z[i:j] / (j - i) for i, j in zip(r, r[1:])])

    _logger.info("initialize gamma, factors")

    if exog_factors is None:
        exog_factors_names = []
        exog_factors = np.empty((ix.levels[0].shape[0], 0))
    else:
        assert exog_factors.index.equals(ix.levels[0])
        exog_factors_names = exog_factors.columns.tolist()
        exog_factors = exog_factors.values

    factor_names = list(range(n_factors)) + exog_factors_names
    if intercept:
        factor_names.append("intercept")

    n_all = len(factor_names)
    active = slice(0, n_factors)
    specified = slice(n_factors, n_all)

    gamma = np.zeros((Z.shape[1], n_all))
    factors = np.zeros((exog_factors.shape[0], n_all))

    f_on, f_off = factors[:, active], factors[:, specified]
    g_on, g_off = gamma[:, active], gamma[:, specified]

    if intercept:
        f_off[:, :-1] = exog_factors
        f_off[:, -1] = 1.0
    else:
        f_off[:] = exog_factors

    if n_factors > 0:
        if n_factors == Z.shape[1]:
            f, s, g = np.linalg.svd(Q, full_matrices=False)
        else:
            f, s, g = svds(Q, k=n_factors)
        o = np.argsort(s)[::-1]
        g_on[:] = g[o].T
        f_on[:] = f[:, o] * s[o]

    for ite in range(max_iter):
        factors_old, gamma_old = f_on.copy(), gamma.copy()

        # factors step
        if n_factors > 0:
            m1 = oe.contract("lk,tlm,mn->tkn", g_on, W, g_on)
            m2 = Q @ g_on
            if n_factors != n_all:
                m2 -= oe.contract("lk,tlm,mn,tn->tk", g_on, W, g_off, f_off)
            f_on[:] = np.linalg.solve(m1, m2)

        # gamma step
        numer = oe.contract("tl,tf,t->lf", Q, factors, nobs).reshape(-1)
        denom = oe.contract("tij,tk,tl,t->ikjl", W, factors, factors, nobs)
        denom = denom.reshape((gamma.size, gamma.size))
        gamma[:] = np.linalg.solve(denom, numer).reshape(gamma.shape)

        # identification
        if n_factors > 0:
            # make gamma and factors orthogonal
            R1 = np.linalg.cholesky(g_on.T @ g_on)
            R2, _, _ = np.linalg.svd(R1 @ f_on.T @ f_on @ R1.T)
            f_on[:] = np.linalg.solve(R2, R1 @ f_on.T).T
            g_on[:] = np.linalg.lstsq(g_on.T, R1.T, rcond=None)[0] @ R2

            # make g_off and g_on orthogonal
            if n_factors != n_all:
                g_off[:] -= g_on @ g_on.T @ g_off
                f_on[:] += f_off @ g_off.T @ g_on

            # factors should have a positive mean
            sgn = np.sign(f_on.mean(0))
            sgn[sgn == 0] = 1
            f_on[:] *= sgn
            g_on[:] *= sgn

        # exit condition
        tol_f, tol_g = -np.inf, -np.inf
        if n_all > 0:
            tol_g = np.abs(gamma - gamma_old).max()
            if n_factors > 0:
                tol_f = np.abs(f_on - factors_old).max()

        _logger.info(f"iter={ite} tol_g={tol_g:.4} tol_f={tol_f:.4}")
        if max(tol_g, tol_f) < tol:
            break
    else:
        _logger.warning("ipca did not converge")

    gamma = pd.DataFrame(gamma, index=Z.columns, columns=factor_names)
    factors = pd.DataFrame(factors, index=ix.levels[0], columns=factor_names)
    return gamma, factors


def predict(Z, gamma, factors):
    """Implements the IPCA algorithm by Kelly, Pruitt, Su (2017).

    Parameters
    ----------
    Z : pandas.DataFrame
        Panel of characteristics. Columns are individual characteristics.
        Index must be a multi-index with time as the first component
        and symbols as the second component.
        The time component must be sorted.

    gamma : pandas.DataFrame
        Characteristics loadings.

    factors : pandas.DataFrame
        Factor estimates.

    Returns
    -------
    pandas.Series
        Reconstructed values.

    """
    rhat = Z.values @ gamma.values
    if factors.ndim == 1:
        rhat *= factors.values
    else:
        assert Z.index.levels[0].equals(factors.index)
        rhat *= factors.values[Z.index.codes[0]]

    rhat = rhat.sum(1)
    return pd.Series(rhat, index=Z.index)
