"""
Carver handcrafting - instrument volatility weights  (v2, fixed)
================================================================

Implements Carver (2018) handcrafting + Carver (2020) correlation-uncertainty
weights, for already volatility-normalised futures subsystems.

Input : WEEKLY subsystem percentage returns (one column per instrument).
Output: instrument volatility weights (sum to 1) and IDM.
No Sharpe-ratio adjustment (equal expected SR assumed).

Fixes vs v1
-----------
1. ROOT may hold more than 3 children (e.g. 4 asset classes). They get equal
   weights (ROOT_EQUAL_WEIGHTS=True), or the 3-asset method if ROOT has 3.
2. Diversification multiplier is now the DM of the *actual leaf weights*:
       true_dm = parent_dm * sum(group_weight_i * child_dm_i)
   (v1 lost the children's internal diversification when renormalising, so
   diversified branches were under-weighted and the IDM was understated).
   Final IDM is also checked directly as 1/sqrt(w'Hw) on instrument correlations.
3. No more "drop any week with a missing instrument". Correlations are pairwise
   and node returns are built from whatever members are live that week, so a
   young instrument (e.g. TWN from 2020) no longer truncates its whole branch.
   n_obs passed to the Fisher step = smallest pairwise overlap in the node.
4. A failed SLSQP run falls back to equal weights with a warning instead of
   aborting the whole 729-optimisation loop.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


# ================================================================ CONFIG
P_STEP = 0.10                 # distribution points 0.1 ... 0.9 -> 9^3 = 729 runs
FISHER_FUDGE = 4.0            # Carver 2020: widen sampling error x4
TWO_SIDED_CONF = True         # v1 convention: z = ppf(1 - ci/2). False -> z = ppf(ci)
MIN_RAW_WEIGHT = 0.10         # floor before renormalisation
IDM_CAP = 2.5
ROOT_EQUAL_WEIGHTS = True     # equal weights at ROOT when it has > 3 children
MIN_OVERLAP = 52              # minimum weeks of pairwise overlap required


# ================================================================ TREE
# Temporary: only the two instruments with data so far. Equal split with 2 leaves.
TREE = ["AD1_small", "RX1_small"]


# ================================================================ TREE UTILITIES
def validate_tree(node, path="ROOT", is_root=True):
    if isinstance(node, list):
        if len(node) > 3:
            raise ValueError(f"{path} contains {len(node)} instruments (max 3).")
        return
    if not isinstance(node, dict):
        raise TypeError(f"{path}: expected dict or list.")
    if len(node) > 3 and not (is_root and ROOT_EQUAL_WEIGHTS):
        raise ValueError(f"{path} contains {len(node)} sub-portfolios (max 3).")
    for name, child in node.items():
        validate_tree(child, f"{path}/{name}", is_root=False)


def get_leaves(node):
    if isinstance(node, list):
        return list(node)
    out = []
    for child in node.values():
        out.extend(get_leaves(child))
    return out


# ================================================================ PORTFOLIO MATHS
def nearest_correlation_matrix(matrix):
    m = np.asarray(matrix, dtype=float)
    vals, vecs = np.linalg.eigh((m + m.T) / 2)
    psd = vecs @ np.diag(np.maximum(vals, 1e-10)) @ vecs.T
    d = np.sqrt(np.diag(psd))
    c = psd / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return c


def diversification_multiplier(weights, corr):
    w = np.asarray(weights, dtype=float)
    var = float(w @ corr @ w)
    return 1.0 if var <= 0 else 1.0 / np.sqrt(var)


def pairwise_corr_and_nobs(df):
    """Pairwise-complete correlation matrix and the smallest pairwise overlap."""
    present = df.notna().astype(int)
    overlap = present.T @ present                       # pairwise overlap counts
    n_obs = int(overlap.values.min())
    if n_obs < MIN_OVERLAP:
        raise ValueError(f"Only {n_obs} overlapping weeks between {list(df.columns)}"
                         f" (MIN_OVERLAP={MIN_OVERLAP}).")
    corr = df.corr(min_periods=MIN_OVERLAP).values
    return nearest_correlation_matrix(corr), n_obs


def weighted_available(df, weights):
    """Row-wise weighted sum over the columns available that week (weights renormalised)."""
    w = pd.Series(weights, index=df.columns, dtype=float)
    mask = df.notna()
    wsum = mask.mul(w, axis=1).sum(axis=1)
    out = df.fillna(0.0).mul(w, axis=1).sum(axis=1) / wsum.replace(0, np.nan)
    return out


# ================================================================ OPTIMISATION
def optimise_equal_sr_equal_vol(corr):
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]

    def neg_sr(w):
        v = w @ corr @ w
        return 1e10 if v <= 0 else -w.sum() / np.sqrt(v)

    res = minimize(neg_sr, np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   tol=1e-10)
    if not res.success:
        warnings.warn(f"SLSQP failed ({res.message}); using equal weights for this scenario.")
        return np.ones(n) / n
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


# ================================================================ CARVER 2020 FISHER
def _z(conf):
    return norm.ppf(1.0 - conf / 2.0) if TWO_SIDED_CONF else abs(norm.ppf(conf))


def correlation_distribution_point(corr, n_obs, conf):
    f = np.arctanh(np.clip(corr, -0.999999, 0.999999))
    se = FISHER_FUDGE / np.sqrt(n_obs - 3)
    if conf < 0.5:
        f = f - se * _z(conf)
    elif conf > 0.5:
        f = f + se * _z(1.0 - conf)
    return np.tanh(f)


def apply_min_weight(w):
    w = np.maximum(np.asarray(w, dtype=float), MIN_RAW_WEIGHT)
    return w / w.sum()


def carver_three_asset_weights(corr, n_obs):
    points = np.arange(P_STEP, 1.0 - P_STEP + 1e-12, P_STEP)
    ab, ac, bc = corr[0, 1], corr[0, 2], corr[1, 2]
    ws = []
    for p_ab, p_ac, p_bc in itertools.product(points, repeat=3):
        c = np.array([[1.0, correlation_distribution_point(ab, n_obs, p_ab),
                       correlation_distribution_point(ac, n_obs, p_ac)],
                      [0.0, 1.0, correlation_distribution_point(bc, n_obs, p_bc)],
                      [0.0, 0.0, 1.0]])
        c = c + np.triu(c, 1).T
        ws.append(optimise_equal_sr_equal_vol(nearest_correlation_matrix(c)))
    return apply_min_weight(np.mean(ws, axis=0))


def node_weights(corr, n_obs, is_root=False):
    n = len(corr)
    if n == 1:
        return np.array([1.0])
    if n == 2:
        return np.array([0.5, 0.5])
    if n == 3:
        return carver_three_asset_weights(corr, n_obs)
    if is_root and ROOT_EQUAL_WEIGHTS:      # FIX 1: e.g. 4 asset classes
        return np.ones(n) / n
    raise ValueError("A handcrafted node must contain <= 3 children.")


# ================================================================ RECURSION
class HandcraftedPortfolio:
    def __init__(self, weights, returns, dm, diagnostics=None):
        self.weights = weights          # leaf weights inside this node (sum to 1)
        self.returns = returns          # unit-vol return series of this node
        self.dm = dm                    # DM of the leaf weights (true, see fix 2)
        self.diagnostics = diagnostics or {}


def build_portfolio(node, rets, path="ROOT", is_root=True, diag_out=None):
    diag_out = diag_out if diag_out is not None else []

    if isinstance(node, list):
        names = list(node)
        data = rets[names].replace([np.inf, -np.inf], np.nan)
        if len(names) == 1:
            corr, n_obs = np.array([[1.0]]), int(data.notna().sum().iloc[0])
        else:
            corr, n_obs = pairwise_corr_and_nobs(data)
        w = node_weights(corr, n_obs)
        dm = diversification_multiplier(w, corr)
        node_ret = weighted_available(data, w) * dm
        leaf_w = pd.Series(w, index=names, dtype=float)
        diag_out.append({"path": path, "children": names, "n_obs": n_obs,
                         "weights": np.round(w, 4), "dm": round(dm, 4)})
        return HandcraftedPortfolio(leaf_w, node_ret, dm)

    child_names = list(node)
    children = {n: build_portfolio(sub, rets, f"{path}/{n}", False, diag_out)
                for n, sub in node.items()}

    child_rets = pd.concat({n: c.returns for n, c in children.items()}, axis=1)
    if len(child_names) == 1:
        corr, n_obs = np.array([[1.0]]), int(child_rets.notna().sum().iloc[0])
    else:
        corr, n_obs = pairwise_corr_and_nobs(child_rets)

    gw = node_weights(corr, n_obs, is_root=is_root)
    child_dm = np.array([children[n].dm for n in child_names])

    # leaf weights: child internal weight x group weight x child DM, renormalised
    contrib = pd.concat([children[n].weights * gw[i] * child_dm[i]
                         for i, n in enumerate(child_names)])
    k = float(contrib.sum())                      # = sum(gw_i * dm_i)
    leaf_w = contrib / k

    parent_dm = diversification_multiplier(gw, corr)
    true_dm = parent_dm * k                       # FIX 2: DM of the actual leaf weights
    node_ret = weighted_available(child_rets, gw) * parent_dm   # ~unit vol

    diag_out.append({"path": path, "children": child_names, "n_obs": n_obs,
                     "weights": np.round(gw, 4), "child_dm": np.round(child_dm, 4),
                     "dm": round(true_dm, 4)})
    return HandcraftedPortfolio(leaf_w, node_ret, true_dm)


# ================================================================ MAIN
def handcrafted_instrument_weights(weekly_subsystem_returns, tree=TREE):
    validate_tree(tree)
    missing = set(get_leaves(tree)) - set(weekly_subsystem_returns.columns)
    if missing:
        raise ValueError("Missing instruments: " + ", ".join(sorted(missing)))

    diags = []
    port = build_portfolio(tree, weekly_subsystem_returns, diag_out=diags)
    w = port.weights / port.weights.sum()

    # independent check: IDM from instrument correlations, 1/sqrt(w'Hw)
    H, _ = pairwise_corr_and_nobs(weekly_subsystem_returns[w.index])
    idm_direct = diversification_multiplier(w.values, H)

    result = {
        "weights": w.sort_values(ascending=False),
        "idm": min(idm_direct, IDM_CAP),
        "idm_uncapped": idm_direct,
        "idm_tree": port.dm,                 # should be close to idm_direct
        "diagnostics": pd.DataFrame(diags),
    }
    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "weekly_subsystem_returns.csv"
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    res = handcrafted_instrument_weights(returns)
    out = pd.DataFrame({"Instrument": res["weights"].index,
                        "Weight %": (res["weights"].values * 100).round(2)})
    print(out.to_string(index=False))
    print(f"\nWeight total: {res['weights'].sum():.6f}")
    print(f"IDM (direct, capped {IDM_CAP}): {res['idm']:.3f}   uncapped: {res['idm_uncapped']:.3f}"
          f"   tree: {res['idm_tree']:.3f}")
    out.to_csv("handcrafted_instrument_weights.csv", index=False)
    res["diagnostics"].to_csv("handcrafting_diagnostics.csv", index=False)