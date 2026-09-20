"""Old-vs-new diversification-multiplier comparison (raw correlation vs negatives floored at 0).

Runs the real strategy code (strategies/ewma.py, strategies/carry.py) and the real position-sizing
simulation (steps/p5_framework_one_function.py::framework_main) for every instrument in
DATA/input_instruments, once with the OLD multiplier maths (raw correlation matrix) and once with
the NEW maths (steps/multiplier_utils.py - negative off-diagonals floored at zero). Everything is
written to a temp folder; nothing under DATA/ is touched.

The OLD maths is re-implemented inline below on purpose, so this script keeps working (and keeps
showing a genuine before/after) after production code has switched to the new maths.

Inputs are the production ones: instrument weights from DATA/input_main/input_main.csv (or, if that
file no longer covers the active instrument pool, the newest Settings-page checkpoint that does -
the source is printed), standard costs from p_pages.main_analysis_page._calc_standard_cost (input-file
BID/ASK/cost columns, else the input file's own 'Standard Cost' column - source printed), and the
production strategy weights (EWMA speeds 1/n, Carry vs EWMA_combined 1/2, Strategy Analysis default
family weights). Old and new runs use identical data and dates (asserted).

    python tools/compare_multiplier_methods.py [--weights AD1_small=0.5,RX1_small=0.5] [--out DIR]
        [--cost-source DIR]   # folder holding <instrument>.csv with BID/ASK/EXECUTION_COST/CLEARING_COST
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import streamlit as st  # noqa: E402


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Bar:
    def progress(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def success(self, *a, **k):
        pass


for _name in ("write", "header", "subheader", "dataframe", "info", "warning", "success", "error",
              "caption", "json", "markdown", "text", "pyplot", "table", "title"):
    setattr(st, _name, lambda *a, **k: None)
st.warning = lambda *a, **k: print("  [streamlit warning]", *a)  # surface, do not swallow
st.expander = lambda *a, **k: _Ctx()
st.progress = lambda *a, **k: _Bar()
st.empty = lambda *a, **k: _Bar()

SCRATCH = tempfile.mkdtemp(prefix="multiplier_compare_")
for _d in ("output_instruments", "output_plots"):
    os.makedirs(os.path.join(SCRATCH, _d), exist_ok=True)
_orig_join = os.path.join


def _patched_join(*args):
    if len(args) > 1 and args[0] == "DATA" and args[1] in ("output_instruments", "output_plots"):
        return _orig_join(SCRATCH, *args[1:])
    return _orig_join(*args)


os.path.join = _patched_join

import strategies.carry as carry  # noqa: E402
import strategies.ewma as ewma  # noqa: E402
from p_pages.main_analysis_page import _calc_standard_cost  # noqa: E402
from p_pages.sharpe_ratio_page import _isolated_instrument_sim, _metrics_from_returns  # noqa: E402
from steps.multiplier_utils import diversification_multiplier, validate_multiplier_correlation  # noqa: E402
from steps.volatility import simple_price_volatility  # noqa: E402
import steps.p1_analysis as p1_analysis  # noqa: E402
from steps.p5_framework_one_function import framework_main  # noqa: E402

TRADING_DAYS = 256
CAP = 20.0
EWMA_FDM_CAP = 2.0
COMBINED_FDM_CAP = 2.5
PDM_CAP = 2.0
AUM = 10_000_000
SPEEDS = [2, 4, 8, 16, 32, 64]


def old_multiplier(raw_corr, weights, cap, fill_nan=False):
    """The pre-change maths, exactly as the production sites had it: raw matrix, no flooring."""
    c = raw_corr.fillna(0.0).to_numpy() if fill_nan else raw_corr.to_numpy()
    w = np.asarray(weights, dtype=float)
    wcw = float(np.dot(w.T, np.dot(c, w)))
    if fill_nan:
        return min(1.0 / np.sqrt(wcw), cap) if wcw > 0 else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return min(1.0 / np.sqrt(wcw), cap)


def new_multiplier(raw_corr, weights, cap, fill_nan=False):
    m = raw_corr.fillna(0.0) if fill_nan else raw_corr
    return diversification_multiplier(m, weights, cap)


def _weights_from(path, insts):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if not {"INSTRUMENT", "INSTRUMENT_WEIGHTS"} <= set(df.columns):
        return None
    m = dict(zip(df["INSTRUMENT"], df["INSTRUMENT_WEIGHTS"]))
    return {i: float(m[i]) for i in insts} if all(i in m and pd.notna(m[i]) for i in insts) else None


def resolve_instrument_weights(insts, override):
    if override:
        return {k: float(v) for k, v in (kv.split("=") for kv in override.split(","))}, "--weights override"
    live = _weights_from(os.path.join("DATA", "input_main", "input_main.csv"), insts)
    if live:
        return live, "DATA/input_main/input_main.csv"
    ckpt_root = os.path.join("DATA", "checkpoints")
    for name in (sorted(os.listdir(ckpt_root), reverse=True) if os.path.isdir(ckpt_root) else []):
        found = _weights_from(os.path.join(ckpt_root, name, "input_main", "input_main.csv"), insts)
        if found:
            return found, (f"newest Settings checkpoint covering these instruments (DATA/checkpoints/{name}); "
                           "input_main.csv no longer lists them")
    raise SystemExit("No production weights found for " + ", ".join(insts) + " - pass --weights")


def standard_cost_source(df):
    needed = ["BID", "ASK", "EXECUTION_COST", "CLEARING_COST"]
    if set(needed) <= set(df.columns) and df[needed].iloc[0].notna().all():
        return "computed from the input file's BID/ASK/EXECUTION_COST/CLEARING_COST"
    return "input file's own 'Standard Cost' column (BID/ASK/EXECUTION_COST/CLEARING_COST not present)"


def constraint_row(label, matrix):
    a = np.asarray(matrix, dtype=float)
    validate_multiplier_correlation(a)  # raises if unit diagonal / non-negative / PSD fails
    return {"Matrix": label, "Size": a.shape[0], "Diagonal all 1": bool(np.allclose(np.diag(a), 1.0)),
            "Min entry": a.min(), "Min eigenvalue": np.linalg.eigvalsh((a + a.T) / 2).min()}


def prepare_strategy_input(inst):
    df = pd.read_csv(os.path.join("DATA", "input_instruments", f"{inst}.csv"))
    df["st_dev"] = simple_price_volatility(df["PX_CLOSE_1D"])  # calculated in code, as Strategy Analysis does
    return df


def expanding_combined(frames, weights, cap, use_new):
    """steps/p2_validation.py::forecast() applied on an expanding window, day by day."""
    first = frames[0]
    start = first["Date"].iloc[1]
    rows = []
    for day in first.loc[first["Date"] >= start, "Date"]:
        sub = [f[f["Date"] <= day] for f in frames]
        raw = pd.DataFrame([s["forecast_pct_return"].values for s in sub]).T.corr()
        m = new_multiplier(raw, weights, cap).multiplier if use_new else old_multiplier(raw, weights, cap)
        last = [s["capped_forecast"].iloc[-1] for s in sub]
        rows.append((day, float(np.clip(m * np.dot(weights, last), -CAP, CAP)), m, raw.iloc[0, 1]))
    return pd.DataFrame(rows, columns=["Date", "FinalForecast", "Multiplier", "RawCorr"])


def forecast_stats(ff):
    ff = ff.dropna()
    return {
        "Avg |combined forecast|": ff.abs().mean(),
        "Max |combined forecast|": ff.abs().max(),
        "% days at +/-20": 100.0 * (ff.abs() >= CAP - 1e-9).mean(),
    }


def fmt(df):
    return df.to_string(float_format=lambda x: f"{x:,.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="e.g. AD1_small=0.5,RX1_small=0.5 (default: equal)")
    ap.add_argument("--out", default="", help="optional folder to save CSV copies of every table")
    ap.add_argument("--cost-source", default="", help="folder of <instrument>.csv files to compute the standard "
                    "cost from (default: DATA/input_instruments)")
    args = ap.parse_args()

    insts = sorted(f[:-4] for f in os.listdir(os.path.join("DATA", "input_instruments")) if f.endswith(".csv"))
    weights_map, weights_source = resolve_instrument_weights(insts, args.weights)
    audit_rows = []
    cost_sources = {}

    tables = {}
    per_inst = {}
    old_cf_dir = os.path.join(SCRATCH, "cf_old")
    new_cf_dir = os.path.join(SCRATCH, "cf_new")
    os.makedirs(old_cf_dir)
    os.makedirs(new_cf_dir)
    fm = {}
    sim_turnover = {"old": {}, "new": {}}
    sim_costs = {}

    ewma_rows, comb_rows, strat_rows, inst_rows = [], [], [], []
    date_ranges = {}
    corr_dumps = []

    for inst in insts:
        raw_input = pd.read_csv(os.path.join("DATA", "input_instruments", f"{inst}.csv"))
        prepped = prepare_strategy_input(inst)
        cost_df, cost_origin = prepped, "DATA/input_instruments"
        if args.cost_source:
            cost_df = pd.read_csv(os.path.join(args.cost_source, f"{inst}.csv"))
            cost_df.columns = [c.strip() for c in cost_df.columns]
            cost_origin = args.cost_source
        sc = float(_calc_standard_cost(cost_df, cost_df["POINT_VALUE"].iloc[0]))
        cost_sources[inst] = f"{sc:.6f} <- {standard_cost_source(cost_df)} [{cost_origin}]"
        fx = float(prepped["Exchange rate"].iloc[0])
        pv = float(prepped["POINT_VALUE"].iloc[0])
        tick_size = float(prepped["TICK_SIZE"].iloc[0])
        tick_value = float(prepped["TICK_VALUE"].iloc[0])
        fm[inst] = {"INSTRUMENT": inst, "POINT_VALUE": pv, "EXCHANGE_RATE": fx, "TICK_SIZE": tick_size,
                    "TICK_VALUE": tick_value, "INSTRUMENT_WEIGHTS": weights_map[inst], "STANDARD_COST": sc}
        sim_costs[inst] = sc

        _, passed = ewma.calc(inst, prepped.copy(), SPEEDS, sc, fx, pv)
        carry.calc(inst, prepped.copy(), fx, pv, standard_cost=sc)
        carry_df = pd.read_csv(os.path.join(SCRATCH, "output_instruments", f"{inst}_CARRY.csv"))
        carry_df["Date"] = pd.to_datetime(carry_df["Date"], format="%d/%m/%Y")

        px = pd.to_numeric(prepped["PX_CLOSE_1D"], errors="coerce")
        daily_return = px.ffill().pct_change(fill_method=None)
        dates = pd.to_datetime(prepped["Date"], format="%d/%m/%Y")

        # ---- Stage 1: EWMA internal FDM (blends the passing speeds into EWMA_combined) ----
        speeds = sorted(passed)
        n = len(speeds)
        ret_mat = pd.DataFrame({f: passed[f]["forecast*returns"] for f in speeds})
        fc_mat = pd.DataFrame({f: passed[f]["capped_forecast"] for f in speeds})
        raw_corr = ret_mat.corr()
        w = np.ones(n) / n
        old_fdm = old_multiplier(raw_corr, w, EWMA_FDM_CAP, fill_nan=True)
        res = new_multiplier(raw_corr, w, EWMA_FDM_CAP, fill_nan=True)
        weighted_sum = fc_mat.mul(w, axis=1).sum(axis=1, min_count=1)
        ewma_variants = {}
        for label, fdm in (("old", old_fdm), ("new", res.multiplier)):
            comb = (fdm * weighted_sum).clip(-CAP, CAP)
            ewma_variants[label] = pd.DataFrame({
                "Date": dates, "capped_forecast": comb.to_numpy(),
                "forecast_pct_return": ((comb.shift(1) / 10) * daily_return).to_numpy(),
            })
        off = raw_corr.to_numpy()[~np.eye(n, dtype=bool)]
        ewma_rows.append({
            "Instrument": inst, "Speeds": n, "Negative pairs": int((off < 0).sum() // 2),
            "Min raw corr": np.nanmin(off), "Old FDM": old_fdm, "New FDM": res.multiplier,
            "New-Old": res.multiplier - old_fdm, "PSD projected": res.psd_projected,
            **{f"Old {k}": v for k, v in forecast_stats(ewma_variants["old"]["capped_forecast"]).items()
               if k != "Max |combined forecast|"},
            **{f"New {k}": v for k, v in forecast_stats(ewma_variants["new"]["capped_forecast"]).items()
               if k != "Max |combined forecast|"},
        })
        audit_rows.append(constraint_row(f"{inst} EWMA speeds", res.multiplier_corr))
        corr_dumps.append((f"{inst} EWMA speeds - RAW correlation", res.raw_corr))
        corr_dumps.append((f"{inst} EWMA speeds - correlation USED for multiplier (negatives floored)", res.multiplier_corr))

        # ---- Stage 2: Combined Forecast (Carry + EWMA_combined), expanding-window FDM ----
        carry_frame = carry_df[["Date", "capped_forecast", "forecast_pct_return"]].reset_index(drop=True)
        cw = np.ones(2) / 2
        chains = {}
        for label, use_new in (("old", False), ("new", True)):
            ew = ewma_variants[label].reset_index(drop=True)
            chains[label] = expanding_combined([carry_frame, ew], cw, COMBINED_FDM_CAP, use_new)
        assert chains["old"]["Date"].equals(chains["new"]["Date"]), "old/new runs must cover identical dates"
        date_ranges[inst] = (chains["old"]["Date"].iloc[0].date(), chains["old"]["Date"].iloc[-1].date(), len(chains["old"]))
        last_raw = pd.DataFrame([carry_frame["forecast_pct_return"].values,
                                 ewma_variants["old"]["forecast_pct_return"].values]).T.corr()
        last_res = new_multiplier(last_raw, cw, COMBINED_FDM_CAP)
        audit_rows.append(constraint_row(f"{inst} Combined Forecast (final day)", last_res.multiplier_corr))
        comb_rows.append({
            "Instrument": inst, "Final-day raw corr (Carry vs EWMA)": last_raw.iloc[0, 1],
            "Final-day used corr": last_res.multiplier_corr.iloc[0, 1],
            "Old FDM (final day)": chains["old"]["Multiplier"].iloc[-1],
            "New FDM (final day)": chains["new"]["Multiplier"].iloc[-1],
            "Old FDM (mean)": chains["old"]["Multiplier"].mean(),
            "New FDM (mean)": chains["new"]["Multiplier"].mean(),
            "Days raw corr < 0": int((chains["old"]["RawCorr"] < 0).sum()),
            "Days": len(chains["old"]),
        })

        # ---- Stage 3: Strategy Analysis diagnostic multiplier (all speeds + carry, family weights) ----
        series = [passed[f]["forecast_pct_return"] for f in speeds] + [carry_df["forecast_pct_return"].to_numpy()]
        names = [f"EWMA{f:03d}" for f in speeds] + ["CARRY"]
        sw = np.array([0.5 / n] * n + [0.5])
        sraw = pd.DataFrame(series).T.corr()
        sraw.index = sraw.columns = names
        sres = new_multiplier(sraw, sw, COMBINED_FDM_CAP)
        soff = sraw.to_numpy()[~np.eye(len(names), dtype=bool)]
        strat_rows.append({
            "Instrument": inst, "Models": len(names), "Negative pairs": int((soff < 0).sum() // 2),
            "Old multiplier": old_multiplier(sraw, sw, COMBINED_FDM_CAP), "New multiplier": sres.multiplier,
            "PSD projected": sres.psd_projected,
        })
        audit_rows.append(constraint_row(f"{inst} Strategy Analysis", sres.multiplier_corr))
        corr_dumps.append((f"{inst} Strategy Analysis - RAW correlation", sres.raw_corr))
        corr_dumps.append((f"{inst} Strategy Analysis - correlation USED for multiplier (negatives floored)", sres.multiplier_corr))

        # ---- Instrument-level performance of each chain (isolated: weight=1, PDM=1, $10m) ----
        st_dev = simple_price_volatility(raw_input["PX_CLOSE_1D"])
        raw_dates = pd.to_datetime(raw_input["Date"], format="%d/%m/%Y")
        for label in ("old", "new"):
            ch = chains[label]
            pd.DataFrame({"Date": ch["Date"].dt.strftime("%Y-%m-%d"), "FinalForecast": ch["FinalForecast"],
                          "Multiplier": ch["Multiplier"]}).to_csv(
                os.path.join(old_cf_dir if label == "old" else new_cf_dir, f"{inst}.csv"), index=False)
            merged = pd.DataFrame({"Date": raw_dates, "px": pd.to_numeric(raw_input["PX_CLOSE_1D"]), "sd": st_dev}) \
                .merge(ch[["Date", "FinalForecast"]], on="Date", how="inner").sort_values("Date").reset_index(drop=True)
            sim = _isolated_instrument_sim(merged["px"], merged["sd"], merged["FinalForecast"],
                                           tick_value, tick_size, pv, fx)
            bench = merged["px"].pct_change(fill_method=None)
            m = _metrics_from_returns(sim["dollar_pct_return"], sim["turnover"], sc, merged["Date"], bench,
                                      is_dollar_based=True)
            sim_turnover[label][inst] = sim["turnover"]
            inst_rows.append({
                "Instrument": inst, "Method": label, "Combined FDM (mean)": ch["Multiplier"].mean(),
                **forecast_stats(ch["FinalForecast"]),
                "Avg |target pos|": sim["target_pos"].abs().mean(), "Max |target pos|": sim["target_pos"].abs().max(),
                "Realised vol": m["Standard Deviation"], "Turnover": m["Turnover"], "Costs (SC)": m["Costs (SC)"],
                "Cost %": m["Cost %"], "Gross Sharpe": m["Sharpe Ratio (Gross)"], "Net Sharpe": m["Sharpe Ratio (Net)"],
                "Max drawdown": m["Maximum Drawdown"],
            })

    # ---- Stage 4: portfolio PDM + real multi-instrument simulation ----
    px_pct = {}
    for inst in insts:
        d = pd.read_csv(os.path.join("DATA", "input_instruments", f"{inst}.csv"))
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        px_pct[inst] = d.set_index("Date")["PX_CLOSE_1D"].astype(float).pct_change(fill_method=None)
    px_pct_df = pd.concat(px_pct.values(), axis=1, keys=px_pct.keys()).dropna()
    praw = px_pct_df.corr()
    pw = np.array([weights_map[k] for k in px_pct_df.columns])
    pdm_old = old_multiplier(praw, pw, PDM_CAP)
    pdm_res = new_multiplier(praw, pw, PDM_CAP)
    pdm_row = {"Instruments": ", ".join(insts), "Weights": ", ".join(f"{weights_map[k]:.2f}" for k in px_pct_df.columns),
               "Raw corr (off-diagonal min)": np.nanmin(praw.to_numpy()[~np.eye(len(insts), dtype=bool)]) if len(insts) > 1 else np.nan,
               "Old PDM": pdm_old, "New PDM": pdm_res.multiplier, "PSD projected": pdm_res.psd_projected}
    audit_rows.append(constraint_row("Portfolio PDM", pdm_res.multiplier_corr))
    corr_dumps.append(("Portfolio PDM - RAW correlation", pdm_res.raw_corr))
    corr_dumps.append(("Portfolio PDM - correlation USED for multiplier (negatives floored)", pdm_res.multiplier_corr))

    port_rows = []
    port_index = {}
    for label, pdm, folder in (("old", pdm_old, old_cf_dir), ("new", pdm_res.multiplier, new_cf_dir)):
        raw_csvs = {i: pd.read_csv(os.path.join("DATA", "input_instruments", f"{i}.csv")) for i in insts}
        order_df = framework_main(fm, folder, raw_csvs, pdm, "%d/%m/%Y", AUM, is_markov=False)
        port_index[label] = order_df.index
        rets = order_df["total_pnl_today"] / order_df["AUM"].shift(1)
        tw = {i: weights_map[i] * sim_turnover[label][i] for i in insts if pd.notna(sim_turnover[label][i])}
        p_turn = sum(tw.values()) / sum(weights_map[i] for i in tw)
        p_cost = sum(tw[i] * sim_costs[i] for i in tw) / sum(tw.values())
        m = _metrics_from_returns(rets, p_turn, p_cost, order_df.index.to_series(), None, is_dollar_based=True)
        tp = order_df[[c for c in order_df.columns if c.endswith("_alpha_target_pos")]].apply(pd.to_numeric, errors="coerce")
        ff_all = order_df[[c for c in order_df.columns if c.endswith("_alpha_forecast")]].apply(pd.to_numeric, errors="coerce")
        port_rows.append({
            "Method": label, "PDM": pdm, "Avg |combined forecast|": ff_all.abs().stack().mean(),
            "Max |combined forecast|": ff_all.abs().stack().max(),
            "% at +/-20": 100.0 * (ff_all.abs().stack() >= CAP - 1e-9).mean(),
            "Avg |target pos| (weight x PDM applied)": tp.abs().stack().mean(), "Max |target pos|": tp.abs().stack().max(),
            "Realised vol": m["Standard Deviation"], "Turnover": p_turn, "Costs (SC eff.)": p_cost,
            "Cost %": m["Cost %"], "Gross Sharpe": m["Sharpe Ratio (Gross)"], "Net Sharpe": m["Sharpe Ratio (Net)"],
            "Max drawdown": m["Maximum Drawdown"], "Final AUM": order_df["AUM"].iloc[-1],
        })

    assert port_index["old"].equals(port_index["new"]), "portfolio old/new runs must cover identical dates"

    # ---- Stage 5: EWMA Norm's static FDM table vs a data-driven (raw / zero-floored) equivalent ----
    norm_rows = []
    for inst in insts:
        prepped = prepare_strategy_input(inst)
        out = p1_analysis._compute_ewma_norm(prepped, standard_cost=fm[inst]["STANDARD_COST"],
                                             exchange_rate=fm[inst]["EXCHANGE_RATE"], point_value=fm[inst]["POINT_VALUE"])
        ret = pd.to_numeric(prepped["PX_CLOSE_1D"]).pct_change(fill_method=None)
        spans = [int(c.split("_")[2]) for c in out.columns if c.endswith("_cost_pass") and bool(out[c].iloc[0])]
        if len(spans) < 2:
            continue
        r = pd.DataFrame({sp: out[f"ewma_norm_{sp}_forecast"].shift(1) * ret for sp in spans})
        wn = np.full(len(spans), 1.0 / len(spans))
        raw_c = r.corr()
        norm_rows.append({
            "Instrument": inst, "Rules passing": len(spans),
            "Static table FDM": p1_analysis._get_ewma_norm_fdm(len(spans)),
            "Data-driven, raw corr": old_multiplier(raw_c, wn, 1e9),
            "Data-driven, floored": new_multiplier(raw_c, wn, 1e9).uncapped_multiplier,
        })

    tables = {
        "1_ewma_internal_fdm": pd.DataFrame(ewma_rows),
        "2_combined_forecast_fdm": pd.DataFrame(comb_rows),
        "3_strategy_analysis_multiplier": pd.DataFrame(strat_rows),
        "4_instrument_level_old_vs_new": pd.DataFrame(inst_rows),
        "5_portfolio_pdm": pd.DataFrame([pdm_row]),
        "6_portfolio_old_vs_new": pd.DataFrame(port_rows),
        "7_ewma_norm_static_table_check": pd.DataFrame(norm_rows),
        "8_constraint_audit_matrices_used": pd.DataFrame(audit_rows),
    }
    print(f"\nInstruments: {insts}")
    print(f"Instrument weights: {weights_map}  <- {weights_source}")
    for i in insts:
        print(f"Standard cost {i}: {cost_sources[i]}")
        print(f"Dates {i} (identical in old and new runs): {date_ranges[i][0]} -> {date_ranges[i][1]}, {date_ranges[i][2]} days")
    print("Strategy weights: EWMA speeds 1/n each; Combined Forecast Carry 1/2 + EWMA_combined 1/2; "
          "Strategy Analysis family weights EWMA 0.5 (split over speeds) + CARRY 0.5\n")
    for name, df in tables.items():
        print("=" * 30, name)
        print(fmt(df.set_index(df.columns[0])) if name in ("1_ewma_internal_fdm", "2_combined_forecast_fdm",
              "3_strategy_analysis_multiplier", "7_ewma_norm_static_table_check") else fmt(df))
        print()
    print("=" * 30, "correlation matrices (raw vs used for multiplier)")
    for title, m in corr_dumps:
        print(f"\n{title}")
        print(m.round(3).to_string())

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        for i, (title, m) in enumerate(corr_dumps):
            m.to_csv(os.path.join(args.out, f"corr_{i:02d}_{title.replace(' ', '_').replace('/', '-')[:60]}.csv"))
        print(f"\nSaved CSVs to {args.out}")


if __name__ == "__main__":
    try:
        main()
    finally:
        os.path.join = _orig_join
        shutil.rmtree(SCRATCH, ignore_errors=True)
