"""Score every model on one protocol and write results/ (tables and figures).

Usage:
    python -m scripts.run_experiments --production path/to/intermittent-renewables-production-france.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from renewables import models as m
from renewables.data import (
    SOURCES, TARGET, build_dataset, chronological_split, load_production,
    load_weather, random_split,
)

ROOT = Path(__file__).resolve().parent.parent


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def row(source, protocol, model, y_true, y_pred, n_train, n_test) -> dict:
    return {"source": source, "protocol": protocol, "model": model,
            **m.score(y_true, y_pred), "n_train": n_train, "n_test": n_test}


def evaluate_source(source, production, weather, skip_slow):
    data = build_dataset(source, production, weather)
    log(f"{source}: {len(data)} hourly rows, {data.index.min():%Y-%m-%d} to {data.index.max():%Y-%m-%d}")
    rows, preds = [], {}

    # --- chronological split: the honest forecasting protocol ----------------------
    train, test = chronological_split(data)
    n_tr, n_te = len(train), len(test)
    y = test[TARGET]
    candidates = {
        "hourly_mean (baseline)": m.hourly_mean(train, test),
        "persistence_24h (baseline)": m.persistence_24h(data, test, train),
        "holt_winters": m.holt_winters(train, test),
        "random_forest": m.random_forest(train, test),
    }
    if not skip_slow:
        for name, fn in (("svr", m.support_vector), ("prophet+weather", m.prophet_with_weather),
                         ("bilstm", m.bilstm)):
            t0 = time.time()
            candidates[name] = fn(train, test)
            log(f"{source}: {name} done in {time.time() - t0:.0f}s")
        candidates["ensemble (svr, bilstm, prophet)"] = m.mean_ensemble(
            {k: candidates[k] for k in ("svr", "bilstm", "prophet+weather")})
        candidates["ensemble (rf, svr, bilstm)"] = m.mean_ensemble(
            {k: candidates[k] for k in ("random_forest", "svr", "bilstm")})
    for name, pred in candidates.items():
        rows.append(row(source, "chronological", name, y, pred, n_tr, n_te))
    preds["chronological"] = candidates

    # --- random split: the dissertation's original protocol ---------------------------
    rtrain, rtest = random_split(data)
    for name, fn in (("random_forest", m.random_forest), ("svr", m.support_vector)):
        if skip_slow and name == "svr":
            continue
        pred = fn(rtrain, rtest)
        rows.append(row(source, "random (dissertation protocol)", name, rtest[TARGET], pred, len(rtrain), len(rtest)))
    return rows, preds, test


def alignment_check(source, production_path, weather):
    """Random Forest under the dissertation's timestamp handling versus the corrected one."""
    rows = []
    for alignment in ("dissertation", "utc"):
        data = build_dataset(source, load_production(production_path, alignment), weather)
        for protocol, split in (("random", random_split), ("chronological", chronological_split)):
            train, test = split(data)
            pred = m.random_forest(train, test)
            rows.append({"source": source, "alignment": alignment, "protocol": protocol,
                         "n_rows": len(data), **m.score(test[TARGET], pred)})
    return rows


def plot_results(source, metrics, preds, test, out):
    chrono = metrics[(metrics.source == source) & (metrics.protocol == "chronological")].sort_values("r2")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(chrono.model, chrono.r2, color="#2b6cb0")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("R² on the held-out final 30% (chronological)")
    ax.set_title(f"{source.title()}: model comparison")
    fig.tight_layout(); fig.savefig(out / f"{source}_model_comparison.png", dpi=130); plt.close(fig)

    best = chrono.sort_values("r2").iloc[-1].model
    window = test.index[: 24 * 14]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window, test.loc[window, TARGET], label="actual", color="#222")
    ax.plot(window, preds["chronological"][best].loc[window], label=f"{best}", color="#dd6b20")
    ax.set_ylabel("Production (MWh)"); ax.set_title(f"{source.title()}: first two test weeks, {best}")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out / f"{source}_best_two_weeks.png", dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--weather", type=Path,
                        default=ROOT / "data/raw/POWER_Point_Hourly_20200622_20230630_047d27N_002d36E_LST.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-slow", action="store_true", help="skip SVR, Prophet and LSTM")
    args = parser.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    weather = load_weather(args.weather)
    production = load_production(args.production, "utc")
    all_rows, held = [], {}
    for source in SOURCES:
        rows, preds, test = evaluate_source(source, production, weather, args.skip_slow)
        all_rows += rows
        held[source] = (preds, test)
    metrics = pd.DataFrame(all_rows).round(4)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    for source, (preds, test) in held.items():
        plot_results(source, metrics, preds, test, args.out / "figures")

    align = pd.DataFrame([r for s in SOURCES for r in alignment_check(s, args.production, weather)]).round(4)
    align.to_csv(args.out / "alignment_check.csv", index=False)
    log("finished")
    print(metrics.to_string(index=False)); print(align.to_string(index=False))


if __name__ == "__main__":
    main()
