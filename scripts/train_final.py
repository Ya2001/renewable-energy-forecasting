"""Train the serving models on all data and write artifacts/.

For each source it picks the better of Random Forest and SVR by R² on the
chronological hold-out in results/metrics.csv, then refits it on every row.

Usage:
    python -m scripts.train_final --production path/to/intermittent-renewables-production-france.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from renewables.data import SOURCES, TARGET, WEATHER_FEATURES, build_dataset, load_production, load_weather
from renewables.models import RF_PARAMS, SVR_PARAMS

ROOT = Path(__file__).resolve().parent.parent
SERVING_TREES = 200  # fewer trees than the 500 evaluated, to keep the artifact small


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--weather", type=Path,
                        default=ROOT / "data/raw/POWER_Point_Hourly_20200622_20230630_047d27N_002d36E_LST.csv")
    args = parser.parse_args()

    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    weather, production = load_weather(args.weather), load_production(args.production, "utc")
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    info = {}
    for source in SOURCES:
        pool = metrics[(metrics.source == source) & (metrics.protocol == "chronological")
                       & metrics.model.isin(["random_forest", "svr"])]
        best = pool.sort_values("r2").iloc[-1]
        data = build_dataset(source, production, weather)
        if best.model == "random_forest":
            model = RandomForestRegressor(**{**RF_PARAMS, "n_estimators": SERVING_TREES})
        else:
            model = make_pipeline(StandardScaler(), SVR(**SVR_PARAMS))
        model.fit(data[WEATHER_FEATURES], data[TARGET])
        joblib.dump(model, out / f"{source}_model.joblib", compress=3)
        info[source] = {
            "model": best.model,
            "heldout_chronological": {k: float(best[k]) for k in ("rmse", "mae", "r2")},
            "trained_rows": int(len(data)),
            "data_range": [str(data.index.min().date()), str(data.index.max().date())],
        }
        print(source, info[source])
    (out / "model_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
