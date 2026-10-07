"""Load and align French wind/solar production with NASA POWER weather data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

WEATHER_FEATURES = [
    "ALLSKY_SFC_SW_DWN",  # surface shortwave downward irradiance (Wh/m^2)
    "ALLSKY_SFC_SW_DNI",  # direct normal irradiance (Wh/m^2)
    "ALLSKY_SFC_SW_DIFF",  # diffuse irradiance (Wh/m^2)
    "T2M",  # temperature at 2 m (C)
    "WS10M",  # wind speed at 10 m (m/s)
]
TARGET = "Production"
SOURCES = ("wind", "solar")
MISSING_VALUES = (-999.0, -99.0)  # NASA POWER missing-value markers


def load_weather(path: str | Path) -> pd.DataFrame:
    """Read a NASA POWER hourly point CSV (header block is skipped).

    The file is in local solar time at ~2.36 E, which is within ten minutes of UTC,
    so timestamps are treated as UTC.
    """
    path = Path(path)
    skip = 0
    with path.open() as handle:
        for number, line in enumerate(handle):
            if line.startswith("-END HEADER-"):
                skip = number + 1
                break
    df = pd.read_csv(path, skiprows=skip)
    df["timestamp"] = pd.to_datetime(
        {"year": df["YEAR"], "month": df["MO"], "day": df["DY"], "hour": df["HR"]}, utc=True
    )
    df = df.set_index("timestamp").replace(list(MISSING_VALUES), np.nan)
    return df[WEATHER_FEATURES]


def load_production(path: str | Path, alignment: str = "utc") -> pd.DataFrame:
    """Read hourly wind/solar production.

    alignment="utc" uses the timezone-aware 'Date and Hour' column (correct).
    alignment="dissertation" rebuilds the timestamp from 'Date' + 'StartHour', which
    drops the +01:00/+02:00 offset and shifts production against UTC weather.
    """
    df = pd.read_csv(path)
    if alignment == "utc":
        df["timestamp"] = pd.to_datetime(df["Date and Hour"], utc=True)
    elif alignment == "dissertation":
        df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["StartHour"], utc=True)
    else:
        raise ValueError(f"unknown alignment: {alignment!r}")
    return df[["timestamp", "Source", TARGET]]


def build_dataset(
    source: str, production: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """Merge one source's production with weather on the hour, sorted by time."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    prod = production[production["Source"].str.lower() == source]
    prod = prod.drop_duplicates(subset="timestamp").set_index("timestamp")[[TARGET]]
    merged = prod.join(weather, how="inner").dropna().sort_index()
    return merged


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical hour-of-day and day-of-year features (known in advance)."""
    out = df.copy()
    hour = out.index.hour + out.index.minute / 60
    doy = out.index.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


CALENDAR_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def chronological_split(df: pd.DataFrame, test_frac: float = 0.3):
    """Train on the earliest rows, test on the latest. Never shuffled."""
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def random_split(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 42):
    """The dissertation's original protocol: a shuffled split of hourly rows."""
    train, test = train_test_split(df, test_size=test_frac, random_state=seed)
    return train, test
