import pandas as pd
import pytest

from renewables.data import (
    SOURCES, build_dataset, chronological_split, load_production, load_weather, random_split,
)

WEATHER_CSV = """-BEGIN HEADER-
NASA/POWER test header
-END HEADER-
YEAR,MO,DY,HR,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,ALLSKY_KT,T2M,WS10M,PS
2021,7,1,10,500.0,600.0,100.0,0.5,20.0,3.0,100.0
2021,7,1,11,550.0,650.0,110.0,-999.0,21.0,4.0,100.0
2021,7,1,12,600.0,700.0,120.0,0.6,22.0,5.0,100.0
"""

PRODUCTION_CSV = """Date and Hour,Date,StartHour,EndHour,Source,Production,dayOfYear,dayName,monthName
2021-07-01 12:00:00+02:00,2021-07-01,12:00:00,13:00:00,Solar,900.0,182,Thursday,July
2021-07-01 13:00:00+02:00,2021-07-01,13:00:00,14:00:00,Solar,950.0,182,Thursday,July
2021-07-01 12:00:00+02:00,2021-07-01,12:00:00,13:00:00,Wind,3000.0,182,Thursday,July
"""


@pytest.fixture
def files(tmp_path):
    (tmp_path / "w.csv").write_text(WEATHER_CSV)
    (tmp_path / "p.csv").write_text(PRODUCTION_CSV)
    return tmp_path / "w.csv", tmp_path / "p.csv"


def test_weather_skips_header_and_flags_missing(files):
    weather = load_weather(files[0])
    assert list(weather.index.hour) == [10, 11, 12]
    assert "ALLSKY_KT" not in weather.columns
    assert weather.isna().sum().sum() == 0


def test_utc_alignment_applies_the_timezone_offset(files):
    utc = load_production(files[1], "utc")
    naive = load_production(files[1], "dissertation")
    # 12:00 local (+02:00) is 10:00 UTC; the dissertation parsing keeps 12:00.
    assert utc["timestamp"].iloc[0].hour == 10
    assert naive["timestamp"].iloc[0].hour == 12


def test_build_dataset_joins_on_the_hour(files):
    weather = load_weather(files[0])
    solar = build_dataset("solar", load_production(files[1], "utc"), weather)
    assert len(solar) == 2  # 10:00 and 11:00 UTC match; 12:00 UTC has no production row
    assert solar["Production"].tolist() == [900.0, 950.0]
    wind = build_dataset("wind", load_production(files[1], "utc"), weather)
    assert len(wind) == 1


def test_unknown_source_is_rejected(files):
    with pytest.raises(ValueError):
        build_dataset("hydro", load_production(files[1]), load_weather(files[0]))


def test_chronological_split_never_leaks_the_future():
    idx = pd.date_range("2022-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame({"Production": range(100)}, index=idx)
    train, test = chronological_split(df, 0.3)
    assert train.index.max() < test.index.min()
    assert len(train) == 70 and len(test) == 30


def test_random_split_shuffles_rows():
    idx = pd.date_range("2022-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame({"Production": range(100)}, index=idx)
    train, test = random_split(df, 0.3)
    assert train.index.max() > test.index.min()


def test_sources_constant():
    assert SOURCES == ("wind", "solar")
