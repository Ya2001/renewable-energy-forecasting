"""Forecasting models behind one interface: fit on train, return test predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data import CALENDAR_FEATURES, TARGET, WEATHER_FEATURES

# Hyperparameters taken from the dissertation's GridSearchCV results.
RF_PARAMS = dict(
    n_estimators=500, max_depth=10, min_samples_split=10, min_samples_leaf=1,
    bootstrap=True, random_state=42, n_jobs=-1,
)
SVR_PARAMS = dict(C=1000, epsilon=0.5, gamma="auto", kernel="rbf")


def score(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---- baselines ------------------------------------------------------------------
def hourly_mean(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Average production for each hour of day in the training set."""
    means = train.groupby(train.index.hour)[TARGET].mean()
    return pd.Series(test.index.hour.map(means).to_numpy(), index=test.index)


def persistence_24h(full: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Same hour yesterday, falling back to the hourly mean when yesterday is missing."""
    lagged = full[TARGET].reindex(test.index - pd.Timedelta(hours=24))
    pred = pd.Series(lagged.to_numpy(), index=test.index)
    return pred.fillna(hourly_mean(train, test))


# ---- weather-driven models ----------------------------------------------------
def random_forest(train, test, features=None) -> pd.Series:
    features = features or WEATHER_FEATURES
    model = RandomForestRegressor(**RF_PARAMS).fit(train[features], train[TARGET])
    return pd.Series(model.predict(test[features]), index=test.index)


def support_vector(train, test, features=None) -> pd.Series:
    features = features or WEATHER_FEATURES
    model = make_pipeline(StandardScaler(), SVR(**SVR_PARAMS))
    model.fit(train[features], train[TARGET])
    return pd.Series(model.predict(test[features]), index=test.index)


# ---- time-series models ---------------------------------------------------------
def holt_winters(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Additive daily seasonality, no trend (a trend extrapolated over thousands of
    steps diverges, which is what produced the dissertation's absurd errors)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = train[TARGET].asfreq("h").interpolate(limit_direction="both")
    fit = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=24).fit()
    horizon = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), test.index[-1], freq="h")
    forecast = fit.forecast(len(horizon))
    forecast.index = horizon
    return forecast.reindex(test.index).ffill().bfill()


def prophet_with_weather(train: pd.DataFrame, test: pd.DataFrame, features=None) -> pd.Series:
    from prophet import Prophet

    features = features or WEATHER_FEATURES
    frame = lambda d: d.reset_index().rename(columns={"timestamp": "ds"}).assign(
        ds=lambda x: x["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
    )
    model = Prophet(
        changepoint_prior_scale=0.1, seasonality_prior_scale=1.0,
        seasonality_mode="additive", daily_seasonality=True,
    )
    for name in features:
        model.add_regressor(name)
    model.fit(frame(train).rename(columns={TARGET: "y"})[["ds", "y"] + features])
    forecast = model.predict(frame(test)[["ds"] + features])
    return pd.Series(forecast["yhat"].to_numpy(), index=test.index)


def bilstm(train: pd.DataFrame, test: pd.DataFrame, features=None, window=12, seed=42) -> pd.Series:
    """Bidirectional LSTM over a 12-hour window of weather, predicting the last hour.

    Windows for the first test rows reach back into the training tail; only
    features (never test targets) are used, so there is no leakage.
    """
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    features = features or WEATHER_FEATURES
    tf.keras.utils.set_random_seed(seed)
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    x_train = x_scaler.fit_transform(train[features])
    x_all = x_scaler.transform(pd.concat([train[features], test[features]]))
    y_train = y_scaler.fit_transform(train[[TARGET]])

    def windows(x, start, stop):
        return np.stack([x[i - window + 1 : i + 1] for i in range(start, stop)])

    xtr = windows(x_train, window - 1, len(train))
    ytr = y_train[window - 1 :]
    xte = windows(x_all, len(train), len(train) + len(test))

    model = Sequential([
        Input(shape=(window, len(features))),
        Bidirectional(LSTM(50, return_sequences=True)), Dropout(0.2),
        Bidirectional(LSTM(50)), Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")
    model.fit(
        xtr, ytr, epochs=30, batch_size=64, validation_split=0.1, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )
    pred = y_scaler.inverse_transform(model.predict(xte, verbose=0)).ravel()
    return pd.Series(pred, index=test.index)


def mean_ensemble(predictions: dict[str, pd.Series]) -> pd.Series:
    """Simple average of the supplied model predictions."""
    return pd.concat(predictions.values(), axis=1).mean(axis=1)


__all__ = [
    "score", "hourly_mean", "persistence_24h", "random_forest", "support_vector",
    "holt_winters", "prophet_with_weather", "bilstm", "mean_ensemble", "CALENDAR_FEATURES",
]
