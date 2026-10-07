import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main

GOOD = {
    "source": "wind",
    "weather": {"ALLSKY_SFC_SW_DWN": 300, "ALLSKY_SFC_SW_DNI": 400,
                "ALLSKY_SFC_SW_DIFF": 80, "T2M": 12.5, "WS10M": 6.2},
}


class StubModel:
    def predict(self, frame):
        assert list(frame.columns) == main.FEATURES
        return np.array([4321.987])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda source: StubModel())
    monkeypatch.setattr(main, "model_info", lambda: {"wind": {"model": "stub"}})
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_rounded_estimate(client):
    response = client.post("/predict", json=GOOD)
    assert response.status_code == 200
    assert response.json() == {"source": "wind", "predicted_mwh": 4322.0, "model": "stub"}


def test_negative_predictions_are_clipped_to_zero(client, monkeypatch):
    class Negative:
        def predict(self, frame):
            return np.array([-50.0])
    monkeypatch.setattr(main, "load_model", lambda source: Negative())
    assert client.post("/predict", json=GOOD).json()["predicted_mwh"] == 0.0


@pytest.mark.parametrize("field,value", [("WS10M", -1), ("WS10M", 200), ("T2M", 99), ("ALLSKY_SFC_SW_DWN", -5)])
def test_out_of_range_weather_is_rejected(client, field, value):
    body = {**GOOD, "weather": {**GOOD["weather"], field: value}}
    assert client.post("/predict", json=body).status_code == 422


def test_unknown_source_is_rejected(client):
    assert client.post("/predict", json={**GOOD, "source": "hydro"}).status_code == 422


def test_missing_artifact_gives_503(monkeypatch):
    def missing(source):
        raise FileNotFoundError(source)
    monkeypatch.setattr(main, "load_model", missing)
    assert TestClient(main.app).post("/predict", json=GOOD).status_code == 503
