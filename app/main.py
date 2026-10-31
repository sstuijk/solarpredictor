from __future__ import annotations

import json
import logging
from io import BytesIO
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from websockets.sync.client import connect
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("solar_forecast")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "lightgbm_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", MODEL_DIR / "metrics.json"))

LATITUDE = float(os.getenv("LATITUDE", "50.95"))
LONGITUDE = float(os.getenv("LONGITUDE", "5.87"))
TIMEZONE = os.getenv("TIMEZONE", "Europe/Amsterdam")
PANEL_CAPACITY_KWP = float(os.getenv("PANEL_CAPACITY_KWP", "9.8"))
PANEL_TILT = float(os.getenv("PANEL_TILT", "30"))
PANEL_AZIMUTH = float(os.getenv("PANEL_AZIMUTH", "0"))
HA_URL = os.getenv("HA_URL", "").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")
HA_ENTITY_ID = os.getenv("HA_ENTITY_ID", "")
HA_ENERGY_UNIT = os.getenv("HA_ENERGY_UNIT", "kWh").strip()
# cumulative_energy uses Home Assistant hourly long-term statistics (WebSocket).
# interval_energy uses the REST history endpoint for per-interval readings.
HA_FEED_MODE = os.getenv("HA_FEED_MODE", "cumulative_energy").strip().lower()
MAX_FORECAST_DAYS = int(os.getenv("MAX_FORECAST_DAYS", "7"))
OPEN_METEO_MODEL="best_match"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_WEATHER = [
    "temperature_2m", "relative_humidity_2m", "cloud_cover_low", "cloud_cover_mid",
    "cloud_cover_high", "rain", "snow_depth",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "wind_speed_10m", "precipitation", "is_day", "global_tilted_irradiance",
]
WEATHER_FEATURES = HOURLY_WEATHER
TIME_FEATURES = [
    "hour", "day_of_year", "month", "day_of_week",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
FEATURES = WEATHER_FEATURES + TIME_FEATURES

app = FastAPI(
    title="Solar Production Forecast API",
    version="1.1.0",
    description="Home Assistant energy history + Open-Meteo + LightGBM solar forecast.",
)


class TrainRequest(BaseModel):
    start: datetime | None = Field(default=None, description="Optional start of history; defaults to 365 days ago")
    end: datetime | None = Field(default=None, description="Optional end of history; defaults to now")
    min_rows: int = Field(default=500, ge=24, le=1_000_000)


class ForecastRequest(BaseModel):
    start: datetime | None = Field(default=None, description="Start time; defaults to current hour")
    hours: int = Field(default=168, ge=1, le=168)


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _http_get_json(url: str, params: dict[str, Any] | None = None,
                   headers: dict[str, str] | None = None) -> Any:
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        logger.exception("HTTP request failed")
        raise HTTPException(status_code=502, detail=f"External API request failed: {exc}") from exc


def _weather_params() -> dict[str, Any]:
    # Use UTC timestamps internally to avoid ambiguous/repeated local hours at DST.
    # Time-of-day and daily aggregation are converted to TIMEZONE separately.
    return {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(HOURLY_WEATHER),
        "timezone": "UTC",
        "tilt": PANEL_TILT,
        "azimuth": PANEL_AZIMUTH,
        "models": OPEN_METEO_MODEL,
    }


def _weather_frame(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise HTTPException(status_code=502, detail="Open-Meteo returned no hourly data")
    frame = pd.DataFrame(hourly)
    # Open-Meteo is requested in UTC, so keep all internal timestamps UTC-aware.
    frame["timestamp"] = pd.to_datetime(frame.pop("time"), errors="coerce", utc=True)
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp")


def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    out["hour"] = ts.dt.hour
    out["day_of_year"] = ts.dt.dayofyear
    out["month"] = ts.dt.month
    out["day_of_week"] = ts.dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    for col in WEATHER_FEATURES:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _cumulative_to_hourly(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Convert cumulative meter samples to hourly kWh, using UTC-safe bins.

    Short intervals are apportioned by elapsed time across hour boundaries. Long
    zero-delta intervals are treated as no production (typical overnight state
    silence). Positive deltas spanning long gaps are assigned to the end hour,
    flagged for review, because their true within-gap timing is unknowable from
    cumulative readings alone.
    """
    df = frame[["timestamp", "cumulative_kwh"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["cumulative_kwh"] = pd.to_numeric(df["cumulative_kwh"], errors="coerce")
    df = (df.dropna().sort_values("timestamp")
          .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="At least two numeric cumulative meter readings are needed to calculate production.")

    hourly_index = pd.date_range(
        start=df["timestamp"].iloc[0].floor("h"),
        end=df["timestamp"].iloc[-1].floor("h"),
        freq="h", tz="UTC",
    )
    energy = pd.Series(0.0, index=hourly_index, dtype=float)
    long_positive_gaps = 0
    counter_resets = 0
    max_short_gap = pd.Timedelta(minutes=20)

    for i in range(1, len(df)):
        t0 = df.loc[i - 1, "timestamp"]
        t1 = df.loc[i, "timestamp"]
        c0 = float(df.loc[i - 1, "cumulative_kwh"])
        c1 = float(df.loc[i, "cumulative_kwh"])
        elapsed = t1 - t0
        if elapsed <= pd.Timedelta(0):
            continue

        delta = c1 - c0
        if delta < -0.001:
            # Meter reset/replacement. This interval is invalid; the next
            # pair uses the post-reset value as its baseline automatically.
            counter_resets += 1
            continue
        delta = max(0.0, delta)
        if delta == 0:
            continue

        elapsed_hours = elapsed.total_seconds() / 3600.0
        if delta > PANEL_CAPACITY_KWP * elapsed_hours * 1.25 + 0.02:
            long_positive_gaps += 1
            logger.warning("Skipping implausible cumulative meter increment %.3f kWh over %.1f minutes", delta, elapsed.total_seconds() / 60.0)
            continue

        if elapsed > max_short_gap:
            # A positive change over a long interval cannot be located precisely.
            # Put it in the end hour as a conservative, explicit fallback and log it.
            bucket = t1.floor("h")
            if bucket in energy.index:
                energy.loc[bucket] += delta
            long_positive_gaps += 1
            continue

        # Proportionally allocate a short-interval increment across every UTC
        # hour it overlaps, avoiding errors when readings straddle hour boundaries.
        cursor = t0
        while cursor < t1:
            bucket = cursor.floor("h")
            segment_end = min(t1, bucket + pd.Timedelta(hours=1))
            overlap = (segment_end - cursor).total_seconds()
            energy_fraction = overlap / elapsed.total_seconds()
            if bucket in energy.index:
                energy.loc[bucket] += delta * energy_fraction
            cursor = segment_end

    result = energy.rename("target").rename_axis("timestamp").reset_index()
    if long_positive_gaps:
        logger.warning("Found %d positive meter increments spanning gaps over 20 minutes or failing plausibility checks; inspect HA history and exclude outage periods if needed.", long_positive_gaps)
    if counter_resets:
        logger.warning("Found %d cumulative meter counter decreases/resets; those increments were ignored.", counter_resets)
    return result, long_positive_gaps


def _get_home_assistant_long_term_statistics(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch Home Assistant hourly long-term statistics and derive kWh/hour.

    Home Assistant's ``sum`` statistic is cumulative across hourly statistics
    for supported total-increasing energy sensors. Differencing adjacent sums
    yields energy for each hour while allowing Home Assistant to account for
    cumulative-meter resets. Timestamps are kept UTC-aware throughout.
    """
    ws_url = HA_URL
    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[len("https://"): ]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[len("http://"): ]
    ws_url += "/api/websocket"

    # Include earlier periods to calculate the first requested difference.
    query_start = start_ts.floor("h") - pd.Timedelta(hours=2)
    try:
        with connect(
            ws_url,
            open_timeout=30,
            close_timeout=10,
            max_size=20 * 1024 * 1024,
        ) as ws:
            greeting = json.loads(ws.recv())
            if greeting.get("type") != "auth_required":
                raise ValueError(f"Unexpected WebSocket greeting: {greeting}")

            ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
            auth = json.loads(ws.recv())
            if auth.get("type") != "auth_ok":
                raise ValueError(f"Home Assistant WebSocket authentication failed: {auth}")

            request_id = 1
            ws.send(json.dumps({
                "id": request_id,
                "type": "recorder/statistics_during_period",
                "start_time": query_start.isoformat(),
                "end_time": end_ts.isoformat(),
                "statistic_ids": [HA_ENTITY_ID],
                "period": "hour",
                "types": ["sum"],
                "units": {"energy": "kWh"},
            }))

            while True:
                message = json.loads(ws.recv())
                if message.get("id") == request_id:
                    break

            if not message.get("success"):
                error = message.get("error", {})
                raise ValueError(f"Statistics request failed: {error}")
            rows = (message.get("result") or {}).get(HA_ENTITY_ID, [])
    except Exception as exc:
        logger.exception("Could not fetch Home Assistant long-term statistics")
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not retrieve Home Assistant long-term statistics. Check "
                "HA_URL, token permissions, statistic ID, and application logs. "
                f"Details: {exc}"
            ),
        ) from exc

    if not rows:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No long-term statistics found for '{HA_ENTITY_ID}'. Check "
                "Home Assistant Settings > Developer tools > Statistics and "
                "confirm that this statistic ID has hourly statistics. The "
                "sensor should have a supported energy device class and state class."
            ),
        )

    records = []
    for row in rows:
        start_value = row.get("start")
        sum_value = row.get("sum")
        if start_value is None or sum_value is None:
            continue
        try:
            # HA's WebSocket API normally encodes statistic timestamps as ms.
            if isinstance(start_value, (int, float)):
                timestamp = pd.to_datetime(start_value, unit="ms", utc=True)
            else:
                timestamp = pd.to_datetime(start_value, utc=True, errors="coerce")
            sum_kwh = float(sum_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if pd.isna(timestamp) or not np.isfinite(sum_kwh):
            continue
        records.append({"timestamp": timestamp, "sum_kwh": sum_kwh})

    frame = (
        pd.DataFrame(records)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough numeric Home Assistant long-term statistics to calculate hourly energy.",
        )

    frame["target"] = frame["sum_kwh"].diff()
    # Exclude counter corrections and physically implausible single-hour jumps.
    max_hourly_kwh = PANEL_CAPACITY_KWP * 1.25 + 0.02
    invalid = (frame["target"] < -0.001) | (frame["target"] > max_hourly_kwh)
    invalid_count = int(invalid.sum())
    if invalid_count:
        logger.warning(
            "Skipping %d invalid hourly statistics increments for %s",
            invalid_count, HA_ENTITY_ID,
        )
    frame.loc[invalid, "target"] = np.nan
    frame["target"] = frame["target"].clip(lower=0)

    lower = start_ts.ceil("h")
    upper = end_ts.floor("h")
    frame = frame[
        (frame["timestamp"] >= lower) & (frame["timestamp"] < upper)
    ].dropna(subset=["target"])
    return frame[["timestamp", "target"]].reset_index(drop=True)


def _get_home_assistant_history(start: datetime, end: datetime) -> pd.DataFrame:
    if not HA_URL or not HA_TOKEN or not HA_ENTITY_ID:
        raise HTTPException(
            status_code=500,
            detail="Set HA_URL, HA_TOKEN, and HA_ENTITY_ID in .env before training.",
        )
    if HA_FEED_MODE not in ("interval_energy", "cumulative_energy"):
        raise HTTPException(status_code=500, detail="HA_FEED_MODE must be interval_energy or cumulative_energy.")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(TIMEZONE)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(TIMEZONE)
    start_ts = start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_convert("UTC")

    # Cumulative-energy mode uses HA's hourly long-term statistics, which are
    # retained independently of the recorder's short-term state history.
    if HA_FEED_MODE == "cumulative_energy":
        return _get_home_assistant_long_term_statistics(start_ts, end_ts)

    # Interval-energy mode still uses the REST history endpoint.
    query_start = start_ts - pd.Timedelta(hours=2) if HA_FEED_MODE == "cumulative_energy" else start_ts
    url = f"{HA_URL}/api/history/period/{query_start.isoformat()}"
    params = {
        "filter_entity_id": HA_ENTITY_ID,
        "end_time": end_ts.isoformat(),
        "significant_changes_only": "0",
        "minimal_response": "0",
    }
    payload = _http_get_json(url, params, {"Authorization": f"Bearer {HA_TOKEN}"})
    if not isinstance(payload, list) or not payload or not payload[0]:
        raise HTTPException(status_code=400, detail="Home Assistant returned no history for the configured entity and time range.")

    rows = []
    for item in payload[0]:
        try:
            value = float(item.get("state"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value) or value < 0:
            continue
        timestamp = pd.to_datetime(item.get("last_changed") or item.get("last_updated"), errors="coerce", utc=True)
        if pd.isna(timestamp):
            continue
        unit = item.get("attributes", {}).get("unit_of_measurement", HA_ENERGY_UNIT)
        if unit == "Wh":
            value /= 1000.0
        elif unit != "kWh":
            raise HTTPException(status_code=400, detail=f"Unsupported HA unit '{unit}'. Expected Wh or kWh; check HA_ENERGY_UNIT and entity attributes.")
        column = "cumulative_kwh" if HA_FEED_MODE == "cumulative_energy" else "energy_kwh"
        rows.append({"timestamp": timestamp, column: value})

    if not rows:
        raise HTTPException(status_code=400, detail="No numeric energy values found in Home Assistant history.")
    frame = pd.DataFrame(rows).sort_values("timestamp")

    if HA_FEED_MODE == "cumulative_energy":
        hourly, _ = _cumulative_to_hourly(frame)
    else:
        frame = frame.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
        hourly = frame["energy_kwh"].resample("h").last().to_frame("target").reset_index()
        hourly = hourly.dropna(subset=["target"])

    # Keep only full hourly bins in the requested interval. Internal timestamps
    # are UTC-aware; Open-Meteo is also requested in UTC for a reliable join.
    hourly = hourly[(hourly["timestamp"] >= start_ts.ceil("h")) & (hourly["timestamp"] < end_ts.floor("h"))]
    return hourly[["timestamp", "target"]].reset_index(drop=True)


def _fetch_historical_weather(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    params = _weather_params()
    params.update({
        "start_date": (start_ts.date() - timedelta(days=1)).isoformat(),
        "end_date": (end_ts.date() + timedelta(days=1)).isoformat(),
    })
    return _weather_frame(_http_get_json(ARCHIVE_URL, params))


def _load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=409, detail="No trained model found. Call POST /train first.")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load trained model: {exc}") from exc


@app.get("/",  include_in_schema=False)
def api_docs():
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.exists(),
        "home_assistant_configured": bool(HA_URL and HA_TOKEN and HA_ENTITY_ID),
        "ha_entity_id": HA_ENTITY_ID or None,
        "panel_capacity_kwp": PANEL_CAPACITY_KWP,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
    }


@app.post("/train")
def train(request: TrainRequest) -> dict[str, Any]:
    if lgb is None:
        raise HTTPException(status_code=500, detail="lightgbm is not installed; install requirements.txt")
    _ensure_dirs()
    end = request.end or datetime.now()
    start = request.start or (end - timedelta(days=730))
    production = _get_home_assistant_history(start, end)
    if len(production) < request.min_rows:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(production)} hourly energy rows found; at least {request.min_rows} required. Check long-term statistics availability (cumulative_energy mode), the requested history range, and sensor statistic ID.",
        )
    weather = _add_time_features(_fetch_historical_weather(production["timestamp"].min(), production["timestamp"].max()))
    merged = production.merge(weather, on="timestamp", how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=FEATURES + ["target"])
    
    # Exclude nighttime production readings from model training.
    # Open-Meteo shortwave radiation is measured in W/m².
    NIGHT_RADIATION_THRESHOLD = 5.0
    merged = merged[
        merged["shortwave_radiation"] > NIGHT_RADIATION_THRESHOLD
    ].copy()
    
    if len(merged) < max(48, request.min_rows // 2):
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough daylight production rows remain for training. "
                "Check the historical weather data and requested training period."
            ),
        )
    if len(merged) < max(48, request.min_rows // 2):
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(merged)} production rows matched hourly weather. Check timezone, recorder history, and sensor update semantics.",
        )

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    split = int(len(merged) * 0.75)
    train_df, test_df = merged.iloc[:split], merged.iloc[split:]
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=3600, learning_rate=0.03,
        num_leaves=63, max_depth=-1, min_child_samples=60, subsample=0.85,
        subsample_freq=1, boosting="dart",
        colsample_bytree=0.9, reg_lambda=1.0, random_state=42, verbosity=-1,
    )
    model.fit(
        train_df[FEATURES], train_df["target"],
        eval_set=[(test_df[FEATURES], test_df["target"])],
        callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
    )
    pred = np.maximum(0, model.predict(test_df[FEATURES]))
    actual = test_df["target"].to_numpy()
    metrics = {
        "rows_used": int(len(merged)),
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(test_df)),
        "mae_kwh_per_hour": float(mean_absolute_error(actual, pred)),
        "rmse_kwh_per_hour": float(np.sqrt(mean_squared_error(actual, pred))),
        "median_absolute_percentage_error_per_hour": float(np.median(np.abs((actual - pred)/(actual+1)))*100),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or model.n_estimators),
        "validation_period_start": str(test_df["timestamp"].min()),
        "validation_period_end": str(test_df["timestamp"].max()),
        "target": "kWh per hourly interval",
        "panel_capacity_kwp": PANEL_CAPACITY_KWP,
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return {"status": "trained", "model_path": str(MODEL_PATH), "metrics": metrics}


@app.post("/predict/forecast")
def predict_forecast(request: ForecastRequest) -> dict[str, Any]:
    bundle = _load_model()
    now = pd.Timestamp.now(tz="UTC").floor("h")
    start = pd.Timestamp(request.start) if request.start else now
    if start.tzinfo is None:
        start = start.tz_localize(TIMEZONE).tz_convert("UTC")
    else:
        start = start.tz_convert("UTC")
    start = start.floor("h")
    end = start + pd.Timedelta(hours=request.hours - 1)
    params = _weather_params()
    params["forecast_days"] = 8  # 168 hours from the current time can span 8 calendar dates
    weather = _add_time_features(_weather_frame(_http_get_json(FORECAST_URL, params)))
    selected = weather[(weather["timestamp"] >= start) & (weather["timestamp"] <= end)].copy()
    if len(selected) < request.hours:
        raise HTTPException(
            status_code=502,
            detail=f"Open-Meteo returned {len(selected)} hours for a {request.hours}-hour request. The selected range may extend beyond its forecast window.",
        )
    selected["predicted_energy_kwh"] = np.maximum(
        0, bundle["model"].predict(selected[FEATURES])
    )
    
    # Force production to zero when solar radiation is negligible.
    selected.loc[
        selected["shortwave_radiation"] <= 5.0,
        "predicted_energy_kwh",
    ] = 0.0
    # Energy must be non-negative and a single hourly interval cannot exceed panel kWp
    # by a physically meaningful margin. This cap is a guardrail, not a PV model.
    selected["predicted_energy_kwh"] = selected["predicted_energy_kwh"].clip(lower=0, upper=PANEL_CAPACITY_KWP)
    selected["predicted_energy_kwh"] = selected["predicted_energy_kwh"].round(3)
    records = []
    for row in selected.itertuples(index=False):
        records.append({
            "timestamp": pd.Timestamp(row.timestamp).isoformat(),
            "predicted_energy_kwh": float(row.predicted_energy_kwh),
        })
    local_dates = selected["timestamp"].dt.tz_convert(TIMEZONE).dt.date.astype(str)
    daily = selected.assign(local_date=local_dates).groupby("local_date")["predicted_energy_kwh"].sum().round(3).to_dict()
    return {
        "location": {"latitude": LATITUDE, "longitude": LONGITUDE, "timezone": TIMEZONE},
        "panel_capacity_kwp": PANEL_CAPACITY_KWP,
        "forecast_start": selected["timestamp"].min().isoformat(),
        "forecast_end": selected["timestamp"].max().isoformat(),
        "hours_returned": len(records),
        "predicted_energy_total_kwh": round(float(selected["predicted_energy_kwh"].sum()), 3),
        "daily_energy_kwh": daily,
        "energy_unit_note": "Predictions are kWh per hourly interval; daily and total values are sums of hourly interval energy.",
        "model_metrics": bundle.get("metrics", {}),
        "forecast": records,
    }


@app.get("/predict/forecast.png", response_class=StreamingResponse)
def predict_forecast_png(
    start: datetime | None = None,
    hours: int = 168,
) -> StreamingResponse:
    """Generate an inline PNG chart for the requested forecast."""
    request = ForecastRequest(start=start, hours=hours)
    forecast_data = predict_forecast(request)
    forecast = pd.DataFrame(forecast_data["forecast"])
    if forecast.empty:
        raise HTTPException(status_code=502, detail="The forecast contains no data to plot.")

    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    forecast["local_date"] = forecast["timestamp"].dt.strftime("%Y-%m-%d")
    daily = forecast.groupby("local_date", sort=True)["predicted_energy_kwh"].sum()

    # A two-panel chart keeps hourly detail readable while showing daily totals.
    fig, (ax_hourly, ax_daily) = plt.subplots(
        2, 1, figsize=(12, 7.5), gridspec_kw={"height_ratios": [2.4, 1]},
        constrained_layout=True,
    )

    times = forecast["timestamp"]
    values = forecast["predicted_energy_kwh"]
    ax_hourly.bar(times, values, width=0.035, align="center", color="#8b5cf6")
    ax_hourly.set_ylabel("Energy per hour (kWh)")
    ax_hourly.set_xlim(times.min() - pd.Timedelta(minutes=35),
                       times.max() + pd.Timedelta(minutes=35))
    ax_hourly.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax_hourly.grid(axis="y", alpha=0.25)
    ax_hourly.set_axisbelow(True)

    daily_dates = pd.to_datetime(daily.index)
    bars = ax_daily.bar(daily_dates, daily.values, color="#ec78b7", width=0.65)
    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax_daily.set_ylabel("Daily total (kWh)")
    ax_daily.grid(axis="y", alpha=0.25)
    ax_daily.set_axisbelow(True)
    ax_daily.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax_daily.margins(y=0.18)

    image_buffer = BytesIO()
    fig.savefig(image_buffer, format="png", dpi=160, facecolor="white")
    plt.close(fig)
    image_buffer.seek(0)
    filename = f"solar-forecast-{forecast_data['forecast_start'][:10]}.png"
    return StreamingResponse(
        image_buffer,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return _load_model().get("metrics", {})
