# Solar Forecast API — Home Assistant edition

FastAPI + LightGBM for a 9.8 kWp PV installation. Reads cumulative solar energy from Home Assistant, derives hourly production, aligns it with Open-Meteo historical weather, trains a LightGBM regressor, and returns a 168-hour forecast in kWh per hour.

## Assumptions to verify

- Panel capacity: 9.8 kWp.
- Time zone: `Europe/Amsterdam`.
- Home Assistant entity: configured as `HA_ENTITY_ID`.
- Default mode is `HA_FEED_MODE=cumulative_energy`: the HA entity is a monotonically increasing cumulative energy meter in kWh or Wh. The code differences consecutive readings and bins the increments into hourly kWh. Set `HA_FEED_MODE=interval_energy` only if your entity already reports energy per interval.
- Training requests up to two years of production history by default. In the default `cumulative_energy` mode, it reads Home Assistant's hourly long-term statistics over the WebSocket API, so it does not require the Recorder to retain two years of raw state history. Long-term statistics must exist for the configured statistic ID. `interval_energy` mode still uses the REST history endpoint and is subject to Recorder retention.
- The default latitude/longitude are examples only. Set them to the PV installation's actual coordinates.

## Setup

1. Copy `.env.example` to `.env`.
2. Set `LATITUDE`, `LONGITUDE`, `HA_URL`, `HA_TOKEN`, and `HA_ENTITY_ID`.
3. Create a Home Assistant long-lived access token in your HA profile. Treat it as a password and never commit `.env`.
4. Set `HA_ENTITY_ID` to the Home Assistant statistic ID for your solar-energy sensor (normally the entity ID, such as `sensor.solar_energy_total`). In the default `cumulative_energy` mode, the app connects to `HA_URL/api/websocket` and calls `recorder/statistics_during_period` for hourly long-term statistics. It differences consecutive `sum` values to calculate kWh produced per hour; Home Assistant's statistics machinery handles cumulative-counter resets for supported `total_increasing` sensors. `HA_ENERGY_UNIT` is used by the REST-history path in `interval_energy` mode. In Home Assistant, check **Developer tools → Statistics** to confirm the statistic ID exists and has no repair warning. The entity should have an energy device class and a supported state class (typically `total_increasing`). Long-term statistics are hourly, so original few-minute readings are not retrieved.
5. Start the service:

```bash
docker compose up -d --build
docker compose logs -f solar-forecast
```

Open `http://localhost:8000/docs`.

## Train

Default: last 730 days, requiring at least 500 matched hourly rows.

```bash
curl -X POST http://localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{"min_rows": 500}'
```

Or specify the date range:

```json
{
  "start": "2025-01-01T00:00:00+01:00",
  "end": "2026-10-25T00:00:00+02:00",
  "min_rows": 500
}
```

Use that JSON as the body to `POST /train`. Historical weather is downloaded from Open-Meteo's archive endpoint; the model is saved to `models/lightgbm_model.joblib`.

## Predict the next 7 days

```bash
curl -X POST http://localhost:8000/predict/forecast \
  -H 'Content-Type: application/json' \
  -d '{"hours": 168}'
```

The service requests 8 calendar days of hourly weather so a full 168 hours starting now can be returned; the result contains each hour's `predicted_energy_kwh`, daily sums, and a 7-day total. The hourly energy values are constrained to non-negative values and capped at 9.8 kWh per one-hour interval as a basic guardrail. This cap is not a substitute for an inverter model.

## API endpoints

- `GET /health`
- `POST /train`
- `POST /predict/forecast`
- `GET /metrics`
- `GET /docs`

## Home Assistant connection troubleshooting

- `HA_URL` must be reachable from the Docker container. On Home Assistant OS, the correct hostname/IP depends on where this container runs; `homeassistant.local` may not resolve in every Docker network.
- `401` or `403`: regenerate the token and check the URL.
- No statistics: check `HA_ENTITY_ID` against the statistic ID shown under Home Assistant **Developer tools → Statistics**. Confirm long-term statistics exist for the selected period and the entity has a supported device/state class. This is separate from Recorder's 10-day raw-history retention.
- Too few matched rows: check that the entity is hourly interval energy and that timestamps align with `Europe/Amsterdam`.
- `Wh` is converted to `kWh`; other units are rejected.
- In `cumulative_energy` mode, long-term statistics provide hourly values, not the original state changes. The application differences cumulative `sum` values and filters negative corrections and implausible hourly jumps. Review application logs for skipped increments and check the source statistic if many rows are skipped. Two years of statistics are requested by default, subject to the dates for which Home Assistant has statistics.

## Cumulative meter processing

- The service fetches an extra two hours before the requested training window to obtain a baseline for the first difference. The WebSocket statistics API is a Home Assistant API and may vary across HA versions; inspect application logs if a statistics request fails.
- Hourly targets are kWh generated during each UTC hour. Weather and production are joined in UTC; local calendar/time features use `Europe/Amsterdam`. This avoids the repeated local hour during the autumn daylight-saving transition.
- A long positive increment cannot be precisely allocated to its true production hour using cumulative readings alone. The code assigns it to the ending hour and emits a warning. For best results, investigate these warnings and exclude periods with HA downtime.
- The 9.8 kWp plausibility check is a guardrail, not a complete PV/inverter model.

## Notes on model quality

The model uses temperature, humidity, cloud cover, shortwave/direct/diffuse radiation, wind, precipitation, daylight flag, and time-of-day/season features. It validates on the latest 20% of the matched time series instead of random splitting. Use the returned MAE/RMSE and compare forecasts against measured energy across seasons before operational use.

For best validation, use archived forecast runs rather than only reanalysis observations, because future forecasts are not identical to historical weather analysis. The starter is designed to be extended with explicit panel tilt/azimuth, inverter clipping, shading, curtailment detection, and a rolling backtest.
