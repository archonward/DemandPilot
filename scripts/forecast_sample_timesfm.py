from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/sample/sample_sales_series.csv")
OUTPUT_PATH = Path("data/processed/sample_forecast.csv")
HORIZON = 30
REQUIRED_COLUMNS = ["date", "series_id", "value"]

IDX_Q10 = 1
IDX_Q20 = 2
IDX_Q50 = 5
IDX_Q80 = 8
IDX_Q90 = 9


def ensure_local_src_on_path() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  src_path = repo_root / "src"
  if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def load_sample_series() -> pd.DataFrame:
  if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Required sample CSV not found: {INPUT_PATH.resolve()}"
    )

  df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
  missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
  if missing_columns:
    raise ValueError(
        f"Sample CSV is missing required columns {missing_columns}. "
        f"Found columns: {list(df.columns)}"
    )

  df = df.sort_values("date").reset_index(drop=True)
  if df.empty:
    raise ValueError(f"Sample CSV is empty: {INPUT_PATH.resolve()}")

  return df


def load_model():
  ensure_local_src_on_path()

  try:
    import torch
    import timesfm
  except ImportError as exc:
    raise ImportError(
        "Failed to import TimesFM dependencies. Install the project dependencies "
        "and TimesFM torch extras before running this script."
    ) from exc

  torch.set_float32_matmul_precision("high")

  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
      "google/timesfm-2.5-200m-pytorch"
  )
  model.compile(
      timesfm.ForecastConfig(
          max_context=1024,
          max_horizon=256,
          normalize_inputs=True,
          use_continuous_quantile_head=True,
          force_flip_invariance=True,
          infer_is_positive=True,
          fix_quantile_crossing=True,
          per_core_batch_size=1,
      )
  )
  return model


def main() -> None:
  df = load_sample_series()
  values = df["value"].to_numpy(dtype=np.float32)
  series_id = str(df["series_id"].iloc[0])
  last_date = df["date"].max()

  model = load_model()
  point_forecast, quantile_forecast = model.forecast(
      horizon=HORIZON,
      inputs=[values],
  )

  future_dates = pd.date_range(
      start=last_date + pd.Timedelta(days=1),
      periods=HORIZON,
      freq="D",
  )

  forecast_df = pd.DataFrame(
      {
          "date": future_dates,
          "series_id": series_id,
          "forecast": point_forecast[0].astype(np.float32),
          "lower_80": quantile_forecast[0, :, IDX_Q20].astype(np.float32),
          "median": quantile_forecast[0, :, IDX_Q50].astype(np.float32),
          "upper_80": quantile_forecast[0, :, IDX_Q80].astype(np.float32),
          "lower_90": quantile_forecast[0, :, IDX_Q10].astype(np.float32),
          "upper_90": quantile_forecast[0, :, IDX_Q90].astype(np.float32),
      }
  )

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  forecast_df.to_csv(OUTPUT_PATH, index=False)

  print(f"Input path: {INPUT_PATH}")
  print(f"Selected series_id: {series_id}")
  print(f"Number of historical rows: {len(df)}")
  print(f"Historical date range: {df['date'].min()} to {df['date'].max()}")
  print(f"Output path: {OUTPUT_PATH}")
  print("First 5 forecast rows:")
  print(forecast_df.head())


if __name__ == "__main__":
  main()
