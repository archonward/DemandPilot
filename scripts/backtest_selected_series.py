from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/future_sales_daily.csv")
OUTPUT_DIR = Path("data/processed/backtests")
REQUIRED_COLUMNS = [
    "date",
    "series_id",
    "shop_id",
    "item_id",
    "value",
    "avg_item_price",
]

IDX_Q20 = 2
IDX_Q50 = 5
IDX_Q80 = 8
MIN_HISTORY_ROWS = 180
MAX_HORIZON = 256
MOVING_AVERAGE_WINDOW = 30
SEASONAL_LAG = 7


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Backtest a selected shop/item series with TimesFM and baseline "
          "forecasts."
      )
  )
  parser.add_argument("--shop-id", type=int, required=True)
  parser.add_argument("--item-id", type=int, required=True)
  parser.add_argument("--horizon", type=int, required=True)
  return parser.parse_args()


def ensure_local_src_on_path() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  src_path = repo_root / "src"
  if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def validate_horizon(horizon: int) -> None:
  if horizon <= 0:
    raise ValueError("horizon must be a positive integer")
  if horizon > MAX_HORIZON:
    raise ValueError(
        f"horizon must be less than or equal to {MAX_HORIZON} "
        f"for the configured TimesFM model. Received: {horizon}"
    )


def load_processed_dataset() -> pd.DataFrame:
  if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Processed dataset not found: "
        f"{INPUT_PATH.resolve()}. Expected file: data/processed/future_sales_daily.csv"
    )

  df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
  missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
  if missing_columns:
    raise ValueError(
        f"Processed dataset is missing required columns {missing_columns}. "
        f"Found columns: {list(df.columns)}"
    )
  return df


def select_series(df: pd.DataFrame, shop_id: int, item_id: int) -> pd.DataFrame:
  selected = df.loc[
      (df["shop_id"] == shop_id) & (df["item_id"] == item_id)
  ].copy()
  if selected.empty:
    raise ValueError(f"No data found for shop_id={shop_id}, item_id={item_id}")

  return selected.sort_values("date").reset_index(drop=True)


def validate_series_length(
    selected: pd.DataFrame,
    shop_id: int,
    item_id: int,
    horizon: int,
) -> None:
  required_rows = MIN_HISTORY_ROWS + horizon
  historical_rows = len(selected)
  if historical_rows < required_rows:
    raise ValueError(
        f"Series shop_{shop_id}*item*{item_id} has only {historical_rows} "
        f"historical rows. At least {required_rows} rows are required for a "
        f"{horizon}-day backtest."
    )


def split_backtest_frames(
    selected: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  context_df = selected.iloc[:-horizon].copy().reset_index(drop=True)
  actual_df = selected.iloc[-horizon:].copy().reset_index(drop=True)
  return context_df, actual_df


def load_model():
  ensure_local_src_on_path()

  try:
    import torch
    import timesfm
  except ImportError as exc:
    raise ImportError(
        "Failed to import TimesFM dependencies. Install the project "
        "dependencies and TimesFM torch extras before running this script."
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


def build_naive_forecast(context_values: np.ndarray, horizon: int) -> np.ndarray:
  return np.repeat(context_values[-1], horizon).astype(np.float32)


def build_moving_average_forecast(
    context_values: np.ndarray,
    horizon: int,
) -> np.ndarray:
  moving_average = context_values[-MOVING_AVERAGE_WINDOW:].mean()
  return np.repeat(moving_average, horizon).astype(np.float32)


def build_seasonal_naive_forecast(
    context_values: np.ndarray,
    horizon: int,
) -> np.ndarray:
  naive_value = context_values[-1]
  forecast = np.empty(horizon, dtype=np.float32)

  for step in range(horizon):
    lag_index = len(context_values) - SEASONAL_LAG + step
    if 0 <= lag_index < len(context_values):
      forecast[step] = context_values[lag_index]
    else:
      forecast[step] = naive_value

  return forecast


def calculate_mae(actual: np.ndarray, forecast: np.ndarray) -> float:
  return float(np.mean(np.abs(forecast - actual)))


def calculate_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
  return float(math.sqrt(np.mean((forecast - actual) ** 2)))


def calculate_mape(actual: np.ndarray, forecast: np.ndarray) -> float:
  non_zero_mask = actual != 0
  if not np.any(non_zero_mask):
    return float("nan")

  return float(
      np.mean(
          np.abs(
              (forecast[non_zero_mask] - actual[non_zero_mask])
              / actual[non_zero_mask]
          )
      )
      * 100.0
  )


def calculate_bias(actual: np.ndarray, forecast: np.ndarray) -> float:
  return float(np.mean(forecast - actual))


def build_metrics_table(
    actual: np.ndarray,
    timesfm_forecast: np.ndarray,
    naive_forecast: np.ndarray,
    moving_average_forecast: np.ndarray,
    seasonal_naive_forecast: np.ndarray,
) -> pd.DataFrame:
  model_forecasts = [
      ("TimesFM", timesfm_forecast),
      ("Naive", naive_forecast),
      ("Moving Average 30", moving_average_forecast),
      ("Seasonal Naive 7", seasonal_naive_forecast),
  ]

  rows = []
  for model_name, forecast in model_forecasts:
    rows.append(
        {
            "model": model_name,
            "mae": calculate_mae(actual, forecast),
            "rmse": calculate_rmse(actual, forecast),
            "mape": calculate_mape(actual, forecast),
            "bias": calculate_bias(actual, forecast),
        }
    )

  return pd.DataFrame(rows, columns=["model", "mae", "rmse", "mape", "bias"])


def build_backtest_path(shop_id: int, item_id: int) -> Path:
  return OUTPUT_DIR / f"shop_{shop_id}_item_{item_id}_backtest.csv"


def build_metrics_path(shop_id: int, item_id: int) -> Path:
  return OUTPUT_DIR / f"shop_{shop_id}_item_{item_id}_metrics.csv"


def main() -> None:
  args = parse_args()
  validate_horizon(args.horizon)

  df = load_processed_dataset()
  selected = select_series(df, args.shop_id, args.item_id)
  validate_series_length(selected, args.shop_id, args.item_id, args.horizon)

  context_df, actual_df = split_backtest_frames(selected, args.horizon)
  context_values = context_df["value"].to_numpy(dtype=np.float32)
  actual_values = actual_df["value"].to_numpy(dtype=np.float32)
  series_id = str(selected["series_id"].iloc[0])

  model = load_model()
  point_forecast, quantile_forecast = model.forecast(
      horizon=args.horizon,
      inputs=[context_values],
  )

  timesfm_forecast = point_forecast[0].astype(np.float32)
  timesfm_lower_80 = quantile_forecast[0, :, IDX_Q20].astype(np.float32)
  timesfm_median = quantile_forecast[0, :, IDX_Q50].astype(np.float32)
  timesfm_upper_80 = quantile_forecast[0, :, IDX_Q80].astype(np.float32)

  naive_forecast = build_naive_forecast(context_values, args.horizon)
  moving_average_forecast = build_moving_average_forecast(
      context_values,
      args.horizon,
  )
  seasonal_naive_forecast = build_seasonal_naive_forecast(
      context_values,
      args.horizon,
  )

  backtest_df = pd.DataFrame(
      {
          "date": actual_df["date"].to_numpy(),
          "series_id": series_id,
          "shop_id": args.shop_id,
          "item_id": args.item_id,
          "actual": actual_values,
          "timesfm_forecast": timesfm_forecast,
          "timesfm_lower_80": timesfm_lower_80,
          "timesfm_median": timesfm_median,
          "timesfm_upper_80": timesfm_upper_80,
          "naive_forecast": naive_forecast,
          "moving_average_30_forecast": moving_average_forecast,
          "seasonal_naive_7_forecast": seasonal_naive_forecast,
      },
      columns=[
          "date",
          "series_id",
          "shop_id",
          "item_id",
          "actual",
          "timesfm_forecast",
          "timesfm_lower_80",
          "timesfm_median",
          "timesfm_upper_80",
          "naive_forecast",
          "moving_average_30_forecast",
          "seasonal_naive_7_forecast",
      ],
  )

  metrics_df = build_metrics_table(
      actual_values,
      timesfm_forecast,
      naive_forecast,
      moving_average_forecast,
      seasonal_naive_forecast,
  )

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  backtest_path = build_backtest_path(args.shop_id, args.item_id)
  metrics_path = build_metrics_path(args.shop_id, args.item_id)
  backtest_df.to_csv(backtest_path, index=False)
  metrics_df.to_csv(metrics_path, index=False)

  print(f"Selected shop_id: {args.shop_id}")
  print(f"Selected item_id: {args.item_id}")
  print(f"Selected series_id: {series_id}")
  print(f"Total historical rows: {len(selected)}")
  print(f"Context rows: {len(context_df)}")
  print(f"Backtest horizon: {args.horizon}")
  print(
      f"Context date range: {context_df['date'].min().date()} "
      f"to {context_df['date'].max().date()}"
  )
  print(
      f"Actual test date range: {actual_df['date'].min().date()} "
      f"to {actual_df['date'].max().date()}"
  )
  print(f"Output backtest path: {backtest_path}")
  print(f"Output metrics path: {metrics_path}")
  print("Metrics table:")
  print(metrics_df.to_string(index=False))


if __name__ == "__main__":
  main()
