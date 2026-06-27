from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/future_sales_daily.csv")
OUTPUT_DIR = Path("data/processed/forecasts")
REQUIRED_COLUMNS = [
    "date",
    "series_id",
    "shop_id",
    "item_id",
    "value",
    "avg_item_price",
]

IDX_Q10 = 1
IDX_Q20 = 2
IDX_Q50 = 5
IDX_Q80 = 8
IDX_Q90 = 9
MIN_RECOMMENDED_HISTORY = 30
MAX_HORIZON = 256


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generate a TimesFM forecast for a selected shop/item series."
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


def load_processed_dataset() -> pd.DataFrame:
  if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {INPUT_PATH.resolve()}"
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
    raise ValueError(
        f"No data found for shop_id={shop_id}, item_id={item_id}"
    )

  selected = selected.sort_values("date").reset_index(drop=True)
  return selected


def validate_horizon(horizon: int) -> None:
  if horizon <= 0:
    raise ValueError("horizon must be a positive integer")
  if horizon > MAX_HORIZON:
    raise ValueError(
        f"horizon must be less than or equal to {MAX_HORIZON} "
        f"for the configured TimesFM model. Received: {horizon}"
    )


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


def build_output_path(shop_id: int, item_id: int) -> Path:
  return OUTPUT_DIR / f"shop_{shop_id}_item_{item_id}_forecast.csv"


def main() -> None:
  args = parse_args()
  validate_horizon(args.horizon)

  df = load_processed_dataset()
  selected = select_series(df, args.shop_id, args.item_id)

  if len(selected) < MIN_RECOMMENDED_HISTORY:
    print(
        "Warning: selected series has fewer than 30 rows; "
        "forecast quality may be weak."
    )

  values = selected["value"].to_numpy(dtype=np.float32)
  series_id = str(selected["series_id"].iloc[0])
  last_date = selected["date"].iloc[-1]

  model = load_model()
  point_forecast, quantile_forecast = model.forecast(
      horizon=args.horizon,
      inputs=[values],
  )

  future_dates = pd.date_range(
      start=last_date + pd.Timedelta(days=1),
      periods=args.horizon,
      freq="D",
  )

  forecast_df = pd.DataFrame(
      {
          "date": future_dates,
          "series_id": series_id,
          "shop_id": args.shop_id,
          "item_id": args.item_id,
          "forecast": point_forecast[0].astype(np.float32),
          "lower_80": quantile_forecast[0, :, IDX_Q20].astype(np.float32),
          "median": quantile_forecast[0, :, IDX_Q50].astype(np.float32),
          "upper_80": quantile_forecast[0, :, IDX_Q80].astype(np.float32),
          "lower_90": quantile_forecast[0, :, IDX_Q10].astype(np.float32),
          "upper_90": quantile_forecast[0, :, IDX_Q90].astype(np.float32),
      },
      columns=[
          "date",
          "series_id",
          "shop_id",
          "item_id",
          "forecast",
          "lower_80",
          "median",
          "upper_80",
          "lower_90",
          "upper_90",
      ],
  )

  output_path = build_output_path(args.shop_id, args.item_id)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  forecast_df.to_csv(output_path, index=False)

  print(f"Selected shop_id: {args.shop_id}")
  print(f"Selected item_id: {args.item_id}")
  print(f"Selected series_id: {series_id}")
  print(f"Number of historical rows: {len(selected)}")
  print(
      f"Historical date range: {selected['date'].min().date()} "
      f"to {selected['date'].max().date()}"
  )
  print(f"Forecast horizon: {args.horizon}")
  print(f"Output path: {output_path}")
  print("First 5 forecast rows:")
  print(forecast_df.head().to_string(index=False))


if __name__ == "__main__":
  main()
