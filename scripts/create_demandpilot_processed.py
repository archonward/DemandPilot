from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/sales_train.csv")
OUTPUT_PATH = Path("data/processed/future_sales_daily.csv")


def main() -> None:
  if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Required dataset file not found: {INPUT_PATH.resolve()}")

  df = pd.read_csv(INPUT_PATH, parse_dates=["date"], dayfirst=True)

  processed = (
      df.groupby(["date", "shop_id", "item_id"], as_index=False)
      .agg(
          value=("item_cnt_day", "sum"),
          avg_item_price=("item_price", "mean"),
      )
      .sort_values(["date", "shop_id", "item_id"])
  )
  processed["series_id"] = (
      "shop_"
      + processed["shop_id"].astype(str)
      + "_item_"
      + processed["item_id"].astype(str)
  )
  processed = processed[
      ["date", "series_id", "shop_id", "item_id", "value", "avg_item_price"]
  ]

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  processed.to_csv(OUTPUT_PATH, index=False)

  print(f"Output path: {OUTPUT_PATH}")
  print(f"Row count: {len(processed)}")
  print(f"Number of series: {processed['series_id'].nunique()}")
  print(f"Date range: {processed['date'].min()} to {processed['date'].max()}")
  print("First 5 rows:")
  print(processed.head())


if __name__ == "__main__":
  main()
