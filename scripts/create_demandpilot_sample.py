from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/sales_train.csv")
OUTPUT_PATH = Path("data/sample/sample_sales_series.csv")
PREFERRED_SHOP_ID = 31


def load_sales() -> pd.DataFrame:
  if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Required dataset file not found: {INPUT_PATH.resolve()}")

  df = pd.read_csv(INPUT_PATH, parse_dates=["date"], dayfirst=True)
  daily_sales = (
      df.groupby(["date", "shop_id", "item_id"], as_index=False)
      .agg(item_cnt_day=("item_cnt_day", "sum"))
      .rename(columns={"item_cnt_day": "value"})
  )
  return daily_sales


def pick_series_id(daily_sales: pd.DataFrame) -> tuple[int, int]:
  available_shops = set(daily_sales["shop_id"].unique())
  shop_id = PREFERRED_SHOP_ID if PREFERRED_SHOP_ID in available_shops else int(
      daily_sales["shop_id"].value_counts().idxmax()
  )

  shop_sales = daily_sales[daily_sales["shop_id"] == shop_id]
  item_id = int(shop_sales["item_id"].value_counts().idxmax())
  return shop_id, item_id


def main() -> None:
  daily_sales = load_sales()
  shop_id, item_id = pick_series_id(daily_sales)

  sample = daily_sales[
      (daily_sales["shop_id"] == shop_id) & (daily_sales["item_id"] == item_id)
  ].copy()
  sample["series_id"] = f"shop_{shop_id}_item_{item_id}"
  sample = sample.sort_values("date")[["date", "series_id", "value"]]

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  sample.to_csv(OUTPUT_PATH, index=False)

  print(f"Selected shop_id: {shop_id}")
  print(f"Selected item_id: {item_id}")
  print(f"Selected series_id: shop_{shop_id}_item_{item_id}")
  print(f"Row count: {len(sample)}")
  print(f"Date range: {sample['date'].min()} to {sample['date'].max()}")
  print("First 5 rows:")
  print(sample.head())


if __name__ == "__main__":
  main()
