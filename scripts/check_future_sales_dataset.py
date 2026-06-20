from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/raw/sales_train.csv")


def main() -> None:
  if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Required dataset file not found: {DATASET_PATH.resolve()}"
    )

  df = pd.read_csv(DATASET_PATH, parse_dates=["date"], dayfirst=True)

  print(f"Dataset shape: {df.shape}")
  print(f"Columns: {list(df.columns)}")
  print("First 5 rows:")
  print(df.head())
  print("Missing value counts:")
  print(df.isna().sum())
  print(f"Min date: {df['date'].min()}")
  print(f"Max date: {df['date'].max()}")
  print(f"Unique shop_id values: {df['shop_id'].nunique()}")
  print(f"Unique item_id values: {df['item_id'].nunique()}")
  print("Basic statistics for item_cnt_day and item_price:")
  print(df[["item_cnt_day", "item_price"]].describe())


if __name__ == "__main__":
  main()
