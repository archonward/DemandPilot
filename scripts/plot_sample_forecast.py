from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


HISTORICAL_PATH = Path("data/sample/sample_sales_series.csv")
FORECAST_PATH = Path("data/processed/sample_forecast.csv")
OUTPUT_PATH = Path("outputs/figures/sample_forecast.png")
HISTORICAL_WINDOW_DAYS = 180


def load_csv(path: Path, label: str) -> pd.DataFrame:
  if not path.exists():
    raise FileNotFoundError(f"Required {label} file not found: {path.resolve()}")
  return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(
      drop=True
  )


def main() -> None:
  historical_df = load_csv(HISTORICAL_PATH, "historical sample")
  forecast_df = load_csv(FORECAST_PATH, "forecast")

  series_id = str(historical_df["series_id"].iloc[0])
  historical_plot_df = historical_df.tail(HISTORICAL_WINDOW_DAYS).copy()

  fig, ax = plt.subplots(figsize=(12, 6))

  ax.plot(
      historical_plot_df["date"],
      historical_plot_df["value"],
      label="Historical sales",
      color="tab:blue",
      linewidth=1.8,
  )
  ax.plot(
      forecast_df["date"],
      forecast_df["forecast"],
      label="Forecast",
      color="tab:orange",
      linewidth=2.0,
  )
  ax.fill_between(
      forecast_df["date"],
      forecast_df["lower_90"],
      forecast_df["upper_90"],
      label="90% prediction interval",
      color="tab:orange",
      alpha=0.15,
  )
  ax.fill_between(
      forecast_df["date"],
      forecast_df["lower_80"],
      forecast_df["upper_80"],
      label="80% prediction interval",
      color="tab:orange",
      alpha=0.3,
  )

  ax.set_title(f"Demand Forecast for {series_id}")
  ax.set_xlabel("Date")
  ax.set_ylabel("Sales quantity")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUTPUT_PATH, dpi=150)
  plt.close(fig)

  print(f"Historical input path: {HISTORICAL_PATH}")
  print(f"Forecast input path: {FORECAST_PATH}")
  print(f"Selected series_id: {series_id}")
  print(f"Output chart path: {OUTPUT_PATH}")
  print(f"Number of historical rows plotted: {len(historical_plot_df)}")
  print(f"Number of forecast rows plotted: {len(forecast_df)}")


if __name__ == "__main__":
  main()
