from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROCESSED_HISTORY_PATH = Path("data/processed/future_sales_daily.csv")
FORECASTS_DIR = Path("data/processed/forecasts")
SAMPLE_HISTORICAL_PATH = Path("data/sample/sample_sales_series.csv")
SAMPLE_FORECAST_PATH = Path("data/processed/sample_forecast.csv")
HISTORICAL_WINDOW_DAYS = 180
MIN_HISTORY_ROWS = 180


st.set_page_config(
    page_title="DemandPilot: Retail Demand Forecasting",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_processed_history(path_str: str) -> pd.DataFrame:
  path = Path(path_str)
  return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(
      drop=True
  )


@st.cache_data(show_spinner=False)
def load_forecast_csv(path_str: str) -> pd.DataFrame:
  path = Path(path_str)
  return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(
      drop=True
  )


def load_csv(path: Path, label: str) -> pd.DataFrame | None:
  if not path.exists():
    st.error(f"Missing {label} CSV: `{path}`")
    return None

  return load_forecast_csv(str(path))


def format_date_range(df: pd.DataFrame) -> str:
  return (
      f"{df['date'].min().date().isoformat()} to "
      f"{df['date'].max().date().isoformat()}"
  )


def format_units(value: int) -> str:
  return f"{value:,} units"


def build_series_counts(processed_history_df: pd.DataFrame) -> pd.DataFrame:
  counts_df = (
      processed_history_df.groupby(["series_id", "shop_id", "item_id"], as_index=False)
      .size()
      .rename(columns={"size": "historical_rows"})
  )
  counts_df["is_forecastable"] = (
      counts_df["historical_rows"] >= MIN_HISTORY_ROWS
  )
  return counts_df


def build_selected_forecast_glob(shop_id: int, item_id: int) -> str:
  return f"shop_{shop_id}*item*{item_id}_forecast.csv"


def find_selected_forecast_path(shop_id: int, item_id: int) -> Path | None:
  pattern = build_selected_forecast_glob(shop_id, item_id)
  matches = sorted(FORECASTS_DIR.glob(pattern))
  if not matches:
    return None
  return matches[0]


def build_history_chart(historical_df: pd.DataFrame) -> go.Figure:
  history_plot_df = historical_df.tail(HISTORICAL_WINDOW_DAYS).copy()

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=history_plot_df["date"],
          y=history_plot_df["value"],
          mode="lines",
          name="Historical demand",
          line={"color": "#1f77b4", "width": 2},
      )
  )
  fig.update_layout(
      title="Historical Demand for Selected Series",
      xaxis_title="Date",
      yaxis_title="Sales quantity",
      hovermode="x unified",
      height=450,
      margin={"l": 20, "r": 20, "t": 60, "b": 20},
  )
  return fig


def build_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> go.Figure:
  historical_plot_df = historical_df.tail(HISTORICAL_WINDOW_DAYS).copy()

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=historical_plot_df["date"],
          y=historical_plot_df["value"],
          mode="lines",
          name="Historical sales",
          line={"color": "#1f77b4", "width": 2},
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["upper_90"],
          mode="lines",
          line={"width": 0},
          hoverinfo="skip",
          showlegend=False,
          name="90% upper",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["lower_90"],
          mode="lines",
          line={"width": 0},
          fill="tonexty",
          fillcolor="rgba(255, 127, 14, 0.15)",
          name="90% prediction interval",
          hovertemplate="90% interval<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["upper_80"],
          mode="lines",
          line={"width": 0},
          hoverinfo="skip",
          showlegend=False,
          name="80% upper",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["lower_80"],
          mode="lines",
          line={"width": 0},
          fill="tonexty",
          fillcolor="rgba(255, 127, 14, 0.30)",
          name="80% prediction interval",
          hovertemplate="80% interval<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["forecast"],
          mode="lines",
          name="Forecast",
          line={"color": "#ff7f0e", "width": 3},
      )
  )

  fig.update_layout(
      title="Historical Sales and TimesFM Forecast for Selected Series",
      xaxis_title="Date",
      yaxis_title="Sales quantity",
      hovermode="x unified",
      legend_title_text="Series",
      height=520,
      margin={"l": 20, "r": 20, "t": 60, "b": 20},
  )

  return fig


def main() -> None:
  st.title("DemandPilot: Retail Demand Forecasting")
  st.caption(
      "This dashboard uses TimesFM-generated forecasts to help estimate "
      "future product demand."
  )

  with st.sidebar:
    st.header("Instructions")
    st.caption(
      "Current status: dashboard with product-store historical exploration "
      "and selected-series forecast display when generated."
    )
    st.write("Step 1: Prepare dataset")
    st.write("Step 2: Run TimesFM forecast script")
    st.write("Step 3: View forecast in dashboard")

  if not PROCESSED_HISTORY_PATH.exists():
    st.error(f"Missing processed historical CSV: `{PROCESSED_HISTORY_PATH}`")
    return

  processed_history_df = load_processed_history(str(PROCESSED_HISTORY_PATH))
  series_counts_df = build_series_counts(processed_history_df)

  show_only_forecastable = st.sidebar.checkbox(
      "Show only forecastable series",
      value=True,
  )

  available_series_df = series_counts_df
  if show_only_forecastable:
    available_series_df = series_counts_df.loc[
        series_counts_df["is_forecastable"]
    ].copy()

  if available_series_df.empty:
    st.error(
        "No series are available for the current forecastability filter."
    )
    return

  shop_options = sorted(available_series_df["shop_id"].dropna().unique().tolist())
  selected_shop_id = st.sidebar.selectbox("shop_id", shop_options)

  item_options = sorted(
      available_series_df.loc[
          available_series_df["shop_id"] == selected_shop_id, "item_id"
      ]
      .dropna()
      .unique()
      .tolist()
  )
  if not item_options:
    st.error("No item_id values are available for the selected shop_id.")
    return
  selected_item_id = st.sidebar.selectbox("item_id", item_options)

  historical_df = processed_history_df.loc[
      (processed_history_df["shop_id"] == selected_shop_id)
      & (processed_history_df["item_id"] == selected_item_id)
  ].copy()

  if historical_df.empty:
    st.error(
        "No historical rows found for the selected shop_id and item_id "
        "combination."
    )
    return

  historical_df = historical_df.sort_values("date").reset_index(drop=True)
  series_id = str(historical_df["series_id"].iloc[0])
  historical_rows = len(historical_df)
  is_forecastable = historical_rows >= MIN_HISTORY_ROWS
  coverage_status = "Forecastable" if is_forecastable else "Limited history"

  st.subheader("Selected product-store series")
  st.write(series_id)
  st.write(f"Historical data: {format_date_range(historical_df)}")

  if not show_only_forecastable and not is_forecastable:
    st.warning(
        "This series has limited history and is not recommended for forecasting."
    )

  top_metrics = st.columns(3)
  top_metrics[0].metric("Historical rows", f"{historical_rows:,}")
  top_metrics[1].metric(
      "Average historical daily sales",
      f"{historical_df['value'].mean():.2f}",
  )
  top_metrics[2].metric("Max daily sales", f"{historical_df['value'].max():.0f}")

  lower_metrics = st.columns(3)
  lower_metrics[0].metric(
      "Total historical sales",
      f"{historical_df['value'].sum():.0f}",
  )
  lower_metrics[1].metric(
      "Average item price",
      f"{historical_df['avg_item_price'].mean():.2f}",
  )
  lower_metrics[2].metric(
      "Historical coverage status",
      coverage_status,
  )

  st.subheader("Historical Demand")
  st.caption("The chart shows the most recent 180 rows of historical demand.")
  history_chart = build_history_chart(historical_df)
  st.plotly_chart(history_chart, use_container_width=True)

  selected_forecast_path = find_selected_forecast_path(
      selected_shop_id, selected_item_id
  )
  selected_forecast_df: pd.DataFrame | None = None

  if selected_forecast_path is not None:
    selected_forecast_df = load_csv(
        selected_forecast_path,
        "selected-series forecast data",
    )

  active_forecast_df: pd.DataFrame | None = None
  active_forecast_path: Path | None = None

  if selected_forecast_df is not None:
    st.subheader("Selected Series TimesFM Forecast")

    forecast_horizon = len(selected_forecast_df)
    expected_demand = selected_forecast_df["forecast"].sum()
    conservative_demand = selected_forecast_df["lower_80"].sum()
    high_demand = selected_forecast_df["upper_80"].sum()

    st.write(f"Forecast period: {format_date_range(selected_forecast_df)}")

    selected_metrics_top = st.columns(3)
    selected_metrics_top[0].metric("Forecast horizon", f"{forecast_horizon} days")
    selected_metrics_top[1].metric(
        "Expected forecast demand",
        f"{round(expected_demand):.0f} units",
    )
    selected_metrics_top[2].metric(
        "Conservative demand estimate",
        f"{round(conservative_demand):.0f} units",
    )

    selected_metrics_bottom = st.columns(2)
    selected_metrics_bottom[0].metric(
        "High-demand estimate",
        f"{round(high_demand):.0f} units",
    )
    selected_metrics_bottom[1].metric(
        "Forecast file",
        selected_forecast_path.name,
    )

    st.subheader("Inventory Planning Inputs")
    current_inventory = int(
        st.number_input(
            "Current inventory on hand",
            min_value=0,
            value=0,
            step=1,
        )
    )
    service_level_label = st.selectbox(
        "Service level preference",
        options=[
            "Conservative - use 80% upper forecast",
            "Aggressive - use 90% upper forecast",
            "Expected only - use point forecast",
        ],
    )
    lead_time_days = int(
        st.number_input(
            "Lead time in days",
            min_value=0,
            value=0,
            step=1,
        )
    )

    del lead_time_days

    expected_demand_units = math.ceil(selected_forecast_df["forecast"].sum())
    if service_level_label == "Expected only - use point forecast":
      target_demand = expected_demand_units
    elif service_level_label == "Conservative - use 80% upper forecast":
      target_demand = math.ceil(selected_forecast_df["upper_80"].sum())
    else:
      target_demand = math.ceil(selected_forecast_df["upper_90"].sum())

    safety_buffer = max(0, target_demand - expected_demand_units)
    suggested_reorder_quantity = max(0, target_demand - current_inventory)
    inventory_gap = current_inventory - target_demand

    if current_inventory < expected_demand_units:
      stockout_risk = "High"
    elif current_inventory < target_demand:
      stockout_risk = "Moderate"
    else:
      stockout_risk = "Low"

    st.subheader("Inventory Planning Recommendation")
    planning_metrics_top = st.columns(3)
    planning_metrics_top[0].metric(
        "Expected demand over forecast horizon",
        format_units(expected_demand_units),
    )
    planning_metrics_top[1].metric(
        "Target stock level",
        format_units(target_demand),
    )
    planning_metrics_top[2].metric(
        "Safety buffer",
        format_units(safety_buffer),
    )

    planning_metrics_bottom = st.columns(3)
    planning_metrics_bottom[0].metric(
        "Current inventory",
        format_units(current_inventory),
    )
    planning_metrics_bottom[1].metric(
        "Suggested reorder quantity",
        format_units(suggested_reorder_quantity),
    )
    planning_metrics_bottom[2].metric(
        "Inventory gap",
        format_units(inventory_gap),
    )

    st.markdown(f"**Stockout risk: {stockout_risk}**")

    if suggested_reorder_quantity == 0:
      st.caption(
          "Current inventory is sufficient for the selected service level."
      )
    else:
      st.caption(
          "Recommended reorder quantity is based on the selected service "
          "level and current inventory."
      )

    st.caption(
        "Expected only uses the point forecast. Conservative uses the upper "
        "80% forecast range. Aggressive uses the upper 90% forecast range."
    )

    st.subheader("Forecast Chart")
    st.caption(
        "The chart compares the last 180 historical rows for the selected "
        "series with the TimesFM forecast and prediction intervals."
    )
    chart = build_forecast_chart(historical_df, selected_forecast_df)
    st.plotly_chart(chart, use_container_width=True)

    active_forecast_df = selected_forecast_df
    active_forecast_path = selected_forecast_path
  else:
    if is_forecastable:
      st.info(
          "No TimesFM forecast has been generated for this selected series yet."
      )
      forecast_command = (
          "python scripts/forecast_selected_series.py "
          f"--shop-id {selected_shop_id} "
          f"--item-id {selected_item_id} "
          "--horizon 30"
      )
      st.code(forecast_command, language="bash")
    else:
      st.warning(
          "This series has too little history for a reliable TimesFM forecast. "
          "Select a series with at least 180 historical rows."
      )

    st.subheader("Sample TimesFM Forecast Demo")
    st.caption(
        "This section shows only the precomputed sample forecast and is not "
        "linked to the current sidebar selection."
    )

    sample_historical_df = load_csv(
        SAMPLE_HISTORICAL_PATH,
        "sample historical data",
    )
    sample_forecast_df = load_csv(SAMPLE_FORECAST_PATH, "sample forecast data")

    if sample_historical_df is not None and sample_forecast_df is not None:
      forecast_horizon = len(sample_forecast_df)
      expected_demand = sample_forecast_df["forecast"].sum()
      conservative_demand = sample_forecast_df["lower_80"].sum()
      high_demand = sample_forecast_df["upper_80"].sum()

      st.write(f"Forecast period: {format_date_range(sample_forecast_df)}")

      sample_top_metrics = st.columns(3)
      sample_top_metrics[0].metric("Forecast horizon", f"{forecast_horizon} days")
      sample_top_metrics[1].metric(
          "Expected forecast demand",
          f"{round(expected_demand):.0f} units",
      )
      sample_top_metrics[2].metric(
          "Conservative demand estimate",
          f"{round(conservative_demand):.0f} units",
      )

      sample_lower_metrics = st.columns(2)
      sample_lower_metrics[0].metric(
          "High-demand estimate",
          f"{round(high_demand):.0f} units",
      )
      sample_lower_metrics[1].metric(
          "Sample historical rows",
          f"{len(sample_historical_df):,}",
      )

      st.subheader("Forecast Chart")
      st.caption(
          "This demo chart uses the precomputed sample forecast and sample "
          "historical series only."
      )
      chart = build_forecast_chart(sample_historical_df, sample_forecast_df)
      st.plotly_chart(chart, use_container_width=True)

      active_forecast_df = sample_forecast_df
      active_forecast_path = SAMPLE_FORECAST_PATH

  st.subheader("Forecast Details")
  if active_forecast_df is None or active_forecast_path is None:
    st.error("No forecast CSV is available to display.")
    return

  st.download_button(
      label="Download forecast CSV",
      data=active_forecast_path.read_bytes(),
      file_name=active_forecast_path.name,
      mime="text/csv",
  )

  display_forecast_df = active_forecast_df.copy()
  numeric_columns = display_forecast_df.select_dtypes(include="number").columns
  display_forecast_df[numeric_columns] = display_forecast_df[numeric_columns].round(2)
  st.dataframe(display_forecast_df, use_container_width=True)


if __name__ == "__main__":
  main()
