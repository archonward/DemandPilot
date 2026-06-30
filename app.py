from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROCESSED_HISTORY_PATH = Path("data/processed/future_sales_daily.csv")
FORECASTS_DIR = Path("data/processed/forecasts")
BACKTESTS_DIR = Path("data/processed/backtests")
SAMPLE_HISTORICAL_PATH = Path("data/sample/sample_sales_series.csv")
SAMPLE_FORECAST_PATH = Path("data/processed/sample_forecast.csv")
HISTORICAL_WINDOW_DAYS = 180
MIN_HISTORY_ROWS = 180
UPLOAD_REQUIRED_COLUMNS = ["date", "series_id", "value"]


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


@st.cache_data(show_spinner=False)
def load_backtest_csv(path_str: str) -> pd.DataFrame:
  path = Path(path_str)
  return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(
      drop=True
  )


@st.cache_data(show_spinner=False)
def load_metrics_csv(path_str: str) -> pd.DataFrame:
  path = Path(path_str)
  return pd.read_csv(path)


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


def build_selected_backtest_glob(shop_id: int, item_id: int) -> str:
  return f"shop_{shop_id}*item*{item_id}*backtest.csv"


def build_selected_metrics_glob(shop_id: int, item_id: int) -> str:
  return f"shop*{shop_id}*item*{item_id}_metrics.csv"


def find_selected_forecast_path(shop_id: int, item_id: int) -> Path | None:
  pattern = build_selected_forecast_glob(shop_id, item_id)
  matches = sorted(FORECASTS_DIR.glob(pattern))
  if not matches:
    return None
  return matches[0]


def find_selected_backtest_paths(
    shop_id: int,
    item_id: int,
) -> tuple[Path | None, Path | None]:
  backtest_matches = sorted(
      BACKTESTS_DIR.glob(build_selected_backtest_glob(shop_id, item_id))
  )
  metrics_matches = sorted(
      BACKTESTS_DIR.glob(build_selected_metrics_glob(shop_id, item_id))
  )

  backtest_path = backtest_matches[0] if backtest_matches else None
  metrics_path = metrics_matches[0] if metrics_matches else None
  return backtest_path, metrics_path


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


def build_backtest_chart(backtest_df: pd.DataFrame) -> go.Figure:
  fig = go.Figure()

  if {
      "timesfm_lower_80",
      "timesfm_upper_80",
  }.issubset(backtest_df.columns):
    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["timesfm_upper_80"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            name="TimesFM 80% upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["timesfm_lower_80"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.18)",
            name="TimesFM 80% interval",
            hovertemplate="TimesFM 80% interval<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        )
    )

  fig.add_trace(
      go.Scatter(
          x=backtest_df["date"],
          y=backtest_df["actual"],
          mode="lines",
          name="Actual",
          line={"color": "#1f77b4", "width": 3},
      )
  )
  fig.add_trace(
      go.Scatter(
          x=backtest_df["date"],
          y=backtest_df["timesfm_forecast"],
          mode="lines",
          name="TimesFM forecast",
          line={"color": "#ff7f0e", "width": 3},
      )
  )
  fig.add_trace(
      go.Scatter(
          x=backtest_df["date"],
          y=backtest_df["naive_forecast"],
          mode="lines",
          name="Naive forecast",
          line={"color": "#2ca02c", "width": 2, "dash": "dash"},
      )
  )
  fig.add_trace(
      go.Scatter(
          x=backtest_df["date"],
          y=backtest_df["moving_average_30_forecast"],
          mode="lines",
          name="Moving average 30 forecast",
          line={"color": "#9467bd", "width": 2, "dash": "dot"},
      )
  )
  fig.add_trace(
      go.Scatter(
          x=backtest_df["date"],
          y=backtest_df["seasonal_naive_7_forecast"],
          mode="lines",
          name="Seasonal naive 7 forecast",
          line={"color": "#8c564b", "width": 2, "dash": "dashdot"},
      )
  )

  fig.update_layout(
      title="Backtest Forecast Comparison",
      xaxis_title="Date",
      yaxis_title="Sales quantity",
      hovermode="x unified",
      legend_title_text="Series",
      height=500,
      margin={"l": 20, "r": 20, "t": 60, "b": 20},
  )

  return fig


def build_uploaded_history_chart(uploaded_series_df: pd.DataFrame) -> go.Figure:
  plot_df = uploaded_series_df.tail(HISTORICAL_WINDOW_DAYS).copy()

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=plot_df["date"],
          y=plot_df["value"],
          mode="lines",
          name="Historical demand",
          line={"color": "#1f77b4", "width": 2},
      )
  )
  fig.update_layout(
      title="Historical Demand from Uploaded CSV",
      xaxis_title="Date",
      yaxis_title="Value",
      hovermode="x unified",
      height=450,
      margin={"l": 20, "r": 20, "t": 60, "b": 20},
  )
  return fig


def get_missing_required_columns(df: pd.DataFrame) -> list[str]:
  return [column for column in UPLOAD_REQUIRED_COLUMNS if column not in df.columns]


def validate_uploaded_csv(uploaded_df: pd.DataFrame) -> tuple[pd.DataFrame | None, dict]:
  working_df = uploaded_df.copy()
  validation_summary: dict[str, int] = {}

  parsed_dates = pd.to_datetime(working_df["date"], errors="coerce")
  original_date_strings = working_df["date"].astype("string").str.strip()
  invalid_date_mask = original_date_strings.notna() & (original_date_strings != "") & parsed_dates.isna()
  invalid_date_count = int(invalid_date_mask.sum())
  validation_summary["invalid_date_count"] = invalid_date_count
  if invalid_date_count > 0:
    return None, validation_summary

  numeric_values = pd.to_numeric(working_df["value"], errors="coerce")
  original_value_strings = working_df["value"].astype("string").str.strip()
  invalid_value_mask = (
      original_value_strings.notna()
      & (original_value_strings != "")
      & numeric_values.isna()
  )
  invalid_value_count = int(invalid_value_mask.sum())
  validation_summary["invalid_value_count"] = invalid_value_count
  if invalid_value_count > 0:
    return None, validation_summary

  working_df["date"] = parsed_dates
  working_df["value"] = numeric_values
  working_df["series_id"] = working_df["series_id"].astype("string").str.strip()
  working_df = working_df.dropna(subset=UPLOAD_REQUIRED_COLUMNS)
  working_df = working_df.loc[working_df["series_id"] != ""].copy()
  working_df = working_df[UPLOAD_REQUIRED_COLUMNS]
  working_df = working_df.sort_values(["series_id", "date"]).reset_index(drop=True)
  return working_df, validation_summary


def render_uploaded_csv_mode() -> None:
  st.header("Upload Demand CSV")
  st.write("The uploaded CSV must contain:")
  st.markdown(
      """
      - `date`: daily date column
      - `series_id`: product/store/time-series identifier
      - `value`: numeric demand or sales quantity
      """
  )

  uploaded_file = st.file_uploader(
      "Upload a demand CSV",
      type=["csv"],
      accept_multiple_files=False,
  )

  if uploaded_file is None:
    st.info(
        "Upload forecasting is not enabled yet. This step only validates and "
        "previews uploaded demand data. In the next phase, DemandPilot will "
        "support TimesFM forecasting for uploaded CSV files."
    )
    return

  try:
    uploaded_df = pd.read_csv(uploaded_file)
  except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    return

  st.subheader("Preview")
  st.dataframe(uploaded_df.head(10), use_container_width=True)

  missing_columns = get_missing_required_columns(uploaded_df)
  if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(f"`{column}`" for column in missing_columns)
    )
    return

  cleaned_uploaded_df, validation_summary = validate_uploaded_csv(uploaded_df)

  invalid_date_count = validation_summary.get("invalid_date_count", 0)
  if invalid_date_count > 0:
    st.error(f"Found {invalid_date_count} invalid dates in `date`.")
    return

  invalid_value_count = validation_summary.get("invalid_value_count", 0)
  if invalid_value_count > 0:
    st.error(f"Found {invalid_value_count} invalid values in `value`.")
    return

  if cleaned_uploaded_df is None or cleaned_uploaded_df.empty:
    st.error("No valid rows remain after cleaning the uploaded CSV.")
    return

  st.success("CSV validation passed.")

  quality_metrics_top = st.columns(3)
  quality_metrics_top[0].metric("Total rows", f"{len(cleaned_uploaded_df):,}")
  quality_metrics_top[1].metric(
      "Unique series",
      f"{cleaned_uploaded_df['series_id'].nunique():,}",
  )
  quality_metrics_top[2].metric(
      "Average demand",
      f"{cleaned_uploaded_df['value'].mean():.2f}",
  )

  quality_metrics_bottom = st.columns(3)
  quality_metrics_bottom[0].metric(
      "Earliest date",
      cleaned_uploaded_df["date"].min().date().isoformat(),
  )
  quality_metrics_bottom[1].metric(
      "Latest date",
      cleaned_uploaded_df["date"].max().date().isoformat(),
  )
  quality_metrics_bottom[2].metric(
      "Zero-demand rows",
      f"{int((cleaned_uploaded_df['value'] == 0).sum()):,}",
  )

  uploaded_series_options = cleaned_uploaded_df["series_id"].drop_duplicates().tolist()
  selected_uploaded_series_id = st.selectbox(
      "series_id",
      uploaded_series_options,
  )

  selected_uploaded_series_df = cleaned_uploaded_df.loc[
      cleaned_uploaded_df["series_id"] == selected_uploaded_series_id
  ].copy()

  st.subheader("Selected Uploaded Series")
  uploaded_series_metrics = st.columns(5)
  uploaded_series_metrics[0].metric(
      "Historical rows",
      f"{len(selected_uploaded_series_df):,}",
  )
  uploaded_series_metrics[1].metric(
      "Date range",
      format_date_range(selected_uploaded_series_df),
  )
  uploaded_series_metrics[2].metric(
      "Average daily demand",
      f"{selected_uploaded_series_df['value'].mean():.2f}",
  )
  uploaded_series_metrics[3].metric(
      "Max daily demand",
      f"{selected_uploaded_series_df['value'].max():.2f}",
  )
  uploaded_series_metrics[4].metric(
      "Total demand",
      f"{selected_uploaded_series_df['value'].sum():.2f}",
  )

  if len(selected_uploaded_series_df) < MIN_HISTORY_ROWS:
    st.warning(
        "This uploaded series has limited history. TimesFM forecast quality "
        "may be weaker."
    )

  st.subheader("Historical Demand")
  st.caption("The chart shows the most recent 180 rows from the uploaded series.")
  st.plotly_chart(
      build_uploaded_history_chart(selected_uploaded_series_df),
      use_container_width=True,
  )

  st.subheader("Historical Rows")
  st.dataframe(selected_uploaded_series_df, use_container_width=True)

  st.download_button(
      label="Download cleaned uploaded data",
      data=cleaned_uploaded_df.to_csv(index=False).encode("utf-8"),
      file_name="cleaned_uploaded_demand.csv",
      mime="text/csv",
  )
  st.info(
      "Upload forecasting is not enabled yet. This step only validates and "
      "previews uploaded demand data. In the next phase, DemandPilot will "
      "support TimesFM forecasting for uploaded CSV files."
  )


def render_demo_dataset_mode() -> None:
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
  selected_backtest_path, selected_metrics_path = find_selected_backtest_paths(
      selected_shop_id,
      selected_item_id,
  )
  selected_forecast_df: pd.DataFrame | None = None
  selected_backtest_df: pd.DataFrame | None = None
  selected_metrics_df: pd.DataFrame | None = None

  if selected_forecast_path is not None:
    selected_forecast_df = load_csv(
        selected_forecast_path,
        "selected-series forecast data",
    )

  if selected_backtest_path is not None and selected_metrics_path is not None:
    selected_backtest_df = load_backtest_csv(str(selected_backtest_path))
    selected_metrics_df = load_metrics_csv(str(selected_metrics_path))

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

  if selected_backtest_df is not None and selected_metrics_df is not None:
    st.subheader("Backtest Evaluation")
    st.caption(
        "Backtesting hides the last part of historical data, forecasts it, "
        "and compares predicted demand against the actual observed demand."
    )

    display_metrics_df = selected_metrics_df.copy()
    metrics_column_map = {
        column: {
            "model": "Model",
            "mae": "MAE",
            "rmse": "RMSE",
            "mape": "MAPE",
            "bias": "Bias",
        }.get(column.lower(), column)
        for column in display_metrics_df.columns
    }
    display_metrics_df = display_metrics_df.rename(columns=metrics_column_map)
    if "Model" in display_metrics_df.columns:
      display_metrics_df["Model"] = display_metrics_df["Model"].astype(str)
    numeric_metric_columns = display_metrics_df.select_dtypes(
        include="number"
    ).columns
    display_metrics_df[numeric_metric_columns] = display_metrics_df[
        numeric_metric_columns
    ].round(2)
    st.dataframe(display_metrics_df, use_container_width=True)

    best_model_text = None
    if {"Model", "MAE"}.issubset(display_metrics_df.columns):
      comparable_metrics_df = display_metrics_df.dropna(subset=["MAE"])
      if not comparable_metrics_df.empty:
        best_model_text = str(
            comparable_metrics_df.loc[
                comparable_metrics_df["MAE"].idxmin(),
                "Model",
            ]
        )
        st.write(f"Best model by MAE: {best_model_text}")
        if best_model_text.strip().lower() == "timesfm":
          st.caption(
              "TimesFM performed best on this selected historical backtest."
          )
        else:
          st.caption(
              "A baseline performed better on this selected historical "
              "backtest. This can happen for sparse or stable demand series."
          )

    backtest_chart = build_backtest_chart(selected_backtest_df)
    st.plotly_chart(backtest_chart, use_container_width=True)

    st.subheader("Backtest Details")
    display_backtest_df = selected_backtest_df.copy()
    numeric_backtest_columns = display_backtest_df.select_dtypes(
        include="number"
    ).columns
    display_backtest_df[numeric_backtest_columns] = display_backtest_df[
        numeric_backtest_columns
    ].round(2)
    st.dataframe(display_backtest_df, use_container_width=True)

    download_columns = st.columns(2)
    download_columns[0].download_button(
        label="Download backtest rows CSV",
        data=selected_backtest_path.read_bytes(),
        file_name=selected_backtest_path.name,
        mime="text/csv",
    )
    download_columns[1].download_button(
        label="Download backtest metrics CSV",
        data=selected_metrics_path.read_bytes(),
        file_name=selected_metrics_path.name,
        mime="text/csv",
    )
  else:
    st.subheader("Backtest Evaluation")
    st.info("No backtest has been generated for this selected series yet.")
    if is_forecastable:
      backtest_command = (
          "python scripts/backtest_selected_series.py "
          f"--shop-id {selected_shop_id} "
          f"--item-id {selected_item_id} "
          "--horizon 30"
      )
      st.code(backtest_command, language="bash")
    else:
      st.caption("This series has too little history for reliable backtesting.")

  if selected_forecast_df is None:
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

  if active_forecast_df is None or active_forecast_path is None:
    st.error("No forecast CSV is available to display.")
    return

  st.subheader("Forecast Details")
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


def main() -> None:
  st.title("DemandPilot: Retail Demand Forecasting")
  st.caption(
      "This dashboard uses TimesFM-generated forecasts to help estimate "
      "future product demand."
  )

  with st.sidebar:
    st.header("Instructions")
    st.caption(
      "Current status: dashboard with historical exploration, selected-series "
      "forecasting, inventory recommendations, and backtest evaluation when "
      "generated."
    )
    data_mode = st.radio(
        "Data source",
        options=["Demo dataset", "Upload CSV"],
        index=0,
    )
    st.write("Step 1: Prepare dataset")
    st.write("Step 2: Run TimesFM forecast script")
    st.write("Step 3: View forecast in dashboard")
  if data_mode == "Demo dataset":
    render_demo_dataset_mode()
  else:
    render_uploaded_csv_mode()


if __name__ == "__main__":
  main()
