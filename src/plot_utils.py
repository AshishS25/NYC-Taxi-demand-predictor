from datetime import timedelta
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px

def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions: Optional[pd.Series] = None,
):
    """
    Plots the time series data for a specific location from NYC taxi data.
    """
    # Check if row_id exists in the index
    if row_id not in features.index:
        # Try to find by pickup_location_id
        location_features = features[features["pickup_location_id"] == row_id]
        if len(location_features) == 0:
            raise ValueError(f"No data found for row_id or location_id {row_id}")
    else:
        # Use row_id as index directly
        location_features = features.loc[[row_id]]

    # Get the actual target
    actual_target = targets.iloc[row_id]

    # Identify time series columns
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]

    # Extract time series values (handle single row correctly)
    time_series_values = [
        location_features[col].iloc[0] for col in time_series_columns
    ] + [actual_target]

    # Get pickup hour as a single timestamp
    pickup_hour = location_features["pickup_hour"].iloc[0]

    # Generate corresponding timestamps
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h",
    )

    # Create the plot title
    title = f"Pickup Hour: {pickup_hour}, Location ID: {location_features['pickup_location_id'].iloc[0]}"

    # Create the base line plot
    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        template="plotly_white",
        markers=True,
        title=title,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    # Add the actual target value as a green marker
    fig.add_scatter(
        x=time_series_dates[-1:],
        y=[actual_target],
        line_color="green",
        mode="markers",
        marker_size=10,
        name="Actual Value",
    )

    # Optionally add the prediction as a red marker
    if predictions is not None:
        # Handle different types of prediction objects
        if isinstance(predictions, np.ndarray):
            pred_value = predictions[row_id]
        elif hasattr(predictions, 'iloc'):
            pred_value = predictions.iloc[row_id]
        else:
            pred_value = predictions[row_id]
        fig.add_scatter(
            x=time_series_dates[-1:],  # Last timestamp
            y=[pred_value],
            line_color="red",
            mode="markers",
            marker_symbol="x",
            marker_size=15,
            name="Prediction",
        )
        # fig.add_scatter(
        #     x=time_series_dates[-1:],
        #     y=[predictions.iloc[row_id]],
        #     line_color="red",
        #     mode="markers",
        #     marker_symbol="x",
        #     marker_size=15,
        #     name="Prediction",
        # )

    return fig


# def plot_aggregated_time_series(
#     features: pd.DataFrame,
#     targets: pd.Series,
#     row_id: int,
#     predictions: Optional[pd.Series] = None,
# ):
#     """
#     Plots the time series data for a specific location from NYC taxi data.

#     Args:
#         features (pd.DataFrame): DataFrame containing feature data, including historical ride counts and metadata.
#         targets (pd.Series): Series containing the target values (e.g., actual ride counts).
#         row_id (int): Index of the row to plot.
#         predictions (Optional[pd.Series]): Series containing predicted values (optional).

#     Returns:
#         plotly.graph_objects.Figure: A Plotly figure object showing the time series plot.
#     """
#     # Extract the specific location's features and target
#     # location_features = features[features["pickup_location_id" == row_id]]
#     location_features = features[features["pickup_location_id"] == row_id]
#     actual_target = targets.iloc[row_id]
#     # actual_target = targets[targets["pickup_location_id" == row_id]]

#     # Identify time series columns (e.g., historical ride counts)
#     time_series_columns = [
#         col for col in features.columns if col.startswith("rides_t-")
#     ]
#     time_series_values = [location_features[col] for col in time_series_columns] + [
#         actual_target
#     ]

#     # Generate corresponding timestamps for the time series
#     time_series_dates = pd.date_range(
#         start=location_features["pickup_hour"]
#         - timedelta(hours=len(time_series_columns)),
#         end=location_features["pickup_hour"],
#         freq="h",
#     )

#     # Create the plot title with relevant metadata
#     title = f"Pickup Hour: {location_features['pickup_hour']}, Location ID: {location_features['pickup_location_id']}"

#     # Create the base line plot
#     fig = px.line(
#         x=time_series_dates,
#         y=time_series_values,
#         template="plotly_white",
#         markers=True,
#         title=title,
#         labels={"x": "Time", "y": "Ride Counts"},
#     )

#     # Add the actual target value as a green marker
#     fig.add_scatter(
#         x=time_series_dates[-1:],  # Last timestamp
#         y=[actual_target],  # Actual target value
#         line_color="green",
#         mode="markers",
#         marker_size=10,
#         name="Actual Value",
#     )

#     # Optionally add the prediction as a red marker
#     if predictions is not None:
#         fig.add_scatter(
#             x=time_series_dates[-1:],  # Last timestamp
#             y=[predictions.iloc[row_id]],
#             # y=predictions[
#             #    predictions["pickup_location_id" == row_id]
#             # ],  # Predicted value
#             line_color="red",
#             mode="markers",
#             marker_symbol="x",
#             marker_size=15,
#             name="Prediction",
#         )

#     return fig


def plot_prediction(features: pd.DataFrame, prediction: int):
    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = [
        features[col].iloc[0] for col in time_series_columns
    ] + prediction["predicted_demand"].to_list()

    # Convert pickup_hour Series to single timestamp
    pickup_hour = pd.Timestamp(features["pickup_hour"].iloc[0])

    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h",
    )

    # Create a DataFrame for the historical data
    historical_df = pd.DataFrame(
        {"datetime": time_series_dates, "rides": time_series_values}
    )

    # Create the plot title with relevant metadata
    title = f"Pickup Hour: {pickup_hour}, Location ID: {features['pickup_location_id'].iloc[0]}"

    # Create the base line plot
    fig = px.line(
        historical_df,
        x="datetime",
        y="rides",
        template="plotly_white",
        markers=True,
        title=title,
        labels={"datetime": "Time", "rides": "Ride Counts"},
    )

    # Add prediction point
    fig.add_scatter(
        x=[pickup_hour],  # Last timestamp
        y=prediction["predicted_demand"].to_list(),
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )

    return fig
