from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def keep_time_integrity(df):
    df = df.sort_values(by="Time")
    df["time_diff"] = df["Time"].diff()
    first_large_diff_idx = df[df["time_diff"] > pd.Timedelta(minutes=10)].index.min()
    if pd.notna(first_large_diff_idx):
        filtered_df = df.loc[: first_large_diff_idx - 1].drop(columns="time_diff")
    else:
        filtered_df = df.drop(columns="time_diff")
    return filtered_df


def standardize_series(series, target_length=64):
    """
    Parameters:
    series (pd.Series): The input Series to interpolate.
    target_length (int): The desired length after interpolation.
    
    Returns:
    pd.Series: The standardized Series with the target length.
    """
    original_indices = np.linspace(0, len(series) - 1, num=len(series))
    target_indices = np.linspace(0, len(series) - 1, num=target_length)
    interpolated_values = np.interp(target_indices, original_indices, series)
    standardized_values = (interpolated_values - np.mean(interpolated_values)) / np.std(interpolated_values)
    
    return pd.Series(standardized_values)

def get_dataframe_ferrybox2002to2018():
    datadir = f"{Path.home()}/data_ferrybox"
    df = pd.read_csv(f"{datadir}/ferrybox_colorline_2002-2018.csv", dtype={"Northbound": "str"})
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.iloc[:, :6]
    df = df.dropna()
    # put snippets near oslo to separate dataframes
    df_filtered = df[df["LAT"] > 59.5]
    df_filtered["group"] = (df_filtered.index.to_series().diff() != 1).cumsum()
    dfs = [group for _, group in df_filtered.groupby("group")]
    dfs = [df.drop(columns=["group"]) for df in dfs]
    # keep increasing latitude only (on the way to Oslo)
    dfs = [df[df["LAT"].diff().fillna(1) > 0] for df in dfs]
    # do not include points after a large time gap
    dfs = [keep_time_integrity(df) for df in dfs]
    # a snippet should cover most of the oslo fjord
    dfs = [df for df in dfs if df["LAT"].iloc[0] < 59.6 and df["LAT"].iloc[-1] > 59.8]
    temp_standardized = [standardize_series(df["TEMP.IN"]) for df in dfs]
    salt_standardized = [standardize_series(df["SAL"]) for df in dfs]
    df_temp = pd.DataFrame([s.values for s in temp_standardized])
    df_salt = pd.DataFrame([s.values for s in salt_standardized])
    df_temp.columns = ['temp_' + str(i) for i in df_temp.columns]
    df_salt.columns = ['salt_' + str(i) for i in df_salt.columns]
    times = [df["Time"].iloc[0] for df in dfs]
    flues = [df["FLU.FIELDCAL"].mean() for df in dfs]
    # labeling
    three_days = timedelta(days=3)
    prev_time, prev_fluo = times[0], flues[0]
    labels = []
    for time, fluo in zip(times, flues):
        if fluo > 1 and fluo > 1.5 * prev_fluo and (time - prev_time) < three_days:
            labels.append(1)
        else:
            labels.append(0)
        prev_fluo, prev_time = fluo, time
    df_labels = pd.DataFrame(
        {
            "timestamps": times,
            "fluorescence": flues,
            "labels": labels,
        }
    )
    df_labels = pd.concat([df_labels, df_temp, df_salt], axis=1)
    return dfs, df_labels


def plot_temp_salt_flu(df):
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(20, 3))

    # Plot the first dataset (TEMP.IN) on the primary y-axis
    ax1.plot(df.index, df["TEMP.IN"], color="r", label="Temperature (TEMP.IN)")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Temperature (°C)", color="r")
    ax1.tick_params(axis="y", labelcolor="r")

    # Create a second y-axis for SAL
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["SAL"], color="b", label="Salinity (SAL)")
    ax2.set_ylabel("Salinity", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    # Create a third y-axis for FLU.FIELDCAL using twinx and an offset
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Offset the third y-axis to avoid overlap
    ax3.scatter(df.index, df["FLU.FIELDCAL"], color="c", label="Fluorescence (FLU.FIELDCAL)")
    ax3.set_ylabel("Fluorescence", color="c")
    ax3.tick_params(axis="y", labelcolor="c")

    # Adjust layout
    fig.tight_layout()