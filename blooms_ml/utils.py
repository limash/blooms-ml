# Copyright 2024 The Blooms-ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import functools
import itertools
import os
import random
import time

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

VARS = [
    "ocean_time",
    "s_rho",
    "rho",
    "w",
    "P1_c",
    "N1_p",
    "N3_n",
    "N5_s",
]
VARSMIN = [
    "ocean_time",
    "s_rho",
    "rho",
    "P1_c",
    "N1_p",
    "N3_n",
    "N5_s",
]


def check_iloc(ds: xr.Dataset, xi: int, eta: int):
    """
    Check that the coordinates are not at the land
    """
    rhocheck = (ds.mask_rho.isel(eta_rho=eta, xi_rho=xi) == 1).all().item()
    ucheck = (ds.mask_u.isel(eta_u=eta, xi_u=xi - 1) == 1).all().item()
    vcheck = (ds.mask_v.isel(eta_v=eta - 1, xi_v=xi) == 1).all().item()
    return all([rhocheck, ucheck, vcheck])


def sample_stations(ds: xr.Dataset, num_stations: int):
    xi_limits = [i for i in range(10, 310)]
    eta_limits = [i for i in range(10, 220)]
    stations, st_labels = [], []
    xis, etas = [], []

    i = 0
    while True:
        station = random.sample(xi_limits, k=1)[0], random.sample(eta_limits, k=1)[0]
        if station not in stations and check_iloc(ds, station[0], station[1]):
            stations.append(station)
            st_labels.append(str(i + 1))
            xis.append(station[0])
            etas.append(station[1])
            i += 1
            if i >= num_stations:
                return stations, st_labels, xis, etas


def sample_stations_sparse(ds: xr.Dataset):
    stations, st_labels = [], []
    xis, etas = [], []
    n = 0
    for i in range(10, 310, 3):
        for j in range(10, 220, 3):
            station = i, j
            if station not in stations and check_iloc(ds, i, j):
                stations.append(station)
                st_labels.append(str(n + 1))
                xis.append(i)
                etas.append(j)
                n += 1

    return stations, st_labels, xis, etas


def extract_stations_rho(ds: xr.Dataset, xis: list, etas: list):
    datasets = []
    for xi, eta in zip(xis, etas):
        datasets.append(ds.isel(xi_rho=xi, eta_rho=eta))
    return xr.concat(datasets, dim="station")


def extract_stations_u(ds: xr.Dataset, xis: list, etas: list):
    datasets = []
    for xi, eta in zip(xis, etas):
        datasets.append(ds.isel(xi_u=xi - 1, eta_u=eta))
    return xr.concat(datasets, dim="station")


def extract_stations_v(ds: xr.Dataset, xis: list, etas: list):
    datasets = []
    for xi, eta in zip(xis, etas):
        datasets.append(ds.isel(xi_v=xi, eta_v=eta - 1))
    return xr.concat(datasets, dim="station")


def filter_variables_by_dimension(ds: xr.Dataset, dimension: str):
    return xr.Dataset({var: ds[var] for var in ds.variables if dimension in ds[var].dims})


def merge_edges_to_centers(ds: xr.Dataset):
    """
    ROMS has relative vertical coordinates (https://www.myroms.org/wiki/Vertical_S-coordinate).
    s_rho points are at the layer centers.
    s_w are at the edges.
    There is 1 more point in s_w (2 edges and a point at the center, for example).
    Some variables are at the centers, other are at the edges.
    We don't care a lot about it, take bottom edges only.
    """
    ds = ds.isel(s_w=slice(1, None))
    ds["s_w"] = ds["s_rho"].values
    ds_w = filter_variables_by_dimension(ds, "s_w")
    ds1 = ds_w.rename_dims({"s_w": "s_rho"})
    ds2 = filter_variables_by_dimension(ds, "s_rho")
    return xr.merge([ds1.drop_vars("s_w"), ds2])


def labeling(df_rho):
    df_rho = df_rho.reset_index(drop=True)
    df_rho["label"] = -1 * normalize_series(df_rho["P1_c"]).diff(periods=-1)
    df_rho["label"] = np.where(df_rho["label"] > 0.2, 1, 0)
    return df_rho[:-1]


def labeling_incremented(df_rho):
    df_rho = df_rho.reset_index(drop=True)
    df_rho["label"] = df_rho["P1_c"].shift(periods=-1)
    return df_rho


def labeling_binary_incremented(df_rho):
    df_rho = df_rho.reset_index(drop=True)
    df_rho["label"] = df_rho["P1_c"].shift(periods=-1)
    df_rho = df_rho[df_rho["label"].notna()]
    df_rho["label"] = np.where(df_rho["label"] > 0.2, 1, 0)
    return df_rho


def add_differences(df_rho):
    df_rho = df_rho.reset_index(drop=True)
    df_rho[["rho_diff", "P1_c_diff", "N1_p_diff", "N3_n_diff", "N5_s_diff"]] = (
        df_rho[["rho", "P1_c", "N1_p", "N3_n", "N5_s"]].diff().clip(lower=-1, upper=1)
    )
    return df_rho[1:]


def to_differences(df_rho):
    df_rho = df_rho.reset_index(drop=True)
    df_rho[["rho", "P1_c", "N1_p", "N3_n", "N5_s"]] = (
        df_rho[["rho", "P1_c", "N1_p", "N3_n", "N5_s"]].diff().clip(lower=-1, upper=1)
    )
    return df_rho[1:]


def add_previous(df_rho):
    df_rho = df_rho.reset_index(drop=True)
    columns_original = ["rho", "P1_c", "N1_p", "N3_n", "N5_s"] + [str(i) for i in range(1, 26)]
    columns_shifted1 = [f"{column}_1" for column in columns_original]
    columns_shifted2 = [f"{column}_2" for column in columns_original]
    df_rho[columns_shifted1] = df_rho[columns_original].shift(1)
    df_rho[columns_shifted2] = df_rho[columns_original].shift(2)
    return df_rho[2:]  # remove NaNs from shifting


def normalize_series(row: pd.Series):
    return ((row - row.mean()) / row.std()).round(2).astype("float32")


def normalize_columns(df, columns_slice):
    df.iloc[:, columns_slice] = df.iloc[:, columns_slice].apply(normalize_series, axis=0)
    return df


def normalize_rows(df):
    return df.apply(normalize_series, axis=1)


def append_rho_profiles_and_labels(df_station, nlayers: int = 25):
    # add rho profiles
    df_station = df_station.reset_index(drop=True)
    rho = df_station.pivot(index="ocean_time", columns="s_rho", values="rho")
    new_columns = [str(i) for i in range(1, len(rho.columns) + 1)]
    rho.rename(columns=dict(zip(rho.columns[:], new_columns)), inplace=True)
    rho = rho.apply(normalize_series, axis=1)
    rho = rho.loc[rho.index.repeat(nlayers)]
    rho = rho.rename_axis(None, axis=1)
    rho = rho.reset_index()
    df_station = pd.concat([df_station, rho.iloc[:, 1:]], axis=1)
    if isinstance(df_station, dd.DataFrame):
        df_station = dd.concat([df_station, rho.iloc[:, 1:]], axis=1)
    elif isinstance(df_station, pd.DataFrame):
        df_station = pd.concat([df_station, rho.iloc[:, 1:]], axis=1)
    else:
        raise ValueError("Unsupported dataframe type")
    # add label
    df_station = df_station.reset_index(drop=True).groupby("s_rho").apply(labeling, include_groups=False)
    return df_station


def get_rho_profiles(df_station, nlayers: int = 25):
    df_station = df_station.reset_index(drop=True)
    rho = df_station.pivot(index="ocean_time", columns="s_rho", values="rho")
    new_columns = [str(i) for i in range(1, len(rho.columns) + 1)]
    rho.rename(columns=dict(zip(rho.columns[:], new_columns)), inplace=True)
    rho = rho.loc[rho.index.repeat(nlayers)]
    rho = rho.rename_axis(None, axis=1)
    return rho


def append_rho_profiles(df_station, nlayers: int = 25):
    rho = get_rho_profiles(df_station, nlayers)
    rho = rho.reset_index()
    return pd.concat([df_station, rho.iloc[:, 1:]], axis=1)


def get_from_dia(ds_dia: xr.Dataset, xis: list, etas: list):
    ds = extract_stations_rho(ds_dia, xis, etas)
    ds = merge_edges_to_centers(ds)
    return ds[["light_PAR0", "P1_netPI"]].to_dask_dataframe()


def get_from_avg(ds_avg: xr.Dataset, xis: list, etas: list, is_dask: bool = True):
    ds_rho = extract_stations_rho(ds_avg, xis, etas)
    ds_rho = ds_rho.drop_dims(["eta_u", "eta_v", "eta_psi", "xi_u", "xi_v", "xi_psi"])
    ds_u = extract_stations_u(ds_avg, xis, etas)
    ds_u = ds_u.drop_dims(["eta_rho", "eta_v", "eta_psi", "xi_rho", "xi_v", "xi_psi"])
    ds_v = extract_stations_v(ds_avg, xis, etas)
    ds_v = ds_v.drop_dims(["eta_rho", "eta_u", "eta_psi", "xi_rho", "xi_u", "xi_psi"])
    ds = xr.merge([ds_rho, ds_u, ds_v])

    ds = merge_edges_to_centers(ds)
    ds_subset = ds.drop_vars([var for var in ds.variables if var not in VARS])
    if is_dask:
        return ds_subset.to_dask_dataframe()
    else:
        return ds_subset.to_dataframe()


def get_stations(ds: xr.Dataset, xis: list, etas: list):
    ds = ds.drop_vars([var for var in ds.variables if var not in VARSMIN])
    ds = extract_stations_rho(ds, xis, etas)
    return ds.to_dataframe()


def prepare_data(files_dia: list[str], files_avg: list[str], num_stations: int):
    ds_dia = xr.open_mfdataset(files_dia)
    ds_avg = xr.open_mfdataset(files_avg)
    stations, st_labels, xis, etas = sample_stations(ds_dia, num_stations)

    ddf_dia = get_from_dia(ds_dia, xis, etas)
    df_dia_orig = ddf_dia.compute()

    ddf = get_from_avg(ds_avg, xis, etas)
    df_orig = ddf.compute()

    df_dia = df_dia_orig.reset_index().drop("index", axis=1).set_index(["station", "ocean_time", "s_rho"])
    df = df_orig.reset_index().drop("index", axis=1).set_index(["station", "ocean_time", "s_rho"])

    df["light_PAR0"] = df_dia["light_PAR0"]
    df["P1_netPI"] = df_dia["P1_netPI"]

    df = df.reset_index().groupby("station").apply(append_rho_profiles)
    df = df[df["s_rho"] > -0.3]  # surface
    df = df.reset_index(drop=True)
    df.iloc[:, 3:11] = df.iloc[:, 3:11].apply(normalize_series, axis=0)
    return df


def plot_variable(variable: pd.DataFrame):
    y = variable.index.values
    x = np.arange(variable.shape[1])
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(15, 5))
    cf = ax.contourf(X, Y, variable.values)
    fig.colorbar(cf, ax=ax, location="right", pad=0.01)


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    from: www.kaggle.com .. credit-fraud-dealing-with-imbalanced-datasets
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def get_stats(filepath: str):
    with open(filepath, newline="") as file:
        csv_reader = csv.reader(file)
        for _ in range(2):
            (_, p1_c_mean, n1_p_mean, n3_n_mean, n5_s_mean, p1_c_std, n1_p_std, n3_n_std, n5_s_std) = next(
                csv_reader, None
            )
    return p1_c_mean, n1_p_mean, n3_n_mean, n5_s_mean, p1_c_std, n1_p_std, n3_n_std, n5_s_std


def get_dataframe(datadir):
    # open
    df = pd.read_parquet(os.path.join(datadir, "roho800_weekly_average.parquet"))
    # label
    df = df.groupby(["station", "s_rho"]).apply(labeling, include_groups=False)
    df = df.reset_index().drop(columns="level_2")
    df.rename(columns={"label": "y"}, inplace=True)
    df["label"] = np.where(df["y"] > 3, 1, 0)
    return df


def timeit(func):
    """Decorator to measure and report the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        print(f"Function '{func.__name__}' took {execution_time:.4f} seconds to complete.")
        return result

    return wrapper


@timeit
def get_datasets_classification(datadir):
    df = get_dataframe(datadir)
    (p1_c_mean, n1_p_mean, n3_n_mean, n5_s_mean, p1_c_std, n1_p_std, n3_n_std, n5_s_std) = get_stats(
        os.path.join(datadir, "cnps_mean_std.csv")
    )
    # clean
    df = df[df["y"].notna()]
    df = df.drop(columns=["station", "s_rho", "rho", "y"])
    # "normalize"
    df["P1_c"] = ((df["P1_c"] - float(p1_c_mean)) / float(p1_c_std)).round(2).astype("float32")
    df["N1_p"] = ((df["N1_p"] - float(n1_p_mean)) / float(n1_p_std)).round(2).astype("float32")
    df["N3_n"] = ((df["N3_n"] - float(n3_n_mean)) / float(n3_n_std)).round(2).astype("float32")
    df["N5_s"] = ((df["N5_s"] - float(n5_s_mean)) / float(n5_s_std)).round(2).astype("float32")
    # split
    df_train = df[df["ocean_time"] < "2013-01-01"]
    df_test = df[df["ocean_time"] > "2013-01-01"]
    del df
    train_data = {
        "label": df_train["label"].values,
        "observations": df_train.drop(columns=["ocean_time", "label", "P1_c"]).values,
    }
    test_data = {
        "label": df_test["label"].values,
        "observations": df_test.drop(columns=["ocean_time", "label", "P1_c"]).values,
    }
    return train_data, test_data


@timeit
def get_datasets_classification_stacked(datadir):
    df = pd.read_parquet(os.path.join(datadir, "roho800_weekly_average_stacked.parquet"))

    df = df.groupby(["station", "s_rho"]).apply(labeling_binary_incremented, include_groups=False)
    df = df.reset_index().drop(columns="level_2")
    df = df.drop(columns=["station", "s_rho"])

    df_train = df[df["ocean_time"] < "2013-01-01"]
    df_test = df[df["ocean_time"] > "2013-01-01"]
    del df
    train_data = {
        "label": df_train["label"].values,
        "observations": df_train.drop(columns=["ocean_time", "label"]).values,
    }
    test_data = {
        "label": df_test["label"].values,
        "observations": df_test.drop(columns=["ocean_time", "label"]).values,
    }
    return train_data, test_data


@timeit
def get_datasets_regression(datadir):
    df = pd.read_parquet(os.path.join(datadir, "roho800_weekly_average_stacked.parquet"))

    df = df.groupby(["station", "s_rho"]).apply(labeling_incremented, include_groups=False)
    df = df.reset_index().drop(columns="level_2")
    df.rename(columns={"label": "y"}, inplace=True)
    df = df[df["y"].notna()]
    df = df.drop(columns=["station", "s_rho"])
    df["y"] = df["y"].clip(lower=-1, upper=1)
    # split
    df_train = df[df["ocean_time"] < "2013-01-01"]
    df_test = df[df["ocean_time"] > "2013-01-01"]
    del df
    train_data = {
        "y": df_train["y"].values,
        "observations": df_train.drop(columns=["ocean_time", "y"]).values,
    }
    test_data = {
        "y": df_test["y"].values,
        "observations": df_test.drop(columns=["ocean_time", "y"]).values,
    }
    return train_data, test_data
