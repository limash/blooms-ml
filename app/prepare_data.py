import argparse
import glob
import os
import time
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster

from blooms_ml.utils import (
    get_rho_profiles,
    get_stations,
    sample_stations,  # noqa: F401
    sample_stations_sparse,  # noqa: F401
)

VARS = ['P1_c', 'N1_p', 'N3_n', 'N5_s']


def get_file_name(file_path):
    base_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension


def prepare_file(file, xis, etas):
    ds = xr.open_mfdataset(file)
    df = get_stations(ds, xis, etas)
    # this can be redundant
    df = df.reset_index().set_index(['station', 'ocean_time', 's_rho'])
    rho = df.reset_index().groupby('station').apply(get_rho_profiles, include_groups=False)
    # sort that df indices correspond to rho indices
    df = df.reset_index().set_index(['s_rho', 'station', 'ocean_time']).sort_index(level=['station', 'ocean_time'])
    df = pd.concat([df.reset_index(), rho.reset_index(drop=True)], axis=1)
    df = df[df['s_rho'] >= -0.3]  # surface
    df = df.reset_index(drop=True)

    filename = get_file_name(file)
    df.to_parquet(
        f"{Path.home()}/data_ROHO/{filename}.parquet"
        )
    print(f"File {filename} is ready.")


def prepare_files(files):
    stations, st_labels, xis, etas = sample_stations_sparse(xr.open_dataset(files[0]))
    start = time.perf_counter()
    for file in files:
        prepare_file(file, xis, etas)
        print(f"Time elapsed {time.perf_counter() - start:.2f} seconds.")
    print("Finish.")


def prepare_cnps_mean_std(files):
    ds = xr.open_mfdataset(files)[VARS].isel(s_rho=-1)
    mean_values = ds[VARS].mean().to_dask_dataframe().iloc[:, 1:].rename(columns=lambda x: f"{x}_mean")
    std_values = ds[VARS].std().to_dask_dataframe().iloc[:, 1:].rename(columns=lambda x: f"{x}_std")
    result = dd.merge(mean_values, std_values)
    result.to_csv(f"{Path.home()}/data_ROHO/cnps_mean_std.csv", single_file=True)
    print("Finish.")


def merge_files(files):
    df = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    df = df.set_index(['s_rho', 'station', 'ocean_time']).sort_index(level=['station', 'ocean_time'])
    df = df.reset_index()
    df.to_parquet(f"{Path.home()}/data_ROHO/roho800_weekly_average.parquet", index=False, engine='pyarrow')
    print("Finish.")


def main():
    parser = argparse.ArgumentParser(description='Prepare ROHO800 modeled data for a ML model.')
    parser.add_argument(
        '--mode',
        choices=['s', 'p', 'm'],
        help='Choose mode: s(statistics calculation), p(prepare data), or m(merge data)',
        required=True)
    parser.add_argument('--limit-memory', action='store_true', help='Limits memory usage')

    args = parser.parse_args()

    if args.limit_memory:
        cluster = LocalCluster(memory_limit='8GB')
        client = Client(cluster)  # noqa: F841
    if args.mode == 's':
        files = sorted(glob.glob(
            f"{Path.home()}/fram_shmiak/ROHO800_hindcast_2007_2019_v2bu/roho800_v2bu_avg/*avg*.nc"
        ))
        prepare_cnps_mean_std(files)
    elif args.mode == 'p':
        files = sorted(glob.glob(
            f"{Path.home()}/fram_shmiak/ROHO800_hindcast_2007_2019_v2bu/roho800_v2bu_avg/*avg*.nc"
        ))
        prepare_files(files)
    elif args.mode == 'm':
        files = sorted(glob.glob(
            f"{Path.home()}/data_ROHO/*avg*.parquet"
        ))
        merge_files(files)


if __name__ == "__main__":
    main()
