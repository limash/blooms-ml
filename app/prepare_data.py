import glob
from pathlib import Path

import dask.dataframe as dd
import xarray as xr
from dask.distributed import Client, LocalCluster

from blooms_ml.utils import (
    append_rho_profiles,
    get_from_avg,
    sample_stations,  # noqa: F401
    sample_stations_sparse,  # noqa: F401
)


VARS = ['P1_c', 'N1_p', 'N3_n', 'N5_s']


def prepare_files(files, num_stations=300):
    stations, st_labels, xis, etas = sample_stations(xr.open_dataset(files[0]), num_stations)
    ds = xr.open_mfdataset(files)
    df = get_from_avg(ds, xis, etas, is_dask=False)
    df = df.reset_index().groupby('station').apply(append_rho_profiles, include_groups=False)
    df = df.reset_index().drop(columns='level_1')
    df = df[df['s_rho'] >= -0.3]  # surface
    df = df.reset_index(drop=True)
    df.to_parquet(f"{Path.home()}/data_ROHO/{num_stations}stations-norm.parquet")
    print("Finish")


def prepare_cnps_mean_std(files):
    ds = xr.open_mfdataset(files)[VARS].isel(s_rho=-1)
    mean_values = ds[VARS].mean().to_dask_dataframe().iloc[:, 1:].rename(columns=lambda x: f"{x}_mean")
    std_values = ds[VARS].std().to_dask_dataframe().iloc[:, 1:].rename(columns=lambda x: f"{x}_std")
    result = dd.merge(mean_values, std_values)
    result.to_csv(f"{Path.home()}/data_ROHO/cnps_mean_std.csv", single_file=True)
    print("Finish")


if __name__ == "__main__":
    cluster = LocalCluster(memory_limit='8GB')
    client = Client(cluster)  # noqa: F841

    files = sorted(glob.glob(
        f"{Path.home()}/fram_shmiak/ROHO800_hindcast_2007_2019_v2bu/roho800_v2bu_avg/*avg*.nc"
    ))

    prepare_files(files, 1000)
