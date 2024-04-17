import glob
from pathlib import Path

from blooms_ml.utils import prepare_data


def main():
    last_n_files = 20
    files_dia = sorted(glob.glob(
        f"{Path.home()}/fram_shmiak/ROHO800_hindcast_2007_2019_v2bu/roho800_v2bu_dia/*dia*.nc"
    ))[-last_n_files:]
    files_avg = sorted(glob.glob(
        f"{Path.home()}/fram_shmiak/ROHO800_hindcast_2007_2019_v2bu/roho800_v2bu_avg/*avg*.nc"
    ))[-last_n_files:]

    df = prepare_data(files_dia, files_avg, 500)
    df.to_parquet(f"{Path.home()}/data_ROHO/blooms-ml-stations.parquet")


if __name__ == "__main__":
    main()