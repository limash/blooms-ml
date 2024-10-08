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

# https://hydapi.nve.no/UserDocumentation/
# get hydrological data from "The Norwegian Water Resources and Energy Directorate"
import os
from functools import reduce
from pathlib import Path

import pandas as pd
import requests

RIVERS = ["Glåma", "Glomma", "Drammenselva", "Numedalslågen", "Skienselva", "Solbergfoss"]
STATION_IDS = ["12.534.0", "2.605.0", "8.2.0"]
# the corersponding names (to above IDS) of stations got using get_stations
STATION_NAMES = ["Mjøndalen_bru", "Solbergfoss", "Bjørnegårdssvingen"]
with open(os.path.expanduser("~/Dropbox/access/nve_api")) as file:
    APIKEY = file.read().strip()
DATADIR = f"{Path.home()}/data_ferrybox"


def check_nve_station_name(record):
    for river in RIVERS:
        if river in record["stationName"]:
            print(f"{record["stationName"]}: {record["stationId"]}")


def get_station_ids():
    # Define the API URL
    url = "https://hydapi.nve.no/api/v1/Stations"

    request_headers = {"Accept": "application/json", "X-API-Key": APIKEY}
    params = {"Active": 1}
    response = requests.get(url, headers=request_headers, params=params)
    if response.status_code == 200:
        data = response.json()["data"]
        for record in data:
            check_nve_station_name(record)
    else:
        print(f"Request failed with status code {response.status_code}")


def get_stations():
    # Define the API URL
    url = "https://hydapi.nve.no/api/v1/Stations"

    request_headers = {"Accept": "application/json", "X-API-Key": APIKEY}
    params = {"Active": 1}
    response = requests.get(url, headers=request_headers, params=params)
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return
    data = response.json()["data"]
    for record in data:
        if record["stationId"] in STATION_IDS:
            print(f"{record["stationName"]}: {record["stationId"]}: {record["seriesList"][0]["parameterName"]}.")
    return


def get_daily_water_flow(datadir):
    # Define the API URL
    url = "https://hydapi.nve.no/api/v1/Observations"
    request_headers = {"Accept": "application/json", "X-API-Key": APIKEY}

    dfs = []
    for station_id, station_name in zip(STATION_IDS, STATION_NAMES):
        # 1001 is a waterflow, parameterName from get_stations didn't work
        params = {
            "StationId": station_id,
            "Parameter": "1001",
            "ResolutionTime": "day",
            "ReferenceTime": "2002-01-01/2018-12-31",
        }
        response = requests.get(url, headers=request_headers, params=params)
        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}")
            return
        data = response.json()["data"]
        df = pd.DataFrame(data[0]["observations"])
        df = df.drop(columns=["correction", "quality"])
        df.rename(columns={"value": station_name}, inplace=True)
        dfs.append(df)
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='time', how='outer'), dfs)
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    df_merged.to_csv(f'{datadir}/rivers_2002-2018.csv', index=False)


if __name__ == "__main__":
    datadir = f"{Path.home()}/blooms-ml_data/data_rivers"
    get_daily_water_flow(datadir)
