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

import requests

with open(os.path.expanduser("~/Dropbox/access/nve_api")) as file:
    APIKEY = file.read().strip()
RIVERS = ["Glåma", "Glomma", "Drammenselva", "Numedalslågen", "Skienselva"]


def get_station_ids():
    # Define the API URL
    url = "https://hydapi.nve.no/api/v1/Stations"

    request_headers = {"Accept": "application/json", "X-API-Key": APIKEY}
    params = {"Active": 1}
    response = requests.get(url, headers=request_headers, params=params)
    if response.status_code == 200:
        data = response.json()["data"]
        for record in data:
            for river in RIVERS:
                if river in record["stationName"]:
                    print(f"{record["stationName"]}: {record["stationId"]}")
    else:
        print(f"Request failed with status code {response.status_code}")


if __name__ == "__main__":
    get_station_ids()
