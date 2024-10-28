# Ferrybox data

We also train ML models on the data got from NIVA ferrybox observational program.
This program provides surface measurements of several seawater parameters following tracks of some ships, e.g. Oslo-Kiel, Tromso-Longyearbean, etc.

## Input data

For this project we extract temperature, salinity, and fluorescence snippets from a Color Fantasy cruise ship ferrybox track (Oslo-Kiel) records.
We use surface measurements from 59.7 latitude northwards until Color Fantasy docks in Oslo.
Then we interpolate / extrapolate snippets to the same length (30 points).

An example track snippet:
![fb_track](../images/fb_track.png)

We also use the river discharge data from 3 observation stations around Oslo: Solbergfoss (Glomma), Mjøndalen bru (Drammen), and Bjørnegårdssvingen (inner Oslo fjord).

Input data example before normalization:
![fb_data](../images/fb_data.png)

## Results

As for modeled data instead of snapshots we use increments of variables for 1 and 2 successive periods.
Then we normalize all the data columnwise (through time).

All data is balanced (equal amount of points with label=0 and label=1).
The data used for training and testing is sepatated temporaly.
This is the best approach for autocorrelated (correlation over time) data.
For tests we take all data after 2015.

### Principal Component Analysis (PCA)

![fb_pca](../images/fb_pca.png)
