# Algae blooms predictions

Code: [https://github.com/limash/blooms-ml](https://github.com/limash/blooms-ml)

The main goal of the project is to assess the possibility to predict algae blooms using machine learning (ML) methods.

Predicting algal blooms is interesting in the context of the following task of predicting harmful algal blooms (HABs).
Algal blooms prediction is hard because it happens only a couple times a year (we are usually interested to predict when a bloom starts) and it can be a georaphically dependent task.
These factors limit the amount of training data, which we need a lot in conventional ML methods.

There are 2 main sources of data we use during this project:

1. The modeled data, which comes from the ROMS+NERSEM hydro-physical-biogeochemical model of the Hardangerfjord system and has weekly averages of plenty of seawater parameters.
2. Oslo - Kiel Color Fantasy cruise ship ferrybox data.

We solve algae blooms prediction problem as a binary classification task (predicting classes: bloom or no bloom) and apply several other ML methods to study the data.
The short conclusion of the project is that it is possible to predict algae blooms with conventional ML (decision tree classifiers, deep learning).
We trained different ML models on modeled data and ferrybox data, the thorough description is available following the links:

1. [Modeled data](modeled_data.md)
2. [Ferrybox data](ferrybox_data.md)

## Modeled data based ML conclusion

During the project we had difficulties with fetching the data from NIVA resources.
We weren't able to retrieve nutrients, temperature and salinity profiles observations as a timeseries.
First, the data is scaterred through different databases and private sources.
Then, similar variables has different names even withing one data source, units of similar variables are also often are different, so variable values uncomparable.
To make the data usable it is important to share the data and follow the standards.

## Overall conclusion

According to the mentioned issues it can be better to combine conventional hydro-physical-biogeochemival modeling with the modern machine learning methods.
We believe that the better approach is to develop a digital twin of a reservoir of interest and then use it for different applications including HABs prediction.
Here saying a digital twin we mean a hydro-physical-biogeochemical model powered by ML in several ways:

1. A hydro-physical-biogeochemical model results correction to match observations (data assimilation). 
2. The best parameters of a hydro-physical-biogeochmical model identification for different applications (e.g. for algae blooms prediction).
