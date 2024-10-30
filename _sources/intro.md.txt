# Algal blooms prediction

Code: [https://github.com/limash/blooms-ml](https://github.com/limash/blooms-ml)

The main goal of the project is to evaluate the possibility of predicting algal blooms using machine learning (ML) methods.
This is interesting in the context of the following task of predicting harmful algal blooms (HABs).
Predicting algal blooms is difficult because they occur only a few times a year (we are usually interested in predicting when a bloom will start), and it can be a geographically dependent task.
These factors limit the amount of training data we need in traditional ML methods.

There are 2 main sources of data that we use during this project:

1. The modeled data, which comes from the ROMS+NERSEM hydro-physical-biogeochemical model of the Hardangerfjord system and has weekly averages of plenty of seawater parameters.
2. Oslo - Kiel Color Fantasy cruise ship ferrybox data.

We solve the algae bloom prediction problem as a binary classification task (predicting classes: bloom or no bloom) and apply several other ML methods to study the data.
The short conclusion of the project is that it is possible to predict algal blooms with conventional ML (decision tree classifiers, deep learning).
We trained several ML models on modeled and ferrybox data, the detailed description is available following the links:

1. [Modeled data](modeled_data.md)
2. [Ferrybox data](ferrybox_data.md)

## Modeled data results

Both the decision tree classifier and the neural network classifier that we trained on the modeled data can predict blooms.
The accuracy on the balanced dataset (same amount of bloom and no bloom points, for the original dataset with many more no bloom points the accuracy may be better) is about 60-70%.
We believe that it is possible to improve the prediction accuracy significantly.
However, in order to continue working on this, it is necessary to test the algorithms on the observational data to be sure that they actually work properly.
Unfortunately, we haven't been able to get the observational data in a form similar to the modeled data (temperature and salinity profiles to calculate a density profile, and the corresponding nutrients at the surface).

During the project, we had difficulty retrieving data from the NIVA resources.
We weren't able to retrieve nutrients, temperature and salinity profile observations as a time series.
First, the data is scattered through different databases and private sources.
Then, similar variables have different names even within one data source, units of similar variables are also often different, making them uncomparable.
To make the data usable, it is important to share the data and follow the standards.

## Ferrybox data results

It is possible to train a ML model using only the observational data.
But using only the Ferrybox data is not enough, apart from the horizontal surface profiles of temperature and salinity, we need other data.
In this project, we used river discharge data from the observation stations near the ferrybox track.

Before using river discharge as input data, we tried to predict blooms using only horizontal surface profiles of temperature and salinity.
The test loss and accuracy didn't improve in this case, but the training loss and accuracy improved over time.
This means that the neural network could remember the specific patterns of temperature and salinity profiles that correspond to blooms, but couldn't generalise them.
This could happen for 2 reasons: not enough data, or/and the temperature and salinity profiles don't have enough information to predict blooms.

Adding river discharge to the input data enabled learning.
We expect that adding other data on precipitation, nutrients from the stations, etc. will improve the accuracy of the predictions.

Unfortunately, adding new parameters to the observation vector cannot solve the main problem of using only observations for ML model training - not enough data.
There are usually only 2 blooms per year, in spring and autumn.
For 20-25 years of Ferrybox data, this means that we only have at most 50 blooms that we want to predict.

But the Ferrybox data can still be valuable.
We need to combine modelling and observations in the data-driven model.

## Overall conclusion

According to the mentioned problems, it may be better to combine the conventional hydro-physical-biogeochemical modeling with the modern machine learning methods.
We believe that the better approach is to develop a digital twin of a reservoir of interest and then use it for various applications including HABs prediction.
When we say digital twin, we mean a hydro-physical-biogeochemical model that is powered by ML in several ways:

1. Correction of hydro-physical-biogeochemical model results to match observations (data assimilation).
2. Identification of the best parameters of a hydro-physical-biogeochemical model for different applications (e.g. prediction of algal blooms).
