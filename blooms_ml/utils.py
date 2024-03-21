import itertools

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def extract_stations_rho(ds: xr.Dataset, xis: list, etas: list):
    datasets = []
    for xi, eta in zip(xis, etas):
        datasets.append(ds.isel(xi_rho=xi, eta_rho=eta))
    return xr.concat(datasets, dim="station")


def extract_stations_u(ds: xr.Dataset, xis: list, etas: list):
    datasets = []
    for xi, eta in zip(xis, etas):
        datasets.append(ds.isel(xi_u=xi, eta_u=eta-1))
    return xr.concat(datasets, dim="station")


def extract_stations_v(ds: xr.Dataset, xis: list, etas: list):
    datasets = []
    for xi, eta in zip(xis, etas):
        datasets.append(ds.isel(xi_v=xi-1, eta_v=eta))
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
    ds['s_w'] = ds['s_rho'].values
    ds_w = filter_variables_by_dimension(ds, 's_w')
    ds1 = ds_w.rename_dims({'s_w': 's_rho'})
    ds2 = filter_variables_by_dimension(ds, 's_rho')
    return xr.merge([ds1.drop_vars("s_w"), ds2])


def append_rho_profiles(df: pd.DataFrame):
    nlayers = df.reset_index()['s_rho'].nunique()
    nstations = df.reset_index()['station'].nunique()
    dataframes = []
    # loop through stations to pivot tables and attach rho profiles to all layers
    for station in range(nstations):
        df_station = df.loc[df.index.get_level_values('station') == station]
        df_station = df_station.reset_index()
        rho = df_station.pivot(index='ocean_time', columns='s_rho', values='rho')
        rho = rho.loc[rho.index.repeat(nlayers)]
        rho = rho.rename_axis(None, axis=1)
        rho = rho.reset_index()
        df_station = df_station.drop(columns=['station'])
        dataframes.append(pd.concat([df_station, rho], axis=1))
    return pd.concat(dataframes, axis=0)


def plot_variable(variable: pd.DataFrame):
    y = variable.index.values
    x = np.arange(variable.shape[1])
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(15, 5))
    cf = ax.contourf(X, Y, variable.values)
    fig.colorbar(cf, ax=ax, location='right', pad=0.01)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    from: www.kaggle.com .. credit-fraud-dealing-with-imbalanced-datasets
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
