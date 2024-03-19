import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def extract_stations_rho(ds: xr.Dataset, xis: list, etas: list):
    datasets = []
    for xi, eta in zip(xis, etas):
        datasets.append(ds.isel(xi_rho=xi, eta_rho=eta))
    return xr.concat([*datasets], dim="station")


def extract_stations_u(ds: xr.Dataset, xis: list, etas: list):
    datasets = []
    for xi, eta in zip(xis, etas):
        datasets.append(ds.isel(xi_u=xi, eta_u=eta-1))
    return xr.concat([*datasets], dim="station")


def extract_stations_v(ds: xr.Dataset, xis: list, etas: list):
    datasets = []
    for xi, eta in zip(xis, etas):
        datasets.append(ds.isel(xi_v=xi-1, eta_v=eta))
    return xr.concat([*datasets], dim="station")


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


def plot_variable(variable: pd.DataFrame):
    y = variable.index.values
    x = np.arange(variable.shape[1])
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(15, 5))
    cf = ax.contourf(X, Y, variable.values)
    fig.colorbar(cf, ax=ax, location='right', pad=0.01)
