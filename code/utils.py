import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import re

from itertools import product

import statsmodels.api as sm
from scipy.stats.mstats import winsorize
from linearmodels.panel import PanelOLS
import pyfixest as pf
from pyfixest import feols

from tqdm import tqdm

from importlib import reload

figures_path = "/Users/vbp/Downloads"


###### Functions to load data


def load_baci_data(data_folder):
    baci_folder = os.path.join(data_folder, "raw", "baci")
    country_codes_file = "country_codes_V202301.csv"
    product_codes_file = "product_codes_HS92_V202301.csv"

    baci_files = [
        os.path.join(baci_folder, f)
        for f in os.listdir(baci_folder)
        if f.endswith(".csv") and f not in [country_codes_file, product_codes_file]
    ]

    dfs = []
    for file in baci_files:
        # Read CSV and rename columns
        df = pd.read_csv(
            file,
            header=0,
            names=["year", "exporter", "importer", "hscode", "value", "quantity"],
        )

        # Format HS code as 6 digits
        df["hscode"] = df["hscode"].apply(lambda x: str(x).zfill(6))

        dfs.append(df)
    df = pd.concat(dfs)
    df.sort_values(by=["year", "exporter", "importer", "hscode"], inplace=True)
    df.rename(columns={"hscode": "HS6"}, inplace=True)

    for n in [2, 4]:
        df[f"HS{n}"] = df.HS6.apply(lambda x: x[:n])

    df["quantity"] = pd.to_numeric(df.quantity, errors="coerce")
    df = df.dropna(subset=["quantity", "value"])

    return df


def load_gdp_data(data_folder):
    file = os.path.join(
        data_folder,
        "raw",
        "worldbank",
        "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_88",
        "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_88.csv",
    )
    df = pd.read_csv(file, header=2)
    df = pd.melt(
        df,
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="year",
        value_name="gdp",
    )
    df["year"] = pd.to_numeric(df.year, errors="coerce")
    df.dropna(subset=["year", "gdp"], inplace=True)

    country_codes = pd.read_csv(
        os.path.join(data_folder, "raw", "baci", "country_codes_V202301.csv")
    )
    df = df.merge(
        country_codes, left_on="Country Code", right_on="iso_3digit_alpha", how="inner"
    )

    return df[["iso_3digit_alpha", "country_code", "year", "gdp"]]


def get_naics_subcodes(df):
    df["naics3"] = pd.to_numeric(df.naics.apply(lambda x: str(x)[:3]), errors="coerce")
    df["naics4"] = pd.to_numeric(df.naics.apply(lambda x: str(x)[:4]), errors="coerce")
    df["naics5"] = pd.to_numeric(df.naics.apply(lambda x: str(x)[:5]), errors="coerce")
    df["naics"] = pd.to_numeric(df.naics, errors="coerce")
    return df


def load_sic_hs_crosswalk(data_folder):
    merge_keys_path = os.path.join(
        data_folder,
        "raw",
        "original",
        "pierce_schott_JESM_2012",
        "hs_sic_naics_exports_89_121_20220627.dta",
    )

    # Merge in aggregation level keys code
    merge_keys = pd.read_stata(merge_keys_path)
    merge_keys = merge_keys[
        (merge_keys.year == 1992) & (~merge_keys.sic.isin(["", "."]))
    ].copy()

    merge_keys["HS10"] = (
        merge_keys["commodity"].astype(int).apply(lambda x: str(x).zfill(10))
    )
    merge_keys["HS6"] = merge_keys["HS10"].apply(lambda x: x[:6])
    merge_keys = merge_keys[["HS6", "sic", "sic_matchtype"]].drop_duplicates(
        subset=["HS6", "sic"]
    )

    for n in [2, 4]:
        merge_keys[f"HS{n}"] = merge_keys.HS6.apply(lambda x: x[:n])

    merge_keys["sic"] = pd.to_numeric(merge_keys.sic, errors="coerce")
    merge_keys = merge_keys.dropna()

    return merge_keys


def load_naics_hs_crosswalk(data_folder):
    merge_keys_path = os.path.join(
        data_folder,
        "raw",
        "original",
        "pierce_schott_JESM_2012",
        "hs_sic_naics_exports_89_121_20220627.dta",
    )

    # Merge in aggregation level keys code
    merge_keys = pd.read_stata(merge_keys_path)
    # XXX HS 853180 and 380993 are missing from 1992 data. Should we use the values from later years?
    merge_keys = merge_keys[
        (merge_keys.year == 1992) & (~merge_keys.naics.isin(["", "."]))
    ].copy()

    merge_keys["HS10"] = (
        merge_keys["commodity"].astype(int).apply(lambda x: str(x).zfill(10))
    )
    merge_keys["HS6"] = merge_keys["HS10"].apply(lambda x: x[:6])
    merge_keys = merge_keys[["HS6", "naics", "naics_matchtype"]].drop_duplicates(
        subset=["HS6", "naics"]
    )

    # Extract 3,4,5 digits of naics code and store in separate variables (for some sectors only have first digits of NAICS)
    merge_keys = get_naics_subcodes(merge_keys)

    for n in [2, 4]:
        merge_keys[f"HS{n}"] = merge_keys.HS6.apply(lambda x: x[:n])

    return merge_keys


###### Functions for data manipulation
def get_lag(df, group_cols, shift_col="value", shift_amt=1):
    """
    Add a column 'L_value' to the dataframe that contains the 'value'
    from the previous year for each importer. If the previous year
    data is missing, L_value will be NaN.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'importer', 'year', and 'value'.

    Returns:
        pd.DataFrame: The DataFrame with the new 'L_value' column.
    """
    # Create a copy of the relevant columns
    df_prev = df[group_cols + [shift_col]].copy()

    # Increment the year by 1 to align previous year's value with current year
    df_prev["year"] += shift_amt

    # Rename the column so it doesn't clash with the original 'value'
    if shift_amt < 0:
        prefix = f"F{-shift_amt}_"
    else:
        prefix = f"L{shift_amt}_"

    df_prev.rename(columns={f"{shift_col}": f"{prefix}{shift_col}"}, inplace=True)

    # Merge the original DataFrame with the shifted DataFrame on 'importer' and 'year'
    df_merged = df.merge(df_prev, on=group_cols, how="left")

    return df_merged


def get_product_prices(baci_us, agg_level="HS4"):
    # Get totals by product, year
    df = baci_us.groupby([agg_level, "year"])[["value", "quantity"]].sum().reset_index()
    # df = baci_us.copy()
    df["price"] = df["value"] / df["quantity"]
    # df = (
    #     df.groupby([agg_level, "year"])
    #     .agg({"quantity": "sum", "value": "sum", "price": "mean"})
    #     .reset_index()
    # )

    return df


###### Functions for statistical analysis


def newey_west_cov(X, resid, lags=1):
    """
    Compute Newey-West covariance matrix.
    Assumes X is an (n x k) matrix, resid is a length-n vector,
    and the observations are ordered appropriately.
    """

    n, k = X.shape
    XTX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros((k, k))

    # Add the zero-lag (squared residuals)
    for t in range(n):
        S += resid[t] ** 2 * np.outer(X[t], X[t])

    # Add lagged terms
    for l in range(1, lags + 1):
        Gamma = np.zeros((k, k))
        for t in range(l, n):
            Gamma += np.outer(X[t], X[t - l]) * resid[t] * resid[t - l]
        weight = 1 - l / (lags + 1)
        S += weight * (Gamma + Gamma.T)

    return XTX_inv @ S @ XTX_inv


def regfe(
    data,
    dep_var,
    ind_vars=None,
    fe_vars=None,
    robust=True,
    newey=False,
    cluster=None,  # can be str or list of str
    cluster_type="CRV1",  # 'CRV1' or 'CRV3'
    vcov_extra=None,
    weight_col=None,
):
    """
    Run a fixed effects regression using the pyfixest package.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        dep_var (str): Name of the dependent variable.
        ind_vars (list of str, optional): List of independent variable names.
            If None, the model will include only fixed effects (and a constant).
        fe_vars (str or list of str, optional): Fixed effect variable(s) to be absorbed.
        robust (bool): If True, compute heteroskedasticity-robust standard errors
            (only used if neither `cluster` nor `vcov_extra` is provided).
        cluster (str or list of str, optional): Variable(s) for cluster-robust SEs.
            Overrides `robust` if present (and if `vcov_extra` is None).
        cluster_type (str): Either "CRV1" (classic) or "CRV3" (small sample correction).
        vcov_extra (dict or str, optional): If provided, overrides `robust` and `cluster`.
        weights (str or pd.Series, optional): Weights for the regression.

    Returns:
        results: The fitted regression results object from pyfixest.
    """

    # Build the base formula for the independent variables.
    if ind_vars:
        base_formula = f"{dep_var} ~ " + " + ".join(ind_vars)
    else:
        base_formula = f"{dep_var} ~ 1"  # model with constant only

    # Append fixed effects if provided.
    if fe_vars:
        if isinstance(fe_vars, str):
            fe_vars = [fe_vars]
        fe_string = " + ".join(fe_vars)
        formula = base_formula + " | " + fe_string
    else:
        formula = base_formula

    print("Using formula:", formula)

    # Determine the vcov option
    if vcov_extra is not None:
        # highest priority
        vcov_option = vcov_extra
    elif cluster is not None:
        # Use either CRV1 or CRV3
        vcov_option = {cluster_type: " + ".join(cluster)}
    elif robust:
        # heteroskedasticity-robust
        vcov_option = "hetero"
    else:
        # classical (iid) errors
        vcov_option = None

    # Build the keyword arguments to pass to feols.
    kwargs = {}
    if vcov_option is not None:
        kwargs["vcov"] = vcov_option

    if weight_col is not None:
        kwargs["weights"] = weight_col

    # Run the regression
    results = feols(formula, data=data, **kwargs)

    return results


###### Functions for graphing


def demean_by_fixed_effects(df, x_var, fe_vars, weight_col=None):
    result = regfe(df, x_var, fe_vars=fe_vars, robust=False, weight_col=weight_col)
    return result.resid() + df[x_var].mean()


def binscatter_plot(
    data,
    x_var,
    y_var,
    num_bins=20,
    fe_vars=None,
    weights=None,
    connect_dots=False,
    filename=None,
):
    """
    Create a binscatter plot by binning the x variable and averaging the y variable within each bin.

    Optionally, if fixed effects are provided, the function de-means both x and y by these fixed effects
    before binning. If a weights column is provided, the averages are computed as weighted means.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        x_var (str): Name of the variable for the x-axis.
        y_var (str): Name of the variable for the y-axis.
        num_bins (int): Number of bins into which the x variable is divided. Default is 20.
        fe_vars (str or list of str, optional): Fixed effect variable(s). If provided, both x and y are
            residualized by these fixed effects.
        weights (str, optional): Column name containing the weights. If provided, weighted averages are used.
        connect_dots (bool): Whether to connect the binned points with a line.
        filename (str, optional): If provided, saves the plot to this filename.
    """
    # If fixed effects are to be removed, work with a DataFrame that includes them.
    if fe_vars is not None:
        # If a single fixed effect is provided as a string, convert it to a list.
        if isinstance(fe_vars, str):
            fe_vars = [fe_vars]
        cols = [x_var, y_var] + fe_vars
        if weights is not None:
            cols.append(weights)
        df = data[cols].dropna().copy()

        # Replace x_var and y_var with their demeaned versions.
        df[x_var] = demean_by_fixed_effects(df, x_var, fe_vars, weight_col=weights)
        df[y_var] = demean_by_fixed_effects(df, y_var, fe_vars, weight_col=weights)
    else:
        cols = [x_var, y_var]
        if weights is not None:
            cols.append(weights)
        df = data[cols].dropna().copy()

    # Create quantile-based bins that have an equal number of data points.
    df["bin"] = pd.qcut(df[x_var], q=num_bins, duplicates="drop")

    # Group the data by bins and compute the average (weighted if weights provided) of x and y within each bin.
    if weights is None:
        grouped = df.groupby("bin").agg({x_var: "mean", y_var: "mean"}).dropna()
    else:
        grouped = (
            df.groupby("bin")
            .apply(
                lambda g: pd.Series(
                    {
                        x_var: np.average(g[x_var], weights=g[weights]),
                        y_var: np.average(g[y_var], weights=g[weights]),
                    }
                )
            )
            .dropna()
        )

    # Plot the averaged points.
    plt.figure(figsize=(8, 6))
    plt.scatter(
        grouped[x_var], grouped[y_var], s=50, color="blue", edgecolor="k", alpha=0.7
    )

    if connect_dots:
        plt.plot(grouped[x_var], grouped[y_var], color="red", linewidth=1, alpha=0.7)

    # Label axes and add a title.
    xlabel = x_var if fe_vars is None else f"{x_var} (demeaned)"
    ylabel = y_var if fe_vars is None else f"{y_var} (demeaned)"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = "Binscatter Plot"
    if fe_vars is not None:
        title += " (Fixed Effects Demeaned)"
    if weights is not None:
        title += " (Weighted)"
    plt.title(title)

    if filename:
        plt.savefig(os.path.join("figures_path", filename), bbox_inches="tight")
