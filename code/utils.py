import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import os
import re

from itertools import product

import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
import pyfixest as pf
from pyfixest import feols
from scipy import stats
from scipy.stats.mstats import winsorize
from stargazer.stargazer import Stargazer

import requests
import json

from tqdm import tqdm

tqdm.pandas()

from importlib import reload

import warnings
from linearmodels.panel.model import MissingValueWarning

warnings.filterwarnings("ignore", category=MissingValueWarning)

data_folder = os.path.join("..", "data")
figures_folder = "/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Apps/Overleaf/Inelastic Capital/Figures"

klms_folder = "/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Princeton/MianSufiRoll/round2_response/data_2024_update/data"

# %% OECD country codes
oecd_codes = [
    36,  # Australia
    40,  # Austria
    56,  # Belgium
    124,  # Canada
    152,  # Chile
    203,  # Czech Republic
    208,  # Denmark
    233,  # Estonia
    246,  # Finland
    251,  # France
    276,  # Germany
    300,  # Greece
    348,  # Hungary
    352,  # Iceland
    372,  # Ireland
    376,  # Israel
    380,  # Italy
    392,  # Japan
    410,  # Korea
    428,  # Latvia
    440,  # Lithuania
    442,  # Luxembourg
    484,  # Mexico
    528,  # Netherlands
    554,  # New Zealand
    579,  # Norway
    616,  # Poland
    620,  # Portugal
    703,  # Slovakia
    705,  # Slovenia
    724,  # Spain
    752,  # Sweden
    757,  # Switzerland
    792,  # Turkey
    826,  # United Kingdom
    842,  # United States
]

brics_codes = [
    76,  # Brazil
    156,  # China
    643,  # Russia
    699,  # India
    710,  # South Africa
]

country_codes = oecd_codes + brics_codes


###### Functions to load data
def load_ppi_data(data_folder):
    return pd.read_stata(f"{data_folder}/working/naics_ppi_yearly.dta")


def load_baci_data(data_folder):
    """
    Load BACI data from the raw folder and return a DataFrame with the following columns:
    - year
    - exporter
    - importer
    - HS6
    - value
    - quantity
    """
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


def load_exchange_rate_data(data_folder):
    df = pd.read_csv(
        os.path.join(
            data_folder,
            "raw",
            "unstats",
            "exchange_rates.csv",
        )
    )
    # Convert AMA exchange rate to numeric
    df["AMA exchange rate"] = pd.to_numeric(df["AMA exchange rate"], errors="coerce")
    df["exchange_rate"] = 1 / df["AMA exchange rate"]
    df.dropna(subset=["exchange_rate"], inplace=True)
    df = (
        df[["Country/Area", "Year", "exchange_rate"]]
        .copy()
        .rename(columns={"Country/Area": "country_name", "Year": "year"})
    )

    country_mapping = {
        "Bolivia (Plurinational State of)": "Plurinational State of Bolivia",
        "Bosnia and Herzegovina": "Bosnia Herzegovina",
        "China (mainland)": "China",
        "China, Hong Kong SAR": "China, Hong Kong Special Administrative Region",
        "Former Czechoslovakia": "Czechoslovakia",
        "Former Ethiopia": "Ethiopia",
        "Former Netherlands Antilles": "Netherlands Antilles",
        "Former Yugoslavia": "Serbia and Montenegro",
        "France": "France, Monaco",
        "Iran, Islamic Republic of": "Iran",
        "Kingdom of Eswatini": "Swaziland",
        "Kosovo": "Serbia and Montenegro",
        "Lao People's Democratic Republic": "Lao People's Dem. Rep.",
        "Liechtenstein": "Switzerland, Liechtenstein",
        "Micronesia (Federated States of)": "Federated State of Micronesia",
        "Monaco": "France, Monaco",
        "Norway": "Norway, Svalbard and Jan Mayen",
        "Puerto Rico": "USA, Puerto Rico and US Virgin Islands",
        "Republic of North Macedonia": "The Former Yugoslav Republic of Macedonia",
        "Sint Maarten (Dutch part)": "Saint Maarten (Dutch part)",
        "Switzerland": "Switzerland, Liechtenstein",
        "Syrian Arab Republic": "Syria",
        "Türkiye": "Turkey",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "United Republic of Tanzania: Mainland": "United Republic of Tanzania",
        "United Republic of Tanzania: Zanzibar": "United Republic of Tanzania",
        "United States": "USA, Puerto Rico and US Virgin Islands",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Yemen: Former Democratic Yemen": "Yemen",
        "Yemen: Former Yemen Arab Republic": "Yemen",
    }

    # Assuming your dataframe is named df and the country column is named 'country'
    df["country_name"] = df["country_name"].replace(country_mapping)

    baci_country_codes = pd.read_csv(
        os.path.join(data_folder, "raw", "baci", "country_codes_V202301.csv")
    )
    df = df.merge(
        baci_country_codes,
        left_on=["country_name"],
        right_on=["country_name_full"],
        how="left",
        indicator=True,
    )

    return df[["country_code", "year", "exchange_rate"]].copy()


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

    merge_keys = merge_keys.drop_duplicates(subset=["HS6", "sic"])
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


def weighted_median_price(sub_df, col="price", weight_column="quantity"):
    """
    Calculate the weighted median price of the column using the specified weight column.
    """
    # Sort by price
    sub_df = sub_df.sort_values(col)
    # Calculate the cumulative sum of the weights (quantity)
    cumulative = sub_df[weight_column].cumsum()
    # Define the cutoff (half of the total weight)
    cutoff = sub_df[weight_column].sum() / 2.0
    # Return the price where the cumulative weight meets/exceeds the cutoff
    return sub_df.loc[cumulative >= cutoff, col].iloc[0]


def weighted_mean(df, col="price", weight_column="quantity"):
    """
    Calculate the weighted mean of the column using the specified weight column.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'price' and the weight column.
        weight_column (str): Name of the column to use as weights (default is 'quantity').

    Returns:
        float: Weighted mean of the col.
    """
    # Ensure the weight column is not zero to avoid division errors
    total_weight = df[weight_column].sum()
    if total_weight == 0:
        return float("nan")

    return (df[col] * df[weight_column]).sum() / total_weight


def get_product_prices(df, groupby=["exporter", "HS6"]):
    """
    Calculate the price of each product for each exporter.
    """
    # Average price
    # df = baci_us.groupby([agg_level, "year"])[["value", "quantity"]].sum().reset_index()
    # df["price"] = df["value"] / df["quantity"]

    # Median price
    df["price"] = df["value"] / df["quantity"]
    grouped = df.groupby(groupby + ["year"])
    df = grouped.progress_apply(
        lambda sub_df: pd.Series(
            {
                "quantity": sub_df["quantity"].sum(),
                "value": sub_df["value"].sum(),
                "price": weighted_median_price(sub_df),
            }
        )
    ).reset_index()
    return df


def construct_relative_prices(df, agg_level="HS6"):
    grouped = (
        df.groupby([agg_level, "year"])
        .progress_apply(
            lambda sub_df: weighted_mean(sub_df, weight_column="quantity", col="price")
        )
        .reset_index()
        .rename(columns={0: "price_mn"})
    )
    df = df.merge(grouped, on=[agg_level, "year"], how="left")

    df["relative_price"] = df["price"] / df["price_mn"]
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


def panel_reg(
    data,
    dep_var,
    ind_vars=[],
    time_fe=True,
    group_fe=True,
    robust=True,
    newey=False,
    cluster=None,
    weight_col=None,
    newey_lags=1,
):

    # Optionally include weights if provided
    if weight_col is not None:
        weights = data[weight_col]
    else:
        weights = None

    # Create the exogenous variables (adding a constant)
    exog = sm.add_constant(data[ind_vars])

    # Set up the PanelOLS model, with both entity and time fixed effects if requested.
    model = PanelOLS(
        data[dep_var],
        exog,
        weights=weights,
        entity_effects=group_fe,
        time_effects=time_fe,
    )

    # Set standard errors based on parameters
    kwargs = {}
    if newey:
        # Use a kernel covariance estimator (Bartlett kernel) for Newey–West style errors.
        kwargs["cov_type"] = "kernel"
        kwargs["kernel"] = "bartlett"
        kwargs["bandwidth"] = newey_lags
    elif cluster is not None:
        # Cluster the standard errors on the specified variable.
        kwargs["cov_type"] = "clustered"
        kwargs["clusters"] = data[cluster]
    elif robust:
        # Use robust (heteroskedasticity-consistent) standard errors.
        kwargs["cov_type"] = "robust"
    else:
        kwargs["cov_type"] = "unadjusted"

    results = model.fit(**kwargs)
    return results


def iv_panel_reg(
    df,
    dep_var,
    exog=[],
    endog=[],
    instruments=[],
    time_fe=True,
    group_fe=True,
    robust=True,
    newey=False,
    cluster=None,
    weight_col=None,
    newey_lags=1,
):
    data = df.copy()

    # Optionally include weights if provided
    if weight_col is not None:
        weights = data[weight_col]
    else:
        weights = None

    # De-mean before, since IV module cannot accept fixed effects
    for col in [dep_var] + exog + endog + instruments:
        data[col] = demean_by_fixed_effects(
            data, col, time_fe=time_fe, group_fe=group_fe
        )

    # Run IV
    model = IV2SLS(
        dependent=data[dep_var],
        exog=sm.add_constant(data[exog]),
        endog=data[endog],
        instruments=data[instruments],
        weights=weights,
    )

    # Set standard errors based on parameters
    kwargs = {}
    if newey:
        # Use a kernel covariance estimator (Bartlett kernel) for Newey–West style errors.
        kwargs["cov_type"] = "kernel"
        kwargs["kernel"] = "bartlett"
        kwargs["bandwidth"] = newey_lags
    elif cluster is not None:
        # Cluster the standard errors on the specified variable.
        kwargs["cov_type"] = "clustered"
        kwargs["clusters"] = data[cluster]
    elif robust:
        # Use robust (heteroskedasticity-consistent) standard errors.
        kwargs["cov_type"] = "robust"
    else:
        kwargs["cov_type"] = "unadjusted"

    results = model.fit(**kwargs)
    return results


def panel_reg(
    data,
    dep_var,
    ind_vars=[],
    time_fe=True,
    group_fe=True,
    robust=True,
    newey=False,
    cluster=None,
    weight_col=None,
    newey_lags=1,
):

    # Optionally include weights if provided
    if weight_col is not None:
        weights = data[weight_col]
    else:
        weights = None

    # Create the exogenous variables (adding a constant)
    exog = sm.add_constant(data[ind_vars])

    # Set up the PanelOLS model, with both entity and time fixed effects if requested.
    model = PanelOLS(
        data[dep_var],
        exog,
        weights=weights,
        entity_effects=group_fe,
        time_effects=time_fe,
    )

    # Set standard errors based on parameters
    kwargs = {}
    if newey:
        # Use a kernel covariance estimator (Bartlett kernel) for Newey–West style errors.
        kwargs["cov_type"] = "kernel"
        kwargs["kernel"] = "bartlett"
        kwargs["bandwidth"] = newey_lags
    elif cluster is not None:
        # Cluster the standard errors on the specified variable.
        kwargs["cov_type"] = "clustered"
        kwargs["clusters"] = data[cluster]
    elif robust:
        # Use robust (heteroskedasticity-consistent) standard errors.
        kwargs["cov_type"] = "robust"
    else:
        kwargs["cov_type"] = "unadjusted"

    results = model.fit(**kwargs)
    return results


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
    verbose=False,
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

    if verbose:
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


def demean_by_fixed_effects(df, x_var, time_fe=True, group_fe=True, weight_col=None):
    result = panel_reg(
        df, x_var, time_fe=time_fe, group_fe=group_fe, weight_col=weight_col
    )
    return result.resids + df[x_var].mean()


def binscatter_plot(
    data,
    x_var,
    y_var,
    num_bins=20,
    time_fe=False,
    group_fe=False,
    weights=None,
    connect_dots=False,
    filename=None,
    x_label=None,
    y_label=None,
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
    cols = [x_var, y_var]
    if weights is not None:
        cols.append(weights)
    df = data[cols].dropna().copy()

    # If fixed effects are to be removed, work with a DataFrame that includes them.
    if time_fe or group_fe:
        # Replace x_var and y_var with their demeaned versions.
        df[x_var] = demean_by_fixed_effects(
            df, x_var, weight_col=weights, time_fe=time_fe, group_fe=group_fe
        )
        df[y_var] = demean_by_fixed_effects(
            df, y_var, weight_col=weights, time_fe=time_fe, group_fe=group_fe
        )

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
    plt.xlabel(x_label if x_label else x_var)
    plt.ylabel(y_label if y_label else y_var)

    if filename:
        plt.savefig(os.path.join(figures_folder, filename), bbox_inches="tight")


# %%
