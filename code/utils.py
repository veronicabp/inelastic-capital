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
from linearmodels.iv import Interaction, AbsorbingLS

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

un_iso_map = {
    "Afghanistan": "AFG",
    "Albania": "ALB",
    "Algeria": "DZA",
    "Andorra": "AND",
    "Angola": "AGO",
    "Anguilla": "AIA",
    "Antigua and Barbuda": "ATG",
    "Argentina": "ARG",
    "Armenia": "ARM",
    "Aruba": "ABW",
    "Australia": "AUS",
    "Austria": "AUT",
    "Azerbaijan": "AZE",
    "Bahamas": "BHS",
    "Bahrain": "BHR",
    "Bangladesh": "BGD",
    "Barbados": "BRB",
    "Belarus": "BLR",
    "Belgium": "BEL",
    "Belize": "BLZ",
    "Benin": "BEN",
    "Bermuda": "BMU",
    "Bhutan": "BTN",
    "Bolivia (Plurinational State of)": "BOL",
    "Bosnia and Herzegovina": "BIH",
    "Botswana": "BWA",
    "Brazil": "BRA",
    "British Virgin Islands": "VGB",
    "Brunei Darussalam": "BRN",
    "Bulgaria": "BGR",
    "Burkina Faso": "BFA",
    "Burundi": "BDI",
    "Cabo Verde": "CPV",
    "Cambodia": "KHM",
    "Cameroon": "CMR",
    "Canada": "CAN",
    "Cayman Islands": "CYM",
    "Central African Republic": "CAF",
    "Chad": "TCD",
    "Chile": "CHL",
    "China (mainland)": "CHN",
    "China, Hong Kong SAR": "HKG",
    "China, Macao Special Administrative Region": "MAC",
    "Colombia": "COL",
    "Comoros": "COM",
    "Congo": "COG",
    "Cook Islands": "COK",
    "Costa Rica": "CRI",
    "Croatia": "HRV",
    "Cuba": "CUB",
    "Curaçao": "CUW",
    "Cyprus": "CYP",
    "Czechia": "CZE",
    "Côte d'Ivoire": "CIV",
    "Democratic People's Republic of Korea": "PRK",
    "Democratic Republic of the Congo": "COD",
    "Denmark": "DNK",
    "Djibouti": "DJI",
    "Dominica": "DMA",
    "Dominican Republic": "DOM",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ",
    "Eritrea": "ERI",
    "Estonia": "EST",
    "Ethiopia": "ETH",
    "Fiji": "FJI",
    "Finland": "FIN",
    "France": "FRA",
    "French Polynesia": "PYF",
    "Gabon": "GAB",
    "Gambia": "GMB",
    "Georgia": "GEO",
    "Germany": "DEU",
    "Ghana": "GHA",
    "Greece": "GRC",
    "Greenland": "GRL",
    "Grenada": "GRD",
    "Guatemala": "GTM",
    "Guinea": "GIN",
    "Guinea-Bissau": "GNB",
    "Guyana": "GUY",
    "Haiti": "HTI",
    "Honduras": "HND",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "India": "IND",
    "Indonesia": "IDN",
    "Iran, Islamic Republic of": "IRN",
    "Iraq": "IRQ",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Jamaica": "JAM",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kingdom of Eswatini": "SWZ",
    "Kiribati": "KIR",
    "Kosovo": "XKX",
    "Kuwait": "KWT",
    "Kyrgyzstan": "KGZ",
    "Lao People's Democratic Republic": "LAO",
    "Latvia": "LVA",
    "Lebanon": "LBN",
    "Lesotho": "LSO",
    "Liberia": "LBR",
    "Libya": "LBY",
    "Liechtenstein": "LIE",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Madagascar": "MDG",
    "Malawi": "MWI",
    "Malaysia": "MYS",
    "Maldives": "MDV",
    "Mali": "MLI",
    "Malta": "MLT",
    "Marshall Islands": "MHL",
    "Mauritania": "MRT",
    "Mauritius": "MUS",
    "Mexico": "MEX",
    "Micronesia (Federated States of)": "FSM",
    "Monaco": "MCO",
    "Mongolia": "MNG",
    "Montenegro": "MNE",
    "Montserrat": "MSR",
    "Morocco": "MAR",
    "Mozambique": "MOZ",
    "Myanmar": "MMR",
    "Namibia": "NAM",
    "Nauru": "NRU",
    "Nepal": "NPL",
    "Netherlands": "NLD",
    "New Caledonia": "NCL",
    "New Zealand": "NZL",
    "Nicaragua": "NIC",
    "Niger": "NER",
    "Nigeria": "NGA",
    "Norway": "NOR",
    "Oman": "OMN",
    "Pakistan": "PAK",
    "Palau": "PLW",
    "Panama": "PAN",
    "Papua New Guinea": "PNG",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Puerto Rico": "PRI",
    "Qatar": "QAT",
    "Republic of Korea": "KOR",
    "Republic of Moldova": "MDA",
    "Republic of North Macedonia": "MKD",
    "Romania": "ROU",
    "Russian Federation": "RUS",
    "Rwanda": "RWA",
    "Saint Kitts and Nevis": "KNA",
    "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT",
    "Samoa": "WSM",
    "San Marino": "SMR",
    "Sao Tome and Principe": "STP",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Seychelles": "SYC",
    "Sierra Leone": "SLE",
    "Singapore": "SGP",
    "Sint Maarten (Dutch part)": "SXM",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Solomon Islands": "SLB",
    "Somalia": "SOM",
    "South Africa": "ZAF",
    "South Sudan": "SSD",
    "Spain": "ESP",
    "Sri Lanka": "LKA",
    "State of Palestine": "PSE",
    "Sudan": "SDN",
    "Suriname": "SUR",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Syrian Arab Republic": "SYR",
    "Tajikistan": "TJK",
    "Thailand": "THA",
    "Timor-Leste": "TLS",
    "Togo": "TGO",
    "Tonga": "TON",
    "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN",
    "Turkmenistan": "TKM",
    "Turks and Caicos Islands": "TCA",
    "Tuvalu": "TUV",
    "Türkiye": "TUR",
    "Uganda": "UGA",
    "Ukraine": "UKR",
    "United Arab Emirates": "ARE",
    "United Kingdom of Great Britain and Northern Ireland": "GBR",
    "United Republic of Tanzania: Mainland": "TZA",
    "United Republic of Tanzania: Zanzibar": "TZA",
    "United States": "USA",
    "Uruguay": "URY",
    "Uzbekistan": "UZB",
    "Vanuatu": "VUT",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Viet Nam": "VNM",
    "Yemen": "YEM",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE",
}


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
    cols = list(set(group_cols + [shift_col] + ["year"]))
    df_prev = df[cols].copy()

    # Increment the year by 1 to align previous year's value with current year
    df_prev["year"] += shift_amt

    # Rename the column so it doesn't clash with the original 'value'
    if shift_amt < 0:
        prefix = f"F{-shift_amt}_"
    else:
        prefix = f"L{shift_amt}_"

    df_prev.rename(columns={f"{shift_col}": f"{prefix}{shift_col}"}, inplace=True)

    # Merge the original DataFrame with the shifted DataFrame on 'importer' and 'year'
    df_merged = df.merge(df_prev, on=list(set(group_cols + ["year"])), how="left")

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
        data[col] = demean(data, col, time_fe=time_fe, group_fe=group_fe)

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


def demean(df, x_var, controls=[], time_fe=True, group_fe=True, weight_col=None):
    result = panel_reg(
        df,
        x_var,
        ind_vars=controls,
        time_fe=time_fe,
        group_fe=group_fe,
        weight_col=weight_col,
    )
    return result.resids + df[x_var].mean()


def binscatter_plot(
    data,
    x_var,
    y_var,
    num_bins=20,
    time_fe=False,
    group_fe=False,
    controls=[],
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
    cols = [x_var, y_var] + controls
    if weights is not None:
        cols.append(weights)
    df = data[cols].dropna().copy()

    # If fixed effects are to be removed, work with a DataFrame that includes them.
    if time_fe or group_fe or controls:
        # Replace x_var and y_var with their demeaned versions.
        df[x_var] = demean(
            df,
            x_var,
            weight_col=weights,
            time_fe=time_fe,
            group_fe=group_fe,
            controls=controls,
        )
        df[y_var] = demean(
            df,
            y_var,
            weight_col=weights,
            time_fe=time_fe,
            group_fe=group_fe,
            controls=controls,
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


def clean_GDP_by_ind(dfname):
    dfname["Description"] = dfname["Description"].str.strip()

    dfname["Description"] = dfname["Description"].replace(
        {
            "National defense": "Federal general government (defense)",
            "Nondefense": "Federal general government (nondefense)",
            "Housing": "Housing Services",
            "Other real estate": "Other Real Estate",
        }
    )
    # first instance of "Government enterprises" is "Federal government enterprises", second instance is "State and local government enterprises"
    # General government: "Federal general government", then "State and local general government"
    for base in ["Government enterprises", "General government"]:
        mask = (dfname["Description"] == base) & (
            dfname["Description"].duplicated(keep="first")
        )
        dfname.loc[mask, "Description"] = base + ".1"

    dfname["Description"] = dfname["Description"].replace(
        {
            "Government enterprises": "Federal government enterprises",
            "Government enterprises.1": "State and local government enterprises",
            "General government": "Federal general government",
            "General government.1": "State and local general government",
        }
    )
    return dfname


def split_whitespace_column(df, col, n):
    """
    Split df[col] (a whitespace-separated string) into exactly n columns.
    """
    # 1) strip surrounding quotes
    s = df[col].astype(str).str.strip('"')
    # 2) split on any run of whitespace, expand into DataFrame
    parts = s.str.split(r"\s+", expand=True)
    # 3) pad (or truncate) to exactly n columns
    parts = parts.reindex(columns=range(n), fill_value="")
    # 4) rename
    parts.columns = [f"{col}_{i+1}" for i in range(n)]
    # 5) drop original and concat
    df2 = pd.concat([df.drop(columns=[col]), parts], axis=1)
    return df2


def load_iso_codes(data_folder):
    # 1) load census ISO‐numeric codes
    iso_path = os.path.join(data_folder, "raw", "iso", "iso_census.txt")
    iso = pd.read_csv(iso_path, sep="|", header=0)
    # rename "Code" column "isonumber", rename "Name" column "country"
    iso.columns = iso.columns.str.strip()
    iso.rename(columns={"Code": "isonumber", "Name": "country"}, inplace=True)
    iso["isonumber"] = iso["isonumber"].str.strip()
    iso["country"] = iso["country"].str.strip()
    iso = iso[iso["isonumber"].str.len() <= 4]  # drop overly long codes

    # 2) load PWT country list
    pwt_path = os.path.join(data_folder, "raw", "pwt", "pwt100.dta")
    pwt = pd.read_stata(pwt_path, columns=["countrycode", "country"])
    pwt["country"] = pwt["country"].str.strip()
    pwt = pwt.drop_duplicates(subset="country")

    # 3) harmonize PWT names to match census list
    name_map = {
        "Bolivia (Plurinational State of)": "Bolivia",
        "Brunei Darussalam": "Brunei",
        "China, Hong Kong SAR": "Hong Kong",
        "China, Macao SAR": "Macao",
        "Congo": "Congo, Republic of the Congo",
        "Curaçao": "Curacao",
        "Côte d'Ivoire": "Cote d'Ivoire",
        "D.R. of the Congo": "Congo, Democratic Republic of the Congo (formerly Za",
        "Denmark": "Denmark, except Greenland",
        "Germany": "Germany (Federal Republic of Germany)",
        "Iran (Islamic Republic of)": "Iran",
        "Lao People's DR": "Laos (Lao People's Democratic Republic)",
        "Myanmar": "Burma (Myanmar)",
        "Republic of Korea": "South Korea (Republic of Korea)",
        "Republic of Moldova": "Moldova (Republic of Moldova)",
        "Russian Federation": "Russia",
        "Sint Maarten (Dutch part)": "Sint Maarten",
        "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
        "State of Palestine": "West Bank administered by Israel",
        "Syrian Arab Republic": "Syria (Syrian Arab Republic)",
        "U.R. of Tanzania: Mainland": "Tanzania (United Republic of Tanzania)",
        "United States": "United States of America",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Viet Nam": "Vietnam",
        "Yemen": "Yemen (Republic of Yemen)",
    }
    pwt["country"] = pwt["country"].replace(name_map)

    # 4) merge PWT ↔ census
    df = pd.merge(pwt, iso, on="country", how="left", validate="1:1")

    # 5) fill in any PWT codes that are still missing
    manual_code_map = {
        "Afghanistan": "AFG",
        "American Samoa": "ASM",
        "Andorra": "ADO",
        "British Indian Ocean Territory": "GBR",
        "Christmas Island (in the Indian Ocean)": "AUS",
        "Cocos (Keeling) Islands": "CRI",
        "Cook Islands": "COK",
        "Cuba": "CUB",
        "Eritrea": "ERI",
        "Falkland Islands (Islas Malvinas)": "ARG",
        "Faroe Islands": "FRO",
        "French Guiana": "GUF",
        "French Polynesia": "PYF",
        "French Southern and Antarctic Lands": "FRA",
        "Gaza Strip administered by Israel": "WBG",
        "Gibraltar": "GIBRA",
        "Greenland": "GRL",
        "Guadeloupe": "GLP",
        "Guam": "GUM",
        "Heard Island and McDonald Islands": "AUS",
        "Kiribati": "KIR",
        "Kosovo": "KSV",
        "Libya": "LBY",
        "Liechtenstein": "LIE",
        "Marshall Islands": "MHL",
        "Martinique": "MTQ",
        "Mayotte": "MYT",
        "Micronesia, Federated States of": "FSM",
        "Monaco": "MCO",
        "Nauru": "NRU",
        "New Caledonia": "NCL",
        "Niue": "NIU",
        "Norfolk Island": "AUS",
        "North Korea (Democratic People's Republic of Korea)": "PRK",
        "Northern Mariana Islands": "MNP",
        "Palau": "PLW",
        "Papua New Guinea": "PNG",
        "Pitcairn Islands": "GBR",
        "Puerto Rico": "PRI",
        "Reunion": "REU",
        "Saint Helena": "GBR",
        "Saint Pierre and Miquelon": "FRA",
        "Samoa (Western Samoa)": "WSM",
        "San Marino": "SMR",
        "Solomon Islands": "SLB",
        "Somalia": "SOM",
        "Svalbard and Jan Mayen": "NOR",
        "Timor-Leste": "TMP",
        "Tokelau": "NZL",
        "Tonga": "TON",
        "Tuvalu": "TUV",
        "United States Minor Outlying Islands": "USA",
        "Vanuatu": "VUT",
        "Virgin Islands of the United States": "VIR",
        "Wallis and Futuna": "FRA",
        "Sint Maarten": "DNK",
        "Curacao": "NLD",
        "Romania": "ROM",
        "West Bank administered by Israel": "WBG",
        "Congo, Democratic Republic of the Congo (formerly Za": "ZAR",
    }
    mask = df["country"].isin(manual_code_map)
    df.loc[mask, "countrycode"] = df.loc[mask, "country"].map(manual_code_map)

    # 6) drop any truly missing
    df = df[df["countrycode"].notna() & (df["countrycode"] != "")]

    # 7) final rename
    df = df.rename(columns={"countrycode": "country_code"})

    df["isonumber"] = df["isonumber"].astype(int)

    return df[["country", "country_code", "isonumber"]]


# %%
