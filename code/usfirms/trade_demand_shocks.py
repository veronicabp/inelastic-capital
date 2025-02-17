# %%
from utils import *


def load_gdp_data():
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

    return df[["iso_3digit_alpha", "country_code", "year", "gdp"]].copy()


def load_baci_data(us_only=False):
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

        if us_only:
            # Keep only exports from US
            df = df[df["exporter"] == 842]

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


def load_susb_data(naics_digits=6):
    files = ["us_state_naics_2002.txt", "us_state_6digitnaics_2007.txt"]
    dfs = []
    for file in files:
        susb_df = pd.read_csv(os.path.join(data_folder, "raw", "census", "susb", file))

        susb_df["naics_num"] = pd.to_numeric(susb_df["NAICS"], errors="coerce")
        susb_df = susb_df.dropna(subset=["naics_num"])

        susb_df = susb_df[
            (susb_df.STATE == 0)
            & (susb_df.NAICS.str.len() == naics_digits)
            & (susb_df.ENTRSIZEDSCR == "Total")
        ]
        susb_df.columns = susb_df.columns.str.lower()
        susb_df["year"] = int(file[-8:-4])
        dfs.append(susb_df)
    susb_df = pd.concat(dfs)
    susb_df.sort_values(by=["state", "naics", "year"], inplace=True)
    for col in ["empl", "estb", "payr", "rcpt"]:
        susb_df[f"d_log_{col}"] = np.log(susb_df[col]) - np.log(
            susb_df.groupby(["state", "naics"])[col].shift()
        )
        susb_df.loc[np.abs(susb_df[f"d_log_{col}"]) == np.inf, f"d_log_{col}"] = np.nan

    susb_df = (
        susb_df.groupby(["state", "naics"])[
            [col for col in susb_df.columns if col.startswith("d_log")] + ["estb"]
        ]
        .mean()
        .reset_index()
    )

    susb_df["naics"] = susb_df["naics"].astype(int)

    if naics_digits < 6:
        susb_df.rename(columns={"naics": f"naics{naics_digits}"}, inplace=True)

    return susb_df


def load_wrds_data(
    agg_level="naics", outcomes=["saleq", "assets", "market_value", "Q", "capital"]
):
    wrds = pd.read_stata(os.path.join(data_folder, "working", "firms.dta"))

    if agg_level == "naics":
        wrds["naics"] = wrds.groupby(["gvkey", "year"])["naics"].transform("first")
        wrds = wrds[wrds.naics != ""].copy()
        wrds["naics"] = wrds["naics"].astype(int)
        # Keep cases for which we have 6-digit naics code
        wrds[wrds.naics >= 111110]

    elif agg_level == "HS4":
        # Get firm - product merge keys
        HS_merge_keys = pd.read_csv(
            os.path.join(data_folder, "working", "firm_hscodes.csv")
        )
        # Drop firms that do not produce goods
        HS_merge_keys = HS_merge_keys[HS_merge_keys.HS4_codes != "[]"]

        # Expand to have a row per product
        HS_merge_keys["HS4_codes"] = HS_merge_keys.HS4_codes.apply(
            lambda x: re.findall(r"\d+", x)
        )
        HS_merge_keys = HS_merge_keys.explode("HS4_codes")[
            ["gvkey", "HS4_codes"]
        ].rename(columns={"HS4_codes": "HS4"})

        wrds["gvkey"] = wrds["gvkey"].astype(int)
        wrds = wrds.merge(HS_merge_keys, on="gvkey", how="inner")

    # Save firm names
    firm_names = wrds[["gvkey", "conml"]].drop_duplicates(subset="gvkey")

    # Collapse at yearly firm level
    wrds["year"] = wrds.datadate.dt.year
    wrds = (
        wrds.groupby(["gvkey", "datadate", "year", agg_level])[outcomes]
        .mean()
        .reset_index()
    )
    wrds = wrds.groupby(["gvkey", "year", agg_level])[outcomes].sum().reset_index()
    wrds = wrds.rename(
        columns={col: col[:-1] for col in wrds.columns if col.endswith("q")}
    )

    wrds.sort_values(by=[agg_level, "gvkey", "year"], inplace=True)

    for col in ["sale", "assets", "market_value", "capital"]:
        wrds[f"d_log_{col}"] = np.log(wrds[col]) - np.log(
            wrds.groupby([agg_level, "gvkey"])[col].shift()
        )
        wrds.loc[np.abs(wrds[f"d_log_{col}"]) == np.inf, f"d_log_{col}"] = np.nan

    # XXX Convert NAICS to 1992 for consistency

    if agg_level == "naics":
        wrds = get_naics_subcodes(wrds)
    wrds = wrds.merge(firm_names, on=["gvkey"])

    return wrds


def get_naics_subcodes(df):
    df["naics3"] = pd.to_numeric(df.naics.apply(lambda x: str(x)[:3]), errors="coerce")
    df["naics4"] = pd.to_numeric(df.naics.apply(lambda x: str(x)[:4]), errors="coerce")
    df["naics5"] = pd.to_numeric(df.naics.apply(lambda x: str(x)[:5]), errors="coerce")
    df["naics"] = pd.to_numeric(df.naics, errors="coerce")
    return df


def get_naics_merge_keys():
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

    return merge_keys


def get_product_prices(baci_us, agg_level="HS4"):
    # Get totals by product, year
    df = baci_us.groupby([agg_level, "year"])[["value", "quantity"]].sum().reset_index()
    df["price"] = df["value"] / df["quantity"]

    return df


def get_gvkey_merge_keys():
    merge_keys = None
    return merge_keys


def get_historical_shares(baci, pre_period_years=[1995], agg_level="naics"):
    historical_shares = baci[baci.year.isin(pre_period_years)]
    historical_shares = (
        historical_shares.groupby(["importer", agg_level])["value"].sum().reset_index()
    )

    historical_shares["share"] = historical_shares.value / historical_shares.groupby(
        agg_level
    )["value"].transform("sum")

    return historical_shares


def get_importer_growth(baci=None, timeseries_variation="tot_imports"):

    if timeseries_variation == "tot_imports":
        df = baci.groupby(["importer", "year"])["value"].sum().reset_index()
        df["log_val"] = np.log(df.value)

    elif timeseries_variation == "gdp":
        df = load_gdp_data()
        df.rename(columns={"country_code": "importer"}, inplace=True)
        df["log_val"] = np.log(df.gdp)

    df["log_growth"] = df["log_val"] - df.groupby("importer")["log_val"].shift(1)
    df.dropna(subset="log_growth", inplace=True)
    return df[["importer", "year", "log_growth"]]


def produce_demand_shocks(
    baci_all,
    agg_level="naics",
    pre_period_years=[1995],
    timeseries_variation="tot_imports",
):
    baci = baci_all[baci_all.exporter == 842].copy()  # US exports

    if "naics" in agg_level:
        merge_keys = get_naics_merge_keys()
        baci = baci.merge(merge_keys, on=["HS6"], how="inner")

    ###############################
    # Construct demand shocks
    ###############################
    baci = baci.dropna(subset=agg_level)

    # Get historical export shares
    historical_shares = get_historical_shares(
        baci, pre_period_years=pre_period_years, agg_level=agg_level
    )

    # Get annual growth rate in exports for each country
    importer_growth = get_importer_growth(
        baci=baci_all, timeseries_variation=timeseries_variation
    )

    # Bartik instrument: Z_k = Sum_i (E_ik / E_k) * (X_i)
    df = historical_shares[["importer", agg_level, "share"]].merge(
        importer_growth[["importer", "year", "log_growth"]], on="importer"
    )
    df["shock"] = df["share"] * df["log_growth"]
    df = df.groupby([agg_level, "year"])["shock"].sum().reset_index()
    return df


def create_panel(df, bartik_shocks, agg_level="naics"):
    # Get weighted average sales by year, industry
    difference_cols = [col for col in df.columns if col.startswith("d_log")]
    industry_agg = weighted_mean(
        df,
        ["year", agg_level],
        difference_cols + ["market_value"],
        "market_value",
    )

    # Merge
    df = industry_agg.merge(bartik_shocks, on=[agg_level, "year"], how="inner")
    return df


def panel_regression(df):
    for fe in [["year"], ["year", "HS4"], ["year", "HS4", "gvkey"]]:
        print("\n\nUnweighted:")
        result = regfe(df, "d_log_sale", ["shock"], fe_vars=fe, cluster=["year", "HS4"])
        print(result.summary())

        print("\n\nWeighted:")
        result = regfe(
            df[df.market_value > 1],
            "d_log_sale",
            ["shock"],
            fe_vars=fe,
            weights="market_value",
            cluster=["year", "HS4"],
        )
        print(result.summary())


def cross_sectional_regressions(
    df, start_year=2000, end_year=2007, agg_level="naics", tag=""
):
    difference_cols = [col for col in df.columns if col.startswith("d_log")]

    # Regressions aggregating multiple years (look at boom period)
    df = df[(df.year >= start_year) & (df.year <= end_year)]
    df = (
        df.groupby([agg_level])[["shock"] + difference_cols + ["market_value"]]
        .mean()
        .reset_index()
    )

    for col in difference_cols:
        print(f"\n\n{col}\n{'='*len(col)}\n")
        X = sm.add_constant(df["shock"])
        y = df[col]

        print("Unweighted:")
        result = sm.OLS(y, X).fit(cov_type="HC3")
        print(result.summary())

        print("\n\nWeighted:")
        result = sm.WLS(y, X, weights=df.market_value).fit(cov_type="HC3")
        print(result.summary())

        field = col.replace("d_log_", "")
        binscatter_plot(
            df,
            "shock",
            col,
            filename=f"{agg_level}_{field}_shock{tag}_binscatter.png",
            num_bins=100,
        )


def census_regressions(
    bartik_shocks, agg_level="naics", start_year=2002, end_year=2007, tag=""
):
    naics_digits = int(agg_level[-1]) if agg_level != "naics" else 6
    susb_df = load_susb_data(naics_digits=naics_digits)

    df = bartik_shocks[
        (bartik_shocks.year >= start_year) & (bartik_shocks.year <= end_year)
    ]
    df = df.groupby([agg_level])[["shock"]].sum().reset_index()
    df = df.merge(susb_df, on=[agg_level], how="inner")

    X = sm.add_constant(df["shock"])
    y = df["d_log_rcpt"]

    result = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")
    print("Unweighted:\n", result.summary())

    result = sm.WLS(y, X, weights=df["estb"], missing="drop").fit(cov_type="HC3")
    print("Weight:\n", result.summary())

    binscatter_plot(
        df,
        "shock",
        "d_log_rcpt",
        filename=f"census_rcpt_shock{tag}_binscatter.png",
        num_bins=100,
    )
    binscatter_plot(
        df,
        "shock",
        "d_log_empl",
        filename=f"census_empl_shock{tag}_binscatter.png",
        num_bins=100,
    )


def first_stage():
    agg_level = "HS4"

    # Produce bartik shocks
    baci = load_baci_data()
    baci.to_pickle(os.path.join(data_folder, "working", "baci.p"))

    bartik_shocks = produce_demand_shocks(baci, agg_level=agg_level)

    ################## Using WRDS data ##################
    # Load WRDS data
    wrds = load_wrds_data(agg_level=agg_level)
    wrds.dropna(subset="d_log_sale", inplace=True)

    firm_panel = wrds.merge(bartik_shocks, on=[agg_level, "year"], how="inner")
    panel = create_panel(wrds, bartik_shocks, agg_level=agg_level)

    # Panel regressions
    panel_regression(firm_panel)

    # Cross sectional regressions
    cross_sectional_regressions(panel, agg_level=agg_level)

    ################## Price decomposition ##################
    HS4_prices = get_product_prices(baci[baci.exporter == 842])
    HS4_prices = HS4_prices.sort_values(by=["HS4", "year"])
    HS4_prices["d_log_product_price"] = np.log(HS4_prices.price) - np.log(
        HS4_prices.groupby(["HS4"])["price"].shift()
    )
    df = panel.merge(
        HS4_prices[["year", "HS4", "d_log_product_price"]],
        on=["year", "HS4"],
        how="inner",
    )

    df["d_log_quantity"] = df["d_log_sale"] - df["d_log_product_price"]
    df_collapsed = df[(df.year >= 2000) & (df.year <= 2007)]
    df_collapsed = (
        df.groupby(["HS4"])[
            ["d_log_sale", "d_log_product_price", "d_log_quantity", "shock"]
        ]
        .sum()
        .reset_index()
    )

    binscatter_plot(
        df_collapsed,
        "shock",
        "d_log_quantity",
        filename="HS4_quantity_shock_binscatter",
        num_bins=100,
    )
    binscatter_plot(
        df_collapsed,
        "shock",
        "d_log_product_price",
        filename="HS4_prodprice_shock_binscatter",
        num_bins=100,
    )

    ################## Using census data ##################
    census_regressions(bartik_shocks)


def baci_price_decomposition(baci):
    timeseries_variation = "tot_imports"
    agg_level = "HS6"
    bartik_shocks = produce_demand_shocks(
        baci, agg_level=agg_level, timeseries_variation=timeseries_variation
    )

    prices_quantities = get_product_prices(
        baci[baci.exporter == 842], agg_level=agg_level
    )
    df = prices_quantities.merge(bartik_shocks, on=[agg_level, "year"], how="left")

    df = df.sort_values(by=[agg_level, "year"])
    for col in ["value", "quantity", "price"]:
        df[f"d_log_{col}"] = np.log(df[col]) - np.log(
            df.groupby([agg_level])[col].shift()
        )
        df[f"d5_log_{col}"] = np.log(df[col]) - np.log(
            df.groupby([agg_level])[col].shift(4)
        )

    for col in ["d_log_value", "d_log_quantity", "d_log_price"]:
        df[f"{col}5"] = df.groupby(agg_level)[col].transform(
            lambda s: s.rolling(window=5, min_periods=5).sum()
        )

    df = df.dropna(subset="d_log_value")
    df = df.dropna(subset="shock")

    # Look at 2000-2007 period
    df_collapsed = df[(df.year >= 2000) & (df.year <= 2007)]

    df_collapsed = (
        df_collapsed.groupby(agg_level)[
            [
                "d_log_value",
                "d_log_quantity",
                "d_log_price",
                "shock",
                "value",
                "quantity",
                "price",
            ]
        ]
        .mean()
        .reset_index()
    )

    for col in ["d_log_value", "d_log_quantity", "d_log_price"]:
        binscatter_plot(
            df_collapsed,
            "shock",
            col,
            num_bins=100,
            weight="value",
            filename=f"baci_{col}_shock_2000-2007_binscatter.png",
        )

    # Plot estimate by year
    df["year5"] = np.round(df.year / 5) * 5

    df_collapsed = (
        df.groupby(["HS6", "year5"])[
            ["d_log_value", "d_log_quantity", "d_log_price", "shock"]
        ]
        .sum()
        .reset_index()
    )
    for col in ["d_log_value", "d_log_quantity", "d_log_price"]:
        years = []
        ests = []
        ses = []
        for year in sorted(df.year5.unique()):
            years.append(year)

            sub = df_collapsed[df_collapsed.year5 == year]
            X = sm.add_constant(sub["shock"])
            y = sub[col]

            result = sm.OLS(y, X).fit()
            ests.append(result.params["shock"])
            ses.append(result.bse["shock"])

        ests = np.array(ests)
        ses = np.array(ses)

        plt.figure(figsize=(8, 6))
        plt.plot(years, ests)
        plt.plot(years, ests + 1.96 * ses, alpha=0.5, color="gray")
        plt.plot(years, ests - 1.96 * ses, alpha=0.5, color="gray")
        plt.axhline(0, color="black")
        plt.savefig(
            os.path.join(figures_path, f"baci_shock_effect_by_year5_{col}.png"),
            bbox_inches="tight",
        )


def descriptive_stats(baci):
    # Plot of variation in import shares across 2-digit NAICS industries
    historical_shares = get_historical_shares(baci)

    historical_shares["naics2"] = historical_shares.naics.apply(lambda x: str(x)[:2])
    historical_shares["naics3"] = historical_shares.naics.apply(lambda x: str(x)[:3])

    sectors_to_highlight = {
        "333": "Machinery Manufacturing",
        "334": "Computer and Electronic Product Manufacturing",
        "311": "Food Manufacturing",
        "325": "Chemical Manufacturing",
        "332": "Fabricated Metal Product Manufacturing",
        "336": "Transportation Equipment Manufacturing",
    }

    historical_shares = (
        historical_shares.groupby(["naics3", "importer"])["value"].sum().reset_index()
    )
    historical_shares["share"] = historical_shares.value / historical_shares.groupby(
        "naics3"
    )["value"].transform("sum")

    country_codes = pd.read_csv(
        os.path.join(data_folder, "raw", "baci", "country_codes_V202301.csv")
    )
    historical_shares = historical_shares.merge(
        country_codes, left_on="importer", right_on="country_code", how="left"
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    axes = axes.flatten()  # Flatten the axes array to make it easier to iterate

    # Loop through each NAICS code and plot the top 10 countries
    for i, (naics_code, naics_name) in enumerate(sectors_to_highlight.items()):
        # Filter the dataframe for the current NAICS code
        filtered_df = historical_shares[historical_shares["naics3"] == naics_code]

        # Get the top 10 countries by share
        top_importers = filtered_df.nlargest(8, "share").sort_values(by="share")

        # Create the plot on the corresponding subplot
        ax = axes[i]
        bars = ax.bar(
            np.arange(len(top_importers)), top_importers["share"], width=0.6
        )  # Set bar width

        # Add annotations for the share values
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.2f}",  # Format the number to 2 decimal places
                ha="center",
                va="bottom",
            )  # Position the text at the top of the bar

        ax.set_title(naics_name)
        ax.set_xticks(np.arange(len(top_importers)))
        ax.set_xticklabels(top_importers["iso_3digit_alpha"], rotation=45)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_path, "export_decomposition.png"), bbox_inches="tight"
    )

    ## Timeseries
    baci["quantity"] = pd.to_numeric(baci.quantity, errors="coerce")
    df = baci.dropna(subset="quantity")
    product_series = (
        df.groupby(["HS2", "HS6", "year"])[["value", "quantity"]].sum().reset_index()
    )
    H2_series = (
        product_series.groupby(["HS2", "year"])[["value", "quantity"]]
        .sum()
        .reset_index()
    )

    series = H2_series.groupby(["year"])[["value", "quantity"]].sum().reset_index()
    series["price"] = series.value / series.quantity

    plt.figure(figsize=(8, 6))
    plt.plot(series.year, series.value)

    plt.figure(figsize=(8, 6))
    plt.plot(series.year, series.quantity)


# TO DO:
# How to think about changes in NAICS/HS versions over time
# Correlate elasticity with some basic index of market power
# Cluster SE
# %%
