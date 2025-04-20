from utils import *
import utils
import usfirms.regressions as rr


def get_country_results(demand_shocks, data_folder):
    result_dict = {}
    for code in tqdm(country_codes):
        results = rr.RegressionResults(demand_shocks, country_code=code)
        result_dict[code] = results

    period0 = list(range(2001, 2008))
    period1 = list(range(2015, 2022))

    # Run regressions for each country
    data = {
        "country_code": country_codes,
        "elas_coef": [],
        "elas_se": [],
    }

    for col in ["price", "quantity", "value"]:
        for period in ["", "_period0", "_period1"]:
            data[f"{col}{period}_coef"] = []
            data[f"{col}{period}_se"] = []

    for code in tqdm(country_codes):
        results = result_dict[code]

        # Calculate supply elasticity
        elas, se = results.get_elasticity()
        data["elas_coef"].append(elas)
        data["elas_se"].append(se)

        # Get responsiveness of each variable
        for col in ["price", "quantity", "value"]:
            model = utils.panel_reg(
                results.data,
                f"d_log_{col}_win",
                f"shock_win",
                time_fe=True,
                group_fe=True,
                newey=True,
                newey_lags=4,
            )

            data[f"{col}_coef"].append(model.params["shock_win"])
            data[f"{col}_se"].append(model.std_errors["shock_win"])

            # Repeat separately for each period
            for i, period in enumerate([period0, period1]):
                sub = results.data[
                    results.data.index.get_level_values("year").isin(period)
                ]

                nw = float(
                    np.minimum(4, len(sub.index.get_level_values("year").unique()) - 1)
                )

                model = utils.panel_reg(
                    sub,
                    f"d_log_{col}_win",
                    f"shock_win",
                    time_fe=True,
                    group_fe=True,
                    newey=True,
                    newey_lags=nw,
                )

                data[f"{col}_period{i}_coef"].append(model.params["shock_win"])
                data[f"{col}_period{i}_se"].append(model.std_errors["shock_win"])

    # %%
    elas_df = pd.DataFrame(data)
    country_codes_df = pd.read_csv(f"{data_folder}/raw/baci/country_codes_V202301.csv")
    elas_df = elas_df.merge(country_codes_df, on="country_code", how="inner")

    # %% Compare price responsiveness with quantity responsiveness
    elas_df["weight"] = 1 / elas_df.price_se**2 + 1 / elas_df.quantity_se**2

    return elas_df


def bar_plots(elas_df):

    for col in ["price", "quantity", "value"]:
        elas_df = elas_df[elas_df[f"{col}_se"] < np.abs(elas_df[f"{col}_coef"])].copy()

    for col in ["elas", "price", "quantity", "value"]:
        elas_df = elas_df.sort_values(by=f"{col}_coef")

        plt.figure(figsize=(10, 6))
        plt.bar(
            elas_df.country_name_abbreviation,
            elas_df[f"{col}_coef"],
            yerr=1.96 * elas_df[f"{col}_se"],
            capsize=5,
            color="skyblue",
            edgecolor="black",
        )
        plt.ylabel(f"{col.title()} Coefficient")
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.savefig(f"{figures_folder}/{col}_coef_by_country.png")


def double_bar_plots(elas_df_full):

    for var in ["price", "quantity"]:
        elas_df = elas_df_full.copy()
        elas_df = elas_df.sort_values(by=f"{var}_coef")
        elas_df = elas_df[1.65 * elas_df[f"{var}_se"] < np.abs(elas_df[f"{var}_coef"])]

        # %%
        x = np.arange(len(elas_df))
        width = 0.35  # width of the bars

        plt.figure(figsize=(10, 6))
        # Plot first set of bars
        bars1 = plt.bar(
            x - width / 2,
            elas_df.price_coef,
            width,
            yerr=1.96 * elas_df.price_se,
            capsize=5,
            label="Price",
            color="skyblue",
            edgecolor="black",
        )
        # Plot second set of bars
        bars2 = plt.bar(
            x + width / 2,
            elas_df.quantity_coef,
            width,
            yerr=1.96 * elas_df.quantity_se,
            capsize=5,
            label="Quantity",
            color="salmon",
            edgecolor="black",
        )

        # Labeling the plot
        plt.xlabel("")
        plt.ylabel("Coefficient")
        plt.xticks(
            x, elas_df.country_name_abbreviation, rotation=75
        )  # Rotate tick labels if needed
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_folder}/coefs_by_country_{var}sort.png")

        # %%


def elas_vs_concentration(elas_df, data_folder):
    # %% Plot elasticity against country level concentration ratio
    concentration_df = pd.read_excel(
        f"{data_folder}/raw/worldbank/wits/WITS-Country-Timeseries.xlsx",
        sheet_name="Country-Timeseries",
    )

    # Rename certain countries
    rename_dict = {
        "Czech Republic": "Czechia",
        "Korea, Rep.": "Rep. of Korea",
        "Slovak Republic": "Slovakia",
        "United States": "USA",
    }
    concentration_df["Country Name"] = concentration_df["Country Name"].replace(
        rename_dict
    )

    # Take mean of all columns formatted as \d{4} (excluding nans)
    concentration_df["hhi"] = concentration_df[
        [
            col
            for col in concentration_df.columns
            if re.search(r"\d{4}", col) and int(col) < 2000
        ]
    ].mean(axis=1)

    df = elas_df.merge(
        concentration_df[["Country Name", "hhi"]],
        left_on="country_name_abbreviation",
        right_on="Country Name",
        how="left",
        indicator=True,
    )
    df["weight"] = 1 / df.price_se**2
    sub = df[
        (df.price_se < np.abs(df.price_coef)) & (df.hhi < 0.5) & (df.price_coef > 0)
    ]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sub.hhi, sub.price_coef)
    for idx, row in sub.iterrows():
        plt.annotate(
            row["iso_3digit_alpha"],
            (row["hhi"], row["price_coef"]),
            textcoords="offset points",  # how to position the text
            xytext=(5, 5),  # distance from the point (x,y)
            ha="center",
        )  # horizontal alignment

    plt.savefig(f"{figures_folder}/elas_HH_cross_country.png")
