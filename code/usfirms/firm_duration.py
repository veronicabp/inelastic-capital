from utils import *
import utils


def estimate_industry_duration(data_folder, chunksize=10**6):

    # Load monetary policy shocks
    mp_shocks = pd.read_stata(f"{klms_folder}/proc/master_fomc_level_24.dta")
    mp_shocks = (
        mp_shocks[mp_shocks.unscheduled_meetings == 0][["daten", "mp_klms_U"]]
        .copy()
        .rename(columns={"daten": "date"})
    )

    # Load WRDS data and merge
    chunk_list = []
    for chunk in tqdm(
        pd.read_csv(
            f"{data_folder}/raw/wrds/Compustat_Securities_Daily_all.csv",
            chunksize=chunksize,
            usecols=["gvkey", "datadate", "prccd", "ajexdi", "cshoc", "naics", "fic"],
        )
    ):
        # Filter to only US firms
        chunk.drop(chunk[chunk["fic"] != "USA"].index, inplace=True)
        chunk.dropna(subset=["gvkey", "datadate", "prccd", "ajexdi"], inplace=True)

        # Convert datadate to datetime
        chunk["date"] = pd.to_datetime(chunk["datadate"], format="%Y-%m-%d")

        # Calculate adjusted prices and log returns
        chunk["Padj"] = chunk["prccd"] / chunk["ajexdi"]
        chunk.sort_values(["gvkey", "date"], inplace=True)
        chunk["log_ret"] = np.log(
            chunk["Padj"] / chunk.groupby("gvkey")["Padj"].shift(1)
        )

        chunk = chunk[["date", "gvkey", "naics", "log_ret", "cshoc", "prccd"]].merge(
            mp_shocks[["date", "MPS", "mp_klms_filtered"]],
            on="date",
            how="inner",
        )

        chunk_list.append(chunk)

    # Concatenate all processed chunks
    df = pd.concat(chunk_list)
    df.to_csv(f"{data_folder}/working/stock_returns_fomc.csv", index=False)

    # Merge in market value from fundamentals
    fundamentals = pd.read_csv(
        f"{data_folder}/raw/wrds/Compustat_Fundamentals_Quarterly_all.csv",
        usecols=["gvkey", "fyearq", "fqtr", "prccq", "cshoq"],
    )
    fundamentals["market_value"] = fundamentals["prccq"] * fundamentals["cshoq"]
    fundamentals.dropna(subset="market_value", inplace=True)
    fundamentals.drop(
        fundamentals[fundamentals["market_value"] <= 0].index, inplace=True
    )

    # Take mean by year
    fundamentals = (
        fundamentals.groupby(["gvkey", "fyearq"], as_index=False)
        .agg({"market_value": "mean"})
        .reset_index()
    )

    df["fyearq"] = df["date"].dt.year
    df = df.merge(
        fundamentals,
        on=["gvkey", "fyearq"],
        how="inner",
    )

    # Estimate responsiveness to monetary policy shocks at industry level
    df.dropna(subset=["log_ret", "MPS", "naics"], inplace=True)

    # Convert naics to string
    naics_len = 6
    df["naics"] = df["naics"].astype(int).astype(str)
    df["naics"] = df["naics"].str[:naics_len]
    df.drop(df[df["naics"].str.len() != naics_len].index, inplace=True)

    df = df.set_index(["gvkey", "date"])
    coef = []
    se = []

    naics_codes = df.naics.unique()
    for naics in tqdm(naics_codes):
        df_naics = df[df.naics == naics]
        # Get number of unique gvkeys (which is an index)
        if df_naics.index.get_level_values("gvkey").nunique() < 5:
            coef.append(np.nan)
            se.append(np.nan)
            continue

        model = utils.panel_reg(
            df_naics,
            f"log_ret",
            f"MPS",
            time_fe=False,
            group_fe=True,
            weight_col="market_value",
            newey=True,
            newey_lags=4,
        )

        coef.append(model.params["MPS"])
        se.append(model.std_errors["MPS"])

    if naics_len == 6:
        naics_col = "naics"
    else:
        naics_col = f"naics{naics_len}"

    output = pd.DataFrame(
        {
            naics_col: naics_codes,
            "beta": coef,
            "beta_var": np.array(se) ** 2,
        }
    )
    output.to_csv(f"{data_folder}/working/industry_duration.csv", index=False)
