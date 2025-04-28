from utils import *
import utils


def estimate_industry_duration(data_folder, weight_col="MV"):

    # Load monetary policy shocks
    mp_shocks = pd.read_stata(f"{klms_folder}/proc/master_fomc_level_24.dta")
    mp_shocks = (
        mp_shocks[mp_shocks.unscheduled_meetings == 0][["daten", "mp_klms_U"]]
        .copy()
        .rename(columns={"daten": "date"})
    )

    # Load firm data
    df = pd.read_stata(f"{klms_folder}/proc/master_firm_level_24.dta")
    df = (
        df[["permno", "daten", "shock_hf_30min", "MV"]]
        .copy()
        .rename(columns={"daten": "date"})
    )
    df.dropna(subset=["shock_hf_30min", "MV"], inplace=True)
    df = df.merge(mp_shocks, on="date", how="inner")

    # Merge with product codes
    firm_hscodes = pd.read_csv(os.path.join(data_folder, "working", "firm_hscodes.csv"))
    firm_hscodes["HS4_codes"] = firm_hscodes["HS4_codes"].str.findall(r"\d+")
    firm_hscodes = firm_hscodes.explode("HS4_codes")
    firm_hscodes.dropna(subset=["HS4_codes"], inplace=True)
    df = df.merge(
        firm_hscodes[["permno", "HS4_codes"]],
        on="permno",
        how="inner",
    )

    df = df.set_index(["permno", "date"])
    coef = []
    se = []

    hs_codes = df.HS4_codes.unique()
    for code in tqdm(hs_codes):
        df_sub = df[df.HS4_codes == code]

        if len(df_sub) < 5:
            coef.append(np.nan)
            se.append(np.nan)
            continue

        model = utils.panel_reg(
            df_sub,
            f"shock_hf_30min",
            f"mp_klms_U",
            time_fe=False,
            group_fe=False,
            weight_col=weight_col,
        )

        coef.append(model.params["mp_klms_U"])
        se.append(model.std_errors["mp_klms_U"])

    output = pd.DataFrame(
        {
            "HS4": hs_codes,
            "beta": coef,
            "beta_var": np.array(se) ** 2,
        }
    )
    output.to_csv(f"{data_folder}/working/HS4_duration.csv", index=False)
    return output


def duration_elas_regressions(data_folder, results_HS4):
    durs_dict = {}
    for weight_col in ["MV"]:
        duration = estimate_industry_duration(data_folder, weight_col=weight_col)
        durs_dict[weight_col] = duration

    elas_dict = {}
    for outcome_var in ["d_log_quantity_dm_year", "d_log_price_dm_year"]:
        elas = results_HS4.product_responsiveness(outcome_var=outcome_var)
        elas_dict[outcome_var] = elas

    elas_dict["IV"] = results_HS4.product_elasticity()

    # %%
    for outcome_var in ["d_log_quantity_dm_year", "d_log_price_dm_year", "IV"]:
        elas = elas_dict[outcome_var]
        for weight_col in ["MV"]:
            duration = durs_dict[weight_col]
            df = elas.merge(duration, on="HS4")
            df["weight"] = 1 / (df.beta_var + df.sigma_var)

            var = outcome_var.replace("d_log_", "").replace("_dm_year", "")
            if weight_col == "MV":
                filename = (
                    f"{figures_folder}/HS4_{var}_response_duration_weighted_diff1.png"
                )

            else:
                filename = f"{figures_folder}/HS4_{var}_response_duration_diff1.png"

            utils.binscatter_plot(
                df, "sigma", "beta", weights="weight", filename=filename
            )

            model = sm.WLS(
                df["beta"],
                sm.add_constant(df["sigma"]),
                weights=df["weight"],
                missing="drop",
            ).fit()

            if weight_col == "MV":
                print(outcome_var)
                print("--" * 20)
                print(model.summary())
