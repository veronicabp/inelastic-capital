import utils
from utils import *


class RegressionResults:
    def __init__(
        self,
        ds,
        agg_level="HS6",
        use_prev_year=True,
        growth_var="gdp",
        share_var="quantity",
        country_code=842,
    ):
        self.demand_shocks = ds

        self.agg_level = agg_level
        self.use_prev_year = use_prev_year
        self.growth_var = growth_var
        self.share_var = share_var
        self.country_code = country_code

        self.data_folder = self.demand_shocks.data_folder
        self.difference_amt = self.demand_shocks.difference_amt

        self.country_baci = self.demand_shocks.baci[
            self.demand_shocks.baci.exporter == self.country_code
        ].copy()

        self.covariate_cols = [
            "CR4",
            "HHI",
            "Reg Idx",
            "APP",
            "Log Rel Price",
            "Log Rel Price (mean)",
            "beta",
        ]  # pci

        self.figures_folder = "/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Apps/Overleaf/Inelastic Capital/Figures"
        self.tables_folder = "/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Apps/Overleaf/Inelastic Capital/Tables"

        # Set up main data
        # print("Initializing Data")
        self.initialize_data()

        # For US, merge additional data
        if self.country_code == 842:
            self.merge_additional_data()

        # Format as panel
        self.data = self.data.set_index([self.agg_level, "year"])
        self.demean_variables()

        if self.country_code == 842:
            self.create_interactions()

    def initialize_data(self):

        # Merge shocks with price/quantity/export data
        self.bartik_shocks = self.demand_shocks.get_demand_shocks(
            agg_level=self.agg_level,
            use_prev_year=self.use_prev_year,
            growth_var=self.growth_var,
            share_var=self.share_var,
            country_code=self.country_code,
        )

        self.prices_quantities = pd.read_pickle(
            f"{self.data_folder}/working/baci_product_prices_{self.agg_level}.p"
        )
        self.prices_quantities = self.prices_quantities[
            self.prices_quantities.exporter == self.country_code
        ]

        df = self.prices_quantities.merge(
            self.bartik_shocks, on=[self.agg_level, "year"], how="left"
        )

        # Get logs and lags of variables
        df = df.sort_values(by=[self.agg_level, "year"])
        for col in ["value", "quantity", "price"]:
            df[f"log_{col}"] = np.log(df[col])

            df = utils.get_lag(
                df,
                [self.agg_level, "year"],
                shift_col=col,
                shift_amt=self.difference_amt,
            )
            df[f"d_log_{col}"] = np.log(df[col]) - np.log(
                df[f"L{self.difference_amt}_{col}"]
            )

        df["product_mn_val"] = df.groupby([self.agg_level])["log_value"].transform(
            "mean"
        )
        df["weight"] = df["product_mn_val"]
        df.loc[df.weight < 0, "weight"] = 0.0001

        # Winsorize
        for col in [
            col
            for col in df.columns
            if col.startswith("d")
            or col.startswith("shock")
            or col in ["value", "quantity", "price"]
        ]:
            df[f"{col}_win"] = winsorize(df[col], limits=(0.05, 0.05))

        # Drop missing
        df = df.dropna(subset="d_log_value")
        df = df.dropna(subset="shock")

        self.data = df

    def demean_variables(self):
        # De-mean
        for col in ["value", "quantity", "price"]:
            self.data[f"d_log_{col}_dm"] = utils.demean_by_fixed_effects(
                self.data, f"d_log_{col}_win", time_fe=False, group_fe=True
            )

        self.data[f"shock_dm"] = utils.demean_by_fixed_effects(
            self.data, f"shock_win", time_fe=False, group_fe=True
        )

    def merge_additional_data(self):
        self.load_crosswalks()
        self.merge_concentration_ratios()
        self.merge_regulation_index()
        self.merge_duration()
        self.merge_app_measure()
        # self.merge_product_complexity()
        self.merge_capacity_constraints()
        self.merge_relative_product_price()

    def create_interactions(self):
        for col in self.covariate_cols:
            # Normalize column to have mean 0 std dev 1
            self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[
                col
            ].std()

            for shock in ["shock_win", "shock_dm"]:
                self.data[f"{shock}_x_{col}"] = self.data[shock] * self.data[col]

    def load_crosswalks(self):
        self.naics_hs_crosswalk = utils.load_naics_hs_crosswalk(self.data_folder)
        self.naics_hs_crosswalk = self.naics_hs_crosswalk.drop_duplicates(
            subset=[self.agg_level, "naics"]
        )

        self.sic_hs_crosswalk = utils.load_sic_hs_crosswalk(self.data_folder)
        self.sic_hs_crosswalk = self.sic_hs_crosswalk.drop_duplicates(
            subset=[self.agg_level, "sic"]
        )

    def merge_relative_product_price(self):
        # Load relative price data
        relative_prices = pd.read_pickle(
            f"{self.data_folder}/working/baci_relative_prices_{self.agg_level}.p"
        )
        # Keep only data for current country
        relative_prices = relative_prices[relative_prices.exporter == self.country_code]

        # Winsorize to remove extreme outliers
        relative_prices["relative_price"] = winsorize(
            relative_prices["relative_price"], limits=(0.01, 0.01)
        )

        # Take logs
        relative_prices["Log Rel Price"] = np.log(relative_prices["relative_price"])

        # Get average relative price for each product across all years
        relative_prices["Log Rel Price (mean)"] = relative_prices.groupby(
            [self.agg_level]
        )["Log Rel Price"].transform("mean")

        # Merge with main data
        self.data = self.data.merge(
            relative_prices[
                ["year", self.agg_level, "Log Rel Price", "Log Rel Price (mean)"]
            ],
            on=["year", self.agg_level],
            how="left",
        )

        return

    def merge_concentration_ratios(self):
        concentration = pd.read_excel(
            f"{self.data_folder}/raw/census/economic_census/concentration92-47.xls",
            header=3,
        ).rename(
            columns={
                "SIC Code": "sic",
                "4 largest companies": "CR4",
                "Herfindahl-Hirschman Index for 50 largest companies": "HHI",
            }
        )
        concentration = concentration[concentration.YR == 92][["sic", "CR4", "HHI"]]

        concentration = concentration.merge(self.sic_hs_crosswalk, on="sic")
        concentration = (
            concentration.groupby([self.agg_level])[["CR4", "HHI"]].mean().reset_index()
        )

        self.data = self.data.merge(concentration, on=[self.agg_level], how="left")

    def merge_regulation_index(self):
        regulation_probs = pd.read_csv(
            f"{self.data_folder}/raw/quantgov/RegData-US_5-0/usregdata5.csv"
        )
        regulation_naics = pd.read_csv(
            f"{self.data_folder}/raw/quantgov/RegData-US_5-0/regdata_5_0_naics07_3digit.csv"
        )
        regulation = regulation_probs.merge(regulation_naics, on="document_id")
        regulation["naics_restrictions"] = (
            regulation["probability"] * regulation["restrictions"]
        )
        regulation = (
            regulation.groupby("industry")["naics_restrictions"]
            .sum()
            .reset_index()
            .rename(columns={"industry": "naics3"})
        )
        regulation = regulation.merge(
            self.naics_hs_crosswalk.drop_duplicates(subset=["naics3", self.agg_level]),
            on=["naics3"],
            how="inner",
        )
        regulation = (
            regulation.groupby(self.agg_level)["naics_restrictions"]
            .mean()
            .reset_index()
        )
        regulation["Reg Idx"] = np.log(regulation.naics_restrictions)

        self.data = self.data.merge(regulation, on=self.agg_level, how="left")

    def merge_duration(self, naics_col="naics"):
        duration = pd.read_csv(f"{self.data_folder}/working/industry_duration.csv")
        duration.dropna(subset=["beta"], inplace=True)
        duration["beta_weight"] = 1 / duration["beta_var"]
        duration[naics_col] = duration[naics_col].astype(int)
        duration = duration.merge(
            self.naics_hs_crosswalk.drop_duplicates(subset=[naics_col, self.agg_level]),
            on=[naics_col],
            how="inner",
        )

        duration_mn = (
            duration.groupby(self.agg_level)
            .apply(
                lambda sub_df: weighted_mean(
                    sub_df, weight_column="beta_weight", col="beta"
                )
            )
            .reset_index()
            .rename(columns={0: "beta"})
        )

        duration_w = (
            duration.groupby(self.agg_level)
            .apply(
                lambda sub_df: weighted_mean(
                    sub_df, weight_column="beta_weight", col="beta_weight"
                )
            )
            .reset_index()
            .rename(columns={0: "beta_weight"})
        )

        duration = duration_mn.merge(
            duration_w,
            on=self.agg_level,
            how="inner",
        )

        # Winsorize beta
        duration["beta"] = winsorize(duration["beta"], limits=(0.05, 0.05))

        self.data = self.data.merge(duration, on=self.agg_level, how="left")

    def merge_app_measure(self):
        antras_app = pd.read_csv(
            f"{self.data_folder}/raw/original/antras_tubdenov_2025/complete_ranking_usa_goods.csv"
        )
        antras_app = antras_app.merge(
            self.naics_hs_crosswalk.rename(columns={"naics": "naics6"}),
            on=["naics6"],
            how="inner",
        )
        antras_app = (
            antras_app.groupby(self.agg_level)["invtCogsRatio"].mean().reset_index()
        ).rename(columns={"invtCogsRatio": "APP"})

        self.data = self.data.merge(
            antras_app[[self.agg_level, "APP"]], on=self.agg_level, how="left"
        )

    def merge_product_complexity(self):
        pci = pd.read_csv(f"{self.data_folder}/raw/pci/{self.agg_level}.csv").rename(
            columns={
                self.agg_level: "description",
                f"{self.agg_level} ID": self.agg_level,
            }
        )
        pci[self.agg_level] = pci[self.agg_level].apply(
            lambda x: str(x).zfill(int(self.agg_level[-1]))
        )
        year_cols = [col for col in pci.columns if re.search(r"\d{4}", str(col))]
        pci["pci"] = pci[year_cols].sum(axis=1, min_count=len(year_cols)) / len(
            year_cols
        )

        self.data = self.data.merge(
            pci[[self.agg_level, "pci"]], on=self.agg_level, how="left"
        )

    def merge_capacity_constraints(self):
        return

    def panel_regression_table(
        self,
        indep_vars=["shock_win"],
        dep_vars=["value", "quantity", "price"],
        tag="",
        rename_dict={
            "const": "Intercept",
            "shock_win": "Shock",
        },
    ):
        print(f"Running regressions:\n dep_var ~ {'+'.join(indep_vars)}")
        models = []

        # Cols 1-3: Panel regression
        for var in dep_vars:
            model = utils.panel_reg(
                self.data,
                f"d_log_{var}_win",
                indep_vars,
                time_fe=True,
                group_fe=True,
                newey=True,
                newey_lags=4,
            )
            models.append(model)

        # Cols 4-6: Panel (with weights)
        for var in dep_vars:
            model = utils.panel_reg(
                self.data,
                f"d_log_{var}_win",
                indep_vars,
                time_fe=True,
                group_fe=True,
                newey=True,
                newey_lags=4,
                weight_col="weight",
            )
            models.append(model)

        # Export as table
        stargazer = Stargazer(models)
        stargazer.custom_columns(
            [
                "$\\Delta$ V",
                "$\\Delta$ Q",
                "$\\Delta$ P",
                "$\\Delta$ V",
                "$\\Delta$ Q",
                "$\\Delta$ P",
            ]
        )
        stargazer.covariate_order(indep_vars)
        stargazer.rename_covariates(rename_dict)
        stargazer.show_residual_std_err = False
        stargazer.show_f_statistic = False
        stargazer.show_ngroups = False
        stargazer.show_r2 = False

        stargazer.add_line("Weights", ["No", "No", "No", "Yes", "Yes", "Yes"])

        # Table title and label
        stargazer.title_text = f"{self.agg_level} Panel Regressions"
        if tag:
            stargazer.title_text += f" ({tag.replace('_','')} Interaction)"
        stargazer.table_label = f"tab: {self.agg_level} panel {tag}"

        latex_table = stargazer.render_latex().replace("[!htbp]", "[H]")
        with open(
            f"{self.tables_folder}/panel_{self.agg_level}_{self.growth_var}_{self.share_var}_{self.use_prev_year}{tag}.tex",
            "w",
        ) as f:
            f.write(latex_table)

        for model in models:
            print(model.summary)

    def get_elasticity(self):
        model = utils.iv_panel_reg(
            self.data,
            dep_var="d_log_quantity_win",
            exog=[],
            endog=["d_log_price_win"],
            instruments=["shock_win"],
            newey=True,
            newey_lags=4,
        )
        # print(model.summary)

        return model.params["d_log_price_win"], model.std_errors["d_log_price_win"]

    def run_baseline_regressions(self):
        self.panel_regression_table()

        for col in self.covariate_cols:
            self.panel_regression_table(
                indep_vars=["shock_win", f"shock_win_x_{col}"],
                tag=f"_{col}",
                rename_dict={
                    "shock_win": "Shock",
                    f"shock_win_x_{col}": f"Shock x {col}",
                },
            )

        self.annual_regression_plot(dep_var="d_log_value_dm")
        self.annual_regression_plot(dep_var="d_log_quantity_dm")
        self.annual_regression_plot(dep_var="d_log_price_dm")

        self.make_binscatters()

    def annual_regression_plot(self, indep_var="shock_dm", dep_var="d_log_value_dm"):
        years = []
        coeffs = []
        ses = []
        df = self.data.reset_index()

        for year in sorted(df.year.unique()):
            result = utils.regfe(df[df.year == year], dep_var, [indep_var])
            coeffs.append(result.coef()[indep_var])
            ses.append(result.se()[indep_var])
            years.append(year)

        ub = np.array(coeffs) + 1.96 * np.array(ses)
        lb = np.array(coeffs) - 1.96 * np.array(ses)

        plt.figure()
        plt.plot(years, coeffs)
        plt.plot(years, ub, alpha=0.5, color="gray")
        plt.plot(years, lb, alpha=0.5, color="gray")

        plt.xticks([y for y in range(2000, 2025, 5)])
        plt.xlabel("")
        plt.ylabel("Beta")

        plt.axhline(0)

        plt.savefig(
            f"{self.figures_folder}/annual_coef_{dep_var}_{self.agg_level}_{self.growth_var}_{self.share_var}_{self.use_prev_year}.png"
        )

    def make_binscatters(self):
        for col in ["value", "quantity", "price"]:
            utils.binscatter_plot(
                self.data,
                "shock_win",
                f"d_log_{col}_win",
                num_bins=100,
                time_fe=True,
                group_fe=True,
                x_label=f"Demand Shock ({self.difference_amt} yr)",
                y_label=f"Log Change in {col.title()} ({self.difference_amt} yr)",
                filename=f"{self.figures_folder}/binscatter_{col}_panel_FE_{self.agg_level}_{self.growth_var}_{self.share_var}_{self.use_prev_year}.png",
            )

            # Binscatter of interactions
            # Residualize LHS and RHS variables
            model = utils.panel_reg(
                self.data,
                f"d_log_{col}_win",
                ["shock_win"],
                time_fe=True,
                group_fe=True,
                newey=True,
                newey_lags=4,
            )
            self.data[f"d_log_{col}_win_resids"] = model.resids
            for cov_col in self.covariate_cols:

                model = utils.panel_reg(
                    self.data,
                    f"shock_win_x_{cov_col}",
                    ["shock_win"],
                    time_fe=True,
                    group_fe=True,
                    newey=True,
                    newey_lags=4,
                )
                self.data[f"shock_win_x_{cov_col}_resids"] = model.resids

                utils.binscatter_plot(
                    self.data,
                    f"shock_win_x_{cov_col}_resids",
                    f"d_log_{col}_win_resids",
                    num_bins=100,
                    time_fe=True,
                    group_fe=True,
                    x_label=f"Demand Shock ({self.difference_amt} yr) x {cov_col}",
                    y_label=f"Log Change in {col.title()} ({self.difference_amt} yr)",
                    filename=f"{self.figures_folder}/binscatter_{col}_panel_FE_{cov_col}_interaction_{self.agg_level}_{self.growth_var}_{self.share_var}_{self.use_prev_year}.png",
                )

    def persistence(self, rank_agg_level="HS2"):
        if self.agg_level != "HS6":
            print("Persistence analysis is only set up to work for HS6-level data.")
            return

        df = self.data.reset_index()
        df[f"HS2"] = df.HS6.apply(lambda x: x[:2])

        period0 = list(range(2001, 2008))
        period1 = list(range(2015, 2022))

        # Get elas coefficient for each code in both periods
        for col in ["value", "price", "quantity"]:
            coeffs = []
            ses = []
            codes = []
            periods = []
            for code in tqdm(sorted(df[rank_agg_level].unique())):
                for i, period in enumerate([period0, period1]):
                    sub = df[
                        (df.year.isin(period)) & (df[rank_agg_level] == code)
                    ].copy()
                    result = utils.regfe(
                        sub,
                        f"d_log_{col}_dm",
                        ["shock_dm"],
                        fe_vars=["year", rank_agg_level],
                    )
                    coeffs.append(result.coef()["shock_dm"])
                    ses.append(result.se()["shock_dm"])
                    codes.append(code)
                    periods.append(i)

            results = pd.DataFrame(
                {rank_agg_level: codes, "period": periods, "elas": coeffs, "se": ses}
            )
            results = results.pivot(
                index=rank_agg_level, columns="period", values=["elas", "se"]
            ).reset_index()

            results["weight"] = 1 / results["se"][0] ** 2 + 1 / results["se"][1] ** 2

            # WLS regression to get persistence slope
            res = sm.WLS(
                results["elas"][1],
                sm.add_constant(results["elas"][0]),
                weights=results["weight"],
            ).fit()
            slope = res.params[0]
            intercept = res.params["const"]
            se = res.bse[0]

            # Drop very low weighted observations from graph
            results = results[results.weight > results.weight.quantile(0.3)]

            plt.figure()
            x = np.array(results["elas"][0])
            y = np.array(results["elas"][1])

            plt.scatter(
                x,
                y,
                s=results["weight"] * 10,
            )

            # Add line of best fit
            plt.plot(x, slope * np.array(x) + intercept, color="black", ls="--")

            plt.text(
                0.95,
                0.95,
                f"Slope: {slope:.2f} (SE: {se:.2f})",
                horizontalalignment="right",
                verticalalignment="top",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

            plt.xlabel("First Period Coefficient")
            plt.ylabel("Second Period Coefficient")

            plt.savefig(
                f"{self.figures_folder}/HS2_persistence_{col}_{self.growth_var}_{self.share_var}_{self.use_prev_year}.png"
            )

    def bls_price_regressions(self):
        # Load BLS price data
        ppi = utils.load_ppi_data(self.data_folder)
        naics_shocks = self.demand_shocks.get_demand_shocks(agg_level="naics")
        df = naics_shocks.merge(ppi, on=["naics", "year"], how="inner")

        # Create lags
        df = df.sort_values(by=["naics", "year"])
        df[f"log_ppi"] = np.log(df["ppi"])
        df = utils.get_lag(df, ["naics", "year"], shift_col="ppi")
        df = utils.get_lag(
            df, ["naics", "year"], shift_col="ppi", shift_amt=self.difference_amt
        )
        df[f"d_log_ppi"] = np.log(df["ppi"]) - np.log(df[f"L{self.difference_amt}_ppi"])

        # Winsorize
        for col in ["shock", "d_log_ppi"]:
            df[f"{col}_win"] = winsorize(df[col], limits=(0.05, 0.05))

        # Set as panel
        df = df.set_index(["naics", "year"])

        # Binscatter
        utils.binscatter_plot(
            df,
            "shock_win",
            f"d_log_ppi_win",
            num_bins=100,
            time_fe=True,
            group_fe=True,
            x_label=f"Demand Shock ({self.difference_amt} yr)",
            y_label=f"Log Change in Price ({self.difference_amt} yr)",
            filename=f"{self.figures_folder}/binscatter_dppi_panel_FE_{self.agg_level}_{self.growth_var}_{self.share_var}_{self.use_prev_year}.png",
        )

        utils.binscatter_plot(
            df,
            "shock_win",
            f"d_log_ppi_win",
            num_bins=100,
            time_fe=True,
            group_fe=True,
            x_label="Demand Shock",
            y_label=f"Log Change in Price",
            filename=f"{self.figures_folder}/binscatter_d1ppi_panel_FE_{self.agg_level}_{self.growth_var}_{self.share_var}_{self.use_prev_year}.png",
        )

    def naics_level_elasticity(self):
        # Merge in naics data
        df = (
            self.data.reset_index()
            .merge(
                self.naics_hs_crosswalk[
                    [self.agg_level, "naics", "naics3", "naics4", "naics5"]
                ],
                on=self.agg_level,
                how="left",
            )
            .set_index([self.agg_level, "year"])
        )

        data = {"naics3": []}
        for col in ["price", "quantity", "value"]:
            data[f"{col}_coef"] = []
            data[f"{col}_se"] = []

        for naics3 in tqdm(df.naics3.unique()):

            if len(df[df.naics3 == naics3]) < 20:
                continue

            data["naics3"].append(naics3)

            # Get responsiveness of each variable
            for col in ["price", "quantity", "value"]:
                model = utils.panel_reg(
                    df[df.naics3 == naics3],
                    f"d_log_{col}_win",
                    f"shock_win",
                    time_fe=True,
                    group_fe=True,
                    newey=True,
                    newey_lags=4,
                )

                data[f"{col}_coef"].append(model.params["shock_win"])
                data[f"{col}_se"].append(model.std_errors["shock_win"])

        df = pd.DataFrame(data)
