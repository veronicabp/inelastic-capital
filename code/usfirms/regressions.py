import utils
from utils import *


class RegressionResults:
    def __init__(self, ds, agg_level="HS6", use_prev_year=False, growth_var="gdp"):
        self.demand_shocks = ds
        self.agg_level = agg_level
        self.use_prev_year = use_prev_year
        self.growth_var = growth_var
        self.data_folder = self.demand_shocks.data_folder

        self.figures_folder = "/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Apps/Overleaf/Inelastic Capital/Figures"
        self.tables_folder = "/Users/vbp/Princeton Dropbox/Veronica Backer Peral/Apps/Overleaf/Inelastic Capital/Tables"

        # Set up main data
        print("Initializing Data")
        self.initialize_data()

        # Merge in other data
        self.merge_additional_data()

        # Format as panel
        self.data = self.data.set_index([self.agg_level, "year"])
        self.demean_variables()
        self.create_interactions()

        cs_data = self.data.reset_index()
        self.cs_data = cs_data[cs_data.year == 2007]
        self.cs_data = self.cs_data.set_index([self.agg_level, "year"])

    def initialize_data(self):

        # Merge shocks with price/quantity/export data
        self.bartik_shocks = self.demand_shocks.get_demand_shocks(
            agg_level=self.agg_level,
            use_prev_year=self.use_prev_year,
            growth_var=self.growth_var,
        )

        prices_quantities = utils.get_product_prices(
            self.demand_shocks.baci_us_exports, agg_level=self.agg_level
        )
        df = prices_quantities.merge(
            self.bartik_shocks, on=[self.agg_level, "year"], how="left"
        )

        # Get logs and lags of variables
        df = df.sort_values(by=[self.agg_level, "year"])
        for col in ["value", "quantity", "price"]:
            df[f"log_{col}"] = np.log(df[col])

            df = utils.get_lag(df, [self.agg_level, "year"], shift_col=col)
            df[f"d_log_{col}"] = np.log(df[col]) - np.log(df[f"L1_{col}"])

            df = utils.get_lag(df, [self.agg_level, "year"], shift_col=col, shift_amt=5)
            df[f"d5_log_{col}"] = np.log(df[col]) - np.log(df[f"L5_{col}"])

        df["product_mn_val"] = df.groupby([self.agg_level])["log_value"].transform(
            "mean"
        )

        # Aggregate shocks over 5 years
        for i in range(1, 5):
            df = utils.get_lag(
                df, [self.agg_level, "year"], shift_col="shock", shift_amt=i
            )
        shock_cols = [col for col in df.columns if "shock" in col]
        df["shock5"] = df[shock_cols].sum(axis=1, min_count=len(shock_cols))

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
            self.data[f"d5_log_{col}_dm"] = utils.demean_by_fixed_effects(
                self.data, f"d5_log_{col}_win", time_fe=False, group_fe=True
            )
        self.data[f"shock5_dm"] = utils.demean_by_fixed_effects(
            self.data, f"shock5_win", time_fe=False, group_fe=True
        )

    def merge_additional_data(self):
        self.load_crosswalks()
        self.merge_concentration_ratios()
        self.merge_regulation_index()
        self.merge_app_measure()
        self.merge_product_complexity()
        self.merge_capacity_constraints()

    def create_interactions(self):
        covariate_cols = ["CR4", "HHI", "regidx", "app", "pci"]

        for col in covariate_cols:
            # Normalize column to have mean 0 std dev 1
            self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[
                col
            ].std()

            for shock in ["shock5_win", "shock5_dm"]:
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
        regulation["regidx"] = np.log(regulation.naics_restrictions)

        self.data = self.data.merge(regulation, on=self.agg_level, how="left")

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
        ).rename(columns={"invtCogsRatio": "app"})

        self.data = self.data.merge(
            antras_app[[self.agg_level, "app"]], on=self.agg_level, how="left"
        )

    def merge_product_complexity(self):
        pci = pd.read_csv(f"{self.data_folder}/raw/pci/{self.agg_level}.csv").rename(
            columns={
                self.agg_level: "description",
                f"{self.agg_level} ID": self.agg_level,
            }
        )
        pci[self.agg_level] = pci[self.agg_level].apply(lambda x: str(x).zfill(6))
        year_cols = [col for col in pci.columns if re.search(r"\d{4}", str(col))]
        pci["pci"] = pci[year_cols].sum(axis=1, min_count=len(year_cols)) / len(
            year_cols
        )

        self.data = self.data.merge(
            pci[[self.agg_level, "pci"]], on=self.agg_level, how="left"
        )

    def merge_capacity_constraints(self):
        return

    def cross_sectional_regression_table(
        self, indep_vars=["shock5_dm"], dep_vars=["value", "quantity", "price"], tag=""
    ):
        models = []
        # Cols 1-3: Cross-sectional regression
        for var in dep_vars:
            model = utils.panel_reg(
                self.cs_data,
                f"d5_log_{var}_dm",
                indep_vars,
                time_fe=False,
                group_fe=False,
            )
            models.append(model)

        # Cols 4-6: Cross-sectional (with weights)
        for var in dep_vars:
            model = utils.panel_reg(
                self.cs_data,
                f"d5_log_{var}_dm",
                indep_vars,
                time_fe=False,
                group_fe=False,
                weight_col="product_mn_val",
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
        stargazer.rename_covariates(
            {
                "const": "Intercept",
                "shock5_dm": "Shock (5yr)",
            }
        )

        stargazer.show_residual_std_err = False
        stargazer.show_f_statistic = False
        stargazer.show_ngroups = False
        # stargazer.rename_statistic("Observations", "N")

        stargazer.add_line("Weights", ["No", "No", "No", "Yes", "Yes", "Yes"])

        latex_table = stargazer.render_latex()
        with open(
            f"{self.tables_folder}/crosssectional_{self.agg_level}_{self.growth_var}{tag}.tex",
            "w",
        ) as f:
            f.write(latex_table)

    def panel_regression_table(
        self,
        indep_vars=["shock5_win"],
        dep_vars=["value", "quantity", "price"],
        tag="",
        rename_dict={
            "const": "Intercept",
            "shock5_win": "Shock (5yr)",
        },
    ):
        print(f"Running regressions:\n dep_var ~ {'+'.join(indep_vars)}")
        models = []

        # Cols 1-3: Panel regression
        for var in dep_vars:
            model = utils.panel_reg(
                self.data,
                f"d5_log_{var}_win",
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
                f"d5_log_{var}_win",
                indep_vars,
                time_fe=True,
                group_fe=True,
                newey=True,
                newey_lags=4,
                weight_col="product_mn_val",
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
        # stargazer.rename_statistic("Observations", "N")

        stargazer.add_line("Weights", ["No", "No", "No", "Yes", "Yes", "Yes"])

        latex_table = stargazer.render_latex()
        with open(
            f"{self.tables_folder}/panel_{self.agg_level}_{self.growth_var}{tag}.tex",
            "w",
        ) as f:
            f.write(latex_table)
