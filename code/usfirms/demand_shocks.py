from utils import *


class DemandShocks:
    def __init__(self, data_folder, baci=None):
        self.data_folder = data_folder
        self._initialize_data(baci)

    def _initialize_data(self, baci=None):
        # Store Baci data
        if type(baci) != pd.core.frame.DataFrame:
            baci = load_baci_data(self.data_folder)
        self.baci = baci
        self.baci_us_exports = baci[baci["exporter"] == 842].copy()

        self.gdp = load_gdp_data(self.data_folder)

    def _create_timeseries_component(self, growth_var="gdp"):
        """
        Create variation in importer growth across industries
        """
        if growth_var == "imports":
            df = self.baci.groupby(["importer", "year"])["value"].sum().reset_index()
            df["log_val"] = np.log(df.value)

        elif growth_var == "gdp":
            df = self.gdp
            df.rename(columns={"country_code": "importer"}, inplace=True)
            df["log_val"] = np.log(df.gdp)

        df["log_growth"] = df["log_val"] - df.groupby("importer")["log_val"].shift(1)
        df.dropna(subset="log_growth", inplace=True)
        return df[["importer", "year", "log_growth"]]

    def _create_cross_sectional_component(
        self, use_prev_year=False, start_years=[1995], agg_level="HS6"
    ):
        df = self.baci_us_exports.copy()
        df = df.groupby(["importer", agg_level, "year"])["value"].sum().reset_index()
        df = df.sort_values(by=["importer", agg_level, "year"])

        if use_prev_year:
            df["year"] = df["year"] + 1
        else:
            df = df[df.year.isin(start_years)]
            df["year"] = str(start_years)

        df = df.groupby(["importer", agg_level, "year"])["value"].sum().reset_index()
        df["share"] = df.value / df.groupby([agg_level, "year"])["value"].transform(
            "sum"
        )

        return df[["importer", agg_level, "share", "year"]]

    def _construct_demand_shocks(self, agg_level, growth_var, use_prev_year):
        ts = self._create_timeseries_component(growth_var=growth_var)
        cs = self._create_cross_sectional_component(
            agg_level=agg_level, use_prev_year=use_prev_year
        )

        if use_prev_year:
            df = ts.merge(
                cs,
                on=["importer", "year"],
                how="inner",
            )
        else:
            df = ts.merge(cs.drop(columns="year"), on=["importer"], how="inner")

        df["shock"] = df["share"] * df["log_growth"]
        df = df.groupby([agg_level, "year"])["shock"].sum().reset_index()

        return df

    def initialize_all_demand_shocks(self):
        """
        Construct demand shocks using all variations of parameters and store in a dictionary
        """

        parameters = {
            "agg_level": ["HS6", "HS4", "HS2"],
            "growth_var": ["gdp", "imports"],
            "use_prev_year": [True, False],
        }

        ordered_keys = ("agg_level", "growth_var", "use_prev_year")

        self.demand_shocks_dict = dict()

        # Generate the Cartesian product based on the ordered keys
        values_product = product(*(parameters[key] for key in ordered_keys))

        for combo in values_product:
            print(f"Running {combo}")
            params = dict(zip(ordered_keys, combo))
            self.demand_shocks_dict[combo] = self._construct_demand_shocks(**params)

    def get_demand_shocks(self, agg_level="HS6", growth_var="gdp", use_prev_year=False):
        if (agg_level, growth_var, use_prev_year) in self.demand_shocks_dict:
            return self.demand_shocks_dict[(agg_level, growth_var, use_prev_year)]
        else:
            print("Did not find series with these attributes.")
            return None
