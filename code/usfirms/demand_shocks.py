from utils import *


class DemandShocks:
    def __init__(self, data_folder, baci=None, parameters=None, difference_amt=5):
        self.data_folder = data_folder
        self.difference_amt = difference_amt
        self._initialize_data(baci)

        if not parameters:
            self.parameters = {
                "agg_level": ["HS6", "HS4"],  # , "HS2", "naics"
                "growth_var": ["gdp"],  # "imports"
                "use_prev_year": [True],
                "share_var": ["quantity", "value"],
            }
        else:
            self.parameters = parameters

    def _initialize_data(self, baci=None):
        print("Initializing data")

        # Store Baci data
        if type(baci) != pd.core.frame.DataFrame:
            baci = load_baci_data(self.data_folder)
        self.baci = baci
        self.baci_us_exports = baci[baci["exporter"] == 842].copy()

        self.gdp = load_gdp_data(self.data_folder)
        self.gdp = self.gdp[self.gdp.year >= 1995]

    def _create_timeseries_component(self, growth_var="gdp"):
        """
        Create variation in importer growth across industries
        """
        # print("Creating timeseries component")

        if growth_var == "imports":
            df = self.baci.groupby(["importer", "year"])["value"].sum().reset_index()
            df["log_val"] = np.log(df.value)

        elif growth_var == "us_imports":
            df = (
                self.baci_us_exports.groupby(["importer", "year"])["value"]
                .sum()
                .reset_index()
            )
            df["log_val"] = np.log(df.value)

        elif growth_var == "gdp":
            df = self.gdp
            df.rename(columns={"country_code": "importer"}, inplace=True)
            df["log_val"] = np.log(df.gdp)

        # Get log growth over difference period
        df = get_lag(
            df, ["importer", "year"], shift_col="log_val", shift_amt=self.difference_amt
        )
        df["log_growth"] = df["log_val"] - df[f"L{self.difference_amt}_log_val"]
        df.dropna(subset="log_growth", inplace=True)
        return df[["importer", "year", "log_growth"]]

    def _create_cross_sectional_component(
        self,
        use_prev_year=True,
        start_years=[1995],
        agg_level="HS6",
        share_var="quantity",
    ):
        # print("Creating cross-sectional component")

        df = self.baci

        if agg_level == "naics":
            merge_keys = load_naics_hs_crosswalk(self.data_folder)
            df = df.merge(
                merge_keys[["naics", "naics5", "naics4", "naics3", "HS6"]],
                on=["HS6"],
                how="inner",
            )

        if not use_prev_year:
            df = df[df.year.isin(start_years)]
            df["year"] = str(start_years)

        if agg_level != "HS6":
            df = (
                df.groupby(["exporter", "importer", agg_level, "year"])[share_var]
                .sum()
                .reset_index()
            )

        df["share"] = df[share_var] / df.groupby(["exporter", agg_level, "year"])[
            share_var
        ].transform("sum")

        return df[["exporter", "importer", agg_level, "share", "year"]].copy()

    def _construct_demand_shocks(self, agg_level, growth_var, use_prev_year, share_var):
        # print("Constructing demand shocks")

        ts = self._create_timeseries_component(growth_var=growth_var)
        cs = self._create_cross_sectional_component(
            agg_level=agg_level, use_prev_year=use_prev_year, share_var=share_var
        )

        if use_prev_year:
            # Match timeseries to shares from year before start year
            cs["year"] += self.difference_amt + 1
            df = ts.merge(
                cs,
                on=["importer", "year"],
                how="inner",
            )
        else:
            df = ts.merge(cs.drop(columns="year"), on=["importer"], how="inner")

        df["shock"] = df["share"] * df["log_growth"]
        df = df.groupby(["exporter", agg_level, "year"])["shock"].sum().reset_index()

        return df

    def initialize_all_demand_shocks(
        self, ordered_keys=("agg_level", "growth_var", "use_prev_year", "share_var")
    ):
        """
        Construct demand shocks using all variations of parameters and store in a dictionary
        """

        self.demand_shocks_dict = dict()

        # Generate the Cartesian product based on the ordered keys
        values_product = product(*(self.parameters[key] for key in ordered_keys))

        for combo in values_product:
            print(f"Running {combo}")
            params = dict(zip(ordered_keys, combo))
            self.demand_shocks_dict[combo] = self._construct_demand_shocks(**params)

    def get_demand_shocks(
        self,
        agg_level="HS6",
        growth_var="gdp",
        use_prev_year=True,
        share_var="quantity",
        country_code=842,
    ):
        if (agg_level, growth_var, use_prev_year, share_var) in self.demand_shocks_dict:
            ds = self.demand_shocks_dict[
                (agg_level, growth_var, use_prev_year, share_var)
            ]
            return ds[ds.exporter == country_code]
        else:
            print("Did not find series with these attributes.")
            return None
