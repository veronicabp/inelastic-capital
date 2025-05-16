from utils import *
import utils


def load_frb_data(data_folder):
    """
    Load data from the FRB on capacity and utilization rates
    """

    dfs = []
    for var in ["capacity", "utilization"]:
        df = pd.read_csv(
            os.path.join(data_folder, "raw", "frb", f"fred_{var}.txt"),
            sep="\t",
            header=0,
        )
        df = utils.split_whitespace_column(df, "B50001: Total index", 14)

        # Clean year
        df.rename(columns={"B50001: Total index_2": "year"}, inplace=True)
        df = df[pd.to_numeric(df["year"], errors="coerce").notnull()].copy()
        df["year"] = df["year"].astype(int)

        # Clean naics code
        df.rename(columns={"B50001: Total index_1": "naics_code"}, inplace=True)
        df = df[df["naics_code"].str.match(r"^G\d+$")].copy()
        df["naics_code"] = df["naics_code"].astype(str).str.extract(r"(\d+)")

        # Clean values
        for n in range(3, 15):
            df.rename(
                columns={f"B50001: Total index_{n}": f"{var}_{n-2}"}, inplace=True
            )
        for n in range(1, 13):
            df[f"{var}_{n}"] = df[f"{var}_{n}"].astype(float)
        df[f"{var}_mn"] = df[[f"{var}_{i}" for i in range(1, 13)]].mean(axis=1)
        df.drop(columns=[f"{var}_{i}" for i in range(1, 13)], inplace=True)

        # Demean
        df[f"{var}_dmn"] = df.groupby("naics_code")[f"{var}_mn"].transform(
            lambda x: x - x.mean()
        )

        dfs.append(df)

    capacity_utilization = pd.merge(dfs[0], dfs[1], on=["year", "naics_code"])
    capacity_utilization = capacity_utilization[
        (capacity_utilization["naics_code"].astype(str).str.len() == 3)
        & (capacity_utilization["naics_code"].astype(str).str.startswith("3"))
        & (capacity_utilization["naics_code"].notnull())
    ].copy()
    return capacity_utilization


def load_frb_industrial_production(data_folder):
    indpro_path = os.path.join(data_folder, "raw", "frb", "INDPRO.csv")
    df = pd.read_csv(indpro_path, sep=",", header=0)
    # save first four digits of observtion date as year
    df["year"] = df["observation_date"].str[:4].astype(int)
    # collapse (mean) by year
    df = (
        df.groupby("year", as_index=False)["INDPRO"]
        .mean()
        .rename(columns={"INDPRO": "mean_indpro"})
    )
    return df


def load_schott_exports(data_folder):
    """
    Load exports data from Schott (2008)
    """
    iso_df = utils.load_iso_codes(data_folder)  # new utils function
    print(iso_df.head())

    # get sic/naics concordance and weights
    sic_path = os.path.join(
        data_folder, "raw", "original", "schott_2008", "conc_sic87_naics97.xlsx"
    )
    sic_naics = pd.read_excel(sic_path, sheet_name="Data")
    sic_naics = sic_naics[["sic87", "naics97", "ship8797"]]
    sic_naics["sic87"] = sic_naics["sic87"].astype(str)

    # pre-1988 exports data from .dta ----------
    path = os.path.join(
        data_folder, "raw", "original", "schott_2008", "xm_sic87_72_105_20120424.dta"
    )
    df = pd.read_stata(path, convert_categoricals=False)

    # keep wbcode year sic x
    df = df[["wbcode", "year", "sic", "x"]]
    # rename x val_exports, tostring sic and keep pre-1988
    df.rename(columns={"x": "val_exports"}, inplace=True)
    df = df[df["val_exports"] != 0]
    df["sic"] = df["sic"].astype(int).astype(str)
    df = df[df["sic"] != "nan"]
    df = df[df["year"] <= 1988]

    # merge df many:many sic87 using sic_naics
    df = pd.merge(df, sic_naics, left_on="sic", right_on="sic87", how="left")
    df["val_exports"] = (
        df["val_exports"] * df["ship8797"]
    )  # weight by weights following Boehm 2022
    df["naics"] = df["naics97"].astype(str)
    df["naics3"] = df["naics"].str[:3]
    pre_1988_exports = df.groupby(["year", "naics3", "wbcode"], as_index=False)[
        "val_exports"
    ].sum()
    # convert from millions to USD
    pre_1988_exports["val_exports"] = pre_1988_exports["val_exports"] * 1e6

    # post-1988 exports data from .dta ----------
    # for each year 1989 to 2024, read in the file exp_detl_yearly_`year'_12n.dta
    # collapse by naics3
    # and append to df
    df = pd.DataFrame()

    for year in range(89, 123):
        path = os.path.join(
            data_folder,
            "raw",
            "original",
            "schott_2008",
            "annual_legacy",
            f"exp_detl_yearly_{year}n",
            f"exp_detl_yearly_{year}n.dta",
        )
        df_temp = pd.read_stata(path, convert_categoricals=False)
        df_temp = df_temp[["year", "all_val_yr", "naics", "cty_code"]]
        df_temp.rename(columns={"all_val_yr": "val_exports"}, inplace=True)
        df_temp["naics"] = df_temp["naics"].astype(str)
        df_temp["naics3"] = df_temp["naics"].str[:3]
        df_temp = df_temp.groupby(["year", "naics3", "cty_code"], as_index=False)[
            "val_exports"
        ].sum()
        # append to df
        df = pd.concat([df, df_temp], ignore_index=True)

    # rename cty_code isonumber
    df.rename(columns={"cty_code": "isonumber"}, inplace=True)
    # merge m:1 isonumber using temp_files\isocodes.dta
    df["isonumber"] = df["isonumber"].astype(int).astype(str)
    df["isonumber"] = df["isonumber"].str.strip()
    df = pd.merge(df, iso_df, on="isonumber", how="left")
    # keep year naics3 val_exports wbcode
    df = df[["year", "naics3", "val_exports", "wbcode"]]

    post_1988_exports = df.copy()

    # append pre_1988_exports to post_1988_exports
    exports = pd.concat([pre_1988_exports, post_1988_exports], ignore_index=True)

    # rename wbcode country_abb, rename naics3 naics, rename val_exports exports
    exports.rename(
        columns={"wbcode": "country_abb", "naics3": "naics", "val_exports": "exports"},
        inplace=True,
    )
    # drop if naics == "", drop if naics == ".", drop if country_abb == "", drop if exports == ., drop if year == .
    exports = exports[exports["naics"] != ""]
    exports = exports[exports["naics"] != "."]
    exports = exports[exports["country_abb"] != ""]
    exports = exports[exports["country_abb"] != "."]
    exports = exports[exports["exports"] != "."]
    exports = exports[exports["exports"] != ""]
    exports = exports[exports["year"] != "."]
    exports = exports[exports["year"] != ""]

    # by naics year: egen tot_exp = total(exports)ex
    exports["tot_exp"] = exports.groupby(["naics", "year"])["exports"].transform("sum")
    exports["exp_share"] = exports["exports"] / exports["tot_exp"]
    return exports


def load_bea_io_data(data_folder):
    """
    Load data from the BEA on input-output tables
    """
    hist_xls = os.path.join(data_folder, "raw", "bea", "use_tables_hist.xlsx")
    post_xls = os.path.join(
        data_folder,
        "raw",
        "bea",
        "IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx",
    )

    xls_map = {
        "hist": pd.ExcelFile(hist_xls, engine="openpyxl"),
        "post": pd.ExcelFile(post_xls, engine="openpyxl"),
    }

    ERA_CONFIG = {
        "hist": {
            "years": range(1968, 1997),
            "io": dict(usecols="C:BO", skiprows=7, nrows=65),
            "names": dict(usecols="C:BO", skiprows=5, nrows=1),
            "fd": dict(usecols="CG", skiprows=7, nrows=65),
            "va": dict(usecols="C:BO", skiprows=75, nrows=1),
            "exp": dict(usecols="BV", skiprows=7, nrows=65),
            "gfg": dict(usecols="A:CH", skiprows=6, nrows=70),
        },
        "post": {
            "years": range(1997, 2024),
            "io": dict(usecols="C:BU", skiprows=7, nrows=71),
            "names": dict(usecols="C:BU", skiprows=5, nrows=1),
            "fd": dict(usecols="CQ", skiprows=7, nrows=71),
            "va": dict(usecols="C:BU", skiprows=84, nrows=1),
            "exp": dict(usecols="CC", skiprows=7, nrows=71),
            # no GFG scaling for post
        },
    }

    def read_block(xls_key, sheet, **params):
        """Helper to read, fillna, and return a numpy array (flattened if 1‐row)."""
        df = pd.read_excel(
            xls_map[xls_key],
            sheet_name=str(sheet),
            header=None,
            engine="openpyxl",
            na_values="...",
            **params,
        )
        if params.get("nrows") == 1 or ":" not in params["usecols"]:
            return df.to_numpy().flatten()
        else:
            return df.fillna(0).to_numpy()

    def compute_demand_shares(IO, FD):
        total = IO.sum(axis=1) + FD
        C_p = IO / total[None, :]
        UDS = np.linalg.inv(np.eye(len(total)) - C_p) * np.outer(1 / total, FD)

        # avoid divide by zero invalid value warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            DDS = IO / (total - FD)[:, None]
        return DDS, UDS

    def compute_cost_shares(IO, VA, GFGSCL=None, gfg_source_index=None):
        total = IO.sum(axis=0) + VA
        II = IO.shape[0]
        C_p = IO / total[:, None]
        UCS = np.linalg.inv(np.eye(II) - C_p.T) * VA[None, :] / total[:, None]
        DCS = (IO / (total - VA)).T
        # if historical, apply GFG scaling to IOT where source is GFG
        IOT = IO.T.copy()
        if GFGSCL is not None:
            IOT[gfg_source_index, :] *= GFGSCL
        total_mat = np.outer(total, np.ones(II))
        return DCS, UCS, IOT, total_mat

    demand_side_shares = []
    cost_side_shares = []
    for era, cfg in ERA_CONFIG.items():
        for year in tqdm(cfg["years"]):
            # 1) read all blocks
            names = read_block(era, year, **cfg["names"])
            IO = read_block(era, year, **cfg["io"])
            FD = read_block(era, year, **cfg["fd"])
            VA = read_block(era, year, **cfg["va"])
            EXP = read_block(era, year, **cfg["exp"])

            # 2) if historical, compute GFG scaling
            GFGSCL = None
            if era == "hist":
                gfg_df = pd.read_excel(
                    xls_map["hist"],
                    sheet_name=str(year),
                    header=0,
                    engine="openpyxl",
                    na_values="...",
                    **cfg["gfg"],
                ).fillna(0)
                mask = gfg_df["IOCode"] == "GFG"
                GFGSCL = (
                    gfg_df.loc[mask, "National defense: Consumption expenditures"].iat[
                        0
                    ]
                    / gfg_df.loc[mask, "Total Commodity Output"].iat[0]
                )

            # 3) demand‐side shares & write
            DDS, UDS = compute_demand_shares(IO, FD)
            df_d = pd.DataFrame(DDS, columns=names, index=names).stack().reset_index()
            df_u = pd.DataFrame(UDS, columns=names, index=names).stack().reset_index()
            df_d.columns = ["naics_source", "naics_dest", "DDS"]
            df_u.columns = ["naics_source", "naics_dest", "UDS"]
            out = pd.merge(df_d, df_u, on=["naics_source", "naics_dest"]).assign(
                year=year
            )
            demand_side_shares.append(out)

            # 4) cost‐side shares & write
            DCS, UCS, IOT, TOTAL = compute_cost_shares(
                IO,
                VA,
                GFGSCL,
                gfg_source_index=(
                    list(names).index("GFG") if GFGSCL is not None else None
                ),
            )
            # build DataFrames
            dcs_df = pd.DataFrame(DCS, columns=names, index=names).stack().reset_index()
            ucs_df = pd.DataFrame(UCS, columns=names, index=names).stack().reset_index()
            iot_df = pd.DataFrame(IOT, columns=names, index=names).stack().reset_index()
            tot_df = (
                pd.DataFrame(TOTAL, columns=names, index=names).stack().reset_index()
            )
            exp_df = pd.DataFrame({"naics_source": names, "exp": EXP})

            dcs_df.columns = ["naics_source", "naics_dest", "DCS"]
            ucs_df.columns = ["naics_source", "naics_dest", "UCS"]
            # note: IOT rows come as (dest, source, IOT), swap names
            iot_df.columns = ["naics_dest", "naics_source", "IOT"]
            tot_df.columns = ["naics_dest", "naics_source", "total"]

            cmap = (
                dcs_df.merge(ucs_df, on=["naics_source", "naics_dest"])
                .merge(iot_df, on=["naics_source", "naics_dest"])
                .merge(tot_df, on=["naics_source", "naics_dest"])
                .merge(exp_df, on="naics_source")
            )
            cmap["year"] = year

            cost_side_shares.append(cmap)

    demand_side = pd.concat(demand_side_shares, ignore_index=True)
    cost_side = pd.concat(cost_side_shares, ignore_index=True)
    return demand_side, cost_side


class NaicsDemandShocks:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    # Private methods
    def _construct_gdp_shocks(self):
        """
        Constructs trade shocks for naics level using gdp data
        """
        # TODO

    def _construct_exchange_rate_shocks(self):
        """
        Constructs trade shocks for naics level using exchange rate data
        """
        # TODO

    def _construct_shea_shocks(self):
        """
        Constructs shocks for naics level using data on intermediate inputs
        """
        # TODO

    def _compile_naics_data(self):
        """
        Compile naics data on prices, quantities, investment
        """
        # TODO

    # Public methods
    def initialize_data(self):
        """
        Construct full data
        """
        gdp_shocks = self._construct_gdp_shocks()
        exchange_rate_shocks = self._construct_exchange_rate_shocks()
        shea_shocks = self._construct_shea_shocks()
        naics_data = self._compile_naics_data()
        # TODO: merge all and construct df
        # self.data = df

    def run_regressions(self):
        """
        Run regressions on naics data
        """
