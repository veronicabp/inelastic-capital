from utils import *


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


def construct_bea_shares(data_folder):
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
            "years": range(1970, 1997),
            "io": dict(usecols="C:BO", skiprows=7, nrows=65),
            "names": dict(usecols="C:BO", skiprows=5, nrows=1),
            "fd": dict(usecols="CG", skiprows=7, nrows=65),
            "va": dict(usecols="C:BO", skiprows=75, nrows=1),
            "exp": dict(usecols="BV", skiprows=7, nrows=65),
            "gfg": dict(usecols="A:CH", skiprows=6, nrows=70),
            "full_names": dict(usecols="C:CH", skiprows=5, nrows=1),
            "full": dict(usecols="C:CH", skiprows=7, nrows=67),
        },
        "post": {
            "years": range(1997, 2024),
            "io": dict(usecols="C:BU", skiprows=7, nrows=71),
            "names": dict(usecols="C:BU", skiprows=5, nrows=1),
            "fd": dict(usecols="CQ", skiprows=7, nrows=71),
            "va": dict(usecols="C:BU", skiprows=84, nrows=1),
            "exp": dict(usecols="CC", skiprows=7, nrows=71),
            "full_names": dict(usecols="C:CR", skiprows=5, nrows=1),
            "full": dict(usecols="C:CR", skiprows=7, nrows=73),
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

    def compute_demand_shares(IO, FD, names):
        total = IO.sum(axis=1) + FD
        C_p = IO / total[None, :]
        UDS = np.linalg.inv(np.eye(len(total)) - C_p) * np.outer(1 / total, FD)

        # avoid divide by zero invalid value warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            DDS = IO / (total - FD)[:, None]

        df_d = pd.DataFrame(DDS, columns=names, index=names).stack().reset_index()
        df_u = pd.DataFrame(UDS, columns=names, index=names).stack().reset_index()
        df_d.columns = ["naics_source", "naics_dest", "DDS"]
        df_u.columns = ["naics_source", "naics_dest", "UDS"]
        out = pd.merge(df_d, df_u, on=["naics_source", "naics_dest"], how="outer")
        out.fillna(0, inplace=True)

        return out

    def compute_cost_shares(IO, VA, names):
        total = IO.sum(axis=0) + VA
        II = IO.shape[0]
        C_p = IO / total[:, None]
        UCS = np.linalg.inv(np.eye(II) - C_p.T) * VA[None, :] / total[:, None]
        DCS = (IO / (total - VA)).T

        dcs_df = pd.DataFrame(DCS, columns=names, index=names).stack().reset_index()
        ucs_df = pd.DataFrame(UCS, columns=names, index=names).stack().reset_index()

        dcs_df.columns = ["naics_dest", "naics_source", "DCS"]
        ucs_df.columns = ["naics_dest", "naics_source", "UCS"]

        cmap = dcs_df.merge(ucs_df, on=["naics_source", "naics_dest"], how="outer")
        cmap.fillna(0, inplace=True)

        return cmap

    def compute_sale_shares(FULL, names, full_names):
        # if historical, apply GFG scaling to IOT where source is GFG
        full_names = list(full_names)
        full_names[-2] = "total_final"
        full_names[-1] = "total_output"
        df = pd.DataFrame(
            FULL, columns=full_names, index=list(names) + ["Used", "Other"]
        )

        df_tot = df[["total_output"]].reset_index(names="naics_source")
        df = df.drop(columns=["total_output"]).stack().reset_index()
        df.columns = ["naics_source", "naics_dest", "shipments"]
        df = df.merge(df_tot, on="naics_source", how="left")

        # Drop total rows
        df = df[
            (~df.naics_dest.isin(["T001", "total_final", "nan"]))
            & (~df.naics_dest.isna())
        ].copy()

        # Aggregate by naics
        df = map_naics_codes(df, vars=["naics_source"])
        df = df.groupby(["naics_source", "naics_dest"], as_index=False).sum()

        df["sales_sh"] = df["shipments"] / df["total_output"]
        return df

    demand_side_shares = []
    cost_side_shares = []
    sales_shares = []
    for era, cfg in ERA_CONFIG.items():
        for year in tqdm(cfg["years"]):
            # read all blocks
            names = read_block(era, year, **cfg["names"]).astype(str)
            IO = read_block(era, year, **cfg["io"])
            FD = read_block(era, year, **cfg["fd"])
            VA = read_block(era, year, **cfg["va"])
            EXP = read_block(era, year, **cfg["exp"])
            FULL = read_block(era, year, **cfg["full"])
            full_names = read_block(era, year, **cfg["full_names"]).astype(str)

            # demand‐side shares & write
            out = compute_demand_shares(IO, FD, names)
            out["year"] = year
            demand_side_shares.append(out)

            # cost‐side shares & write
            out = compute_cost_shares(IO, VA, names)
            out["year"] = year
            cost_side_shares.append(out)

            ss = compute_sale_shares(FULL, names, full_names)
            ss["year"] = year
            sales_shares.append(ss)

    demand_side = pd.concat(demand_side_shares, ignore_index=True)
    cost_side = pd.concat(cost_side_shares, ignore_index=True)
    sales_shares = pd.concat(sales_shares, ignore_index=True)
    return demand_side, cost_side, sales_shares


def process_bea_accounts(
    file_path: str,
    sheet_name: str,
    skiprows: int,
    usecols: str,
    nrows: int,
    IOcodes: pd.DataFrame,
    value_name: str = "qty_index",
) -> pd.DataFrame:
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        skiprows=skiprows,
        header=0,
        usecols=usecols,
        nrows=nrows,
        engine="openpyxl",
    )
    # drop any unwanted “Line” or “Unnamed: 2” cols if present
    df = df.drop(
        columns=[c for c in ("Line", "Unnamed: 2") if c in df], errors="ignore"
    )
    df = df.rename(columns={"Unnamed: 1": "Description"})
    df = (
        df.assign(Description=lambda d: d.Description.str.strip())
        .pipe(clean_GDP_by_ind)
        .merge(IOcodes, on="Description", how="inner")
        .drop(columns="Description")
        .assign(naics_code=lambda d: d.IO_Code)
        .pipe(lambda d: map_naics_codes(d, vars=["naics_code"]))
        .replace("...", 0)
        .melt(id_vars=["naics_code", "IO_Code"], var_name="year", value_name=value_name)
    )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notnull()].copy()

    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df[df[value_name].notnull()].copy()

    # Divide by 1997 level
    val_1997 = (
        df[df["year"] == 1997]
        .rename(columns={value_name: f"{value_name}_1997"})
        .drop(columns=["year"])
        .copy()
    )
    df = df.merge(val_1997, on=["naics_code", "IO_Code"], how="left")
    df[value_name] = 100 * df[value_name] / df[f"{value_name}_1997"]
    df.drop(columns=[f"{value_name}_1997"], inplace=True)

    return df


def load_io_codes(file_path: str, nrows: int) -> pd.DataFrame:
    return pd.read_excel(
        file_path,
        sheet_name="Sheet1",
        header=5,
        usecols="A:B",
        nrows=nrows,
        engine="openpyxl",
    ).assign(Description=lambda df: df.Description.str.strip())


def load_bea_industry_accounts(data_folder):
    """
    Load data from the BEA on industry accounts
    """

    # f"{data_folder}/raw/bea/{filename}"

    IOcodes = load_io_codes(
        os.path.join(data_folder, "raw", "bea", "IO_codes.xlsx"), nrows=71
    )
    IOcodes_hist = load_io_codes(
        os.path.join(data_folder, "raw", "bea", "IO_codes_aggregated.xlsx"), nrows=65
    )

    quantities = process_bea_accounts(
        os.path.join(data_folder, "raw", "bea", "II_qty_97_24.xlsx"),
        "Sheet1",
        skiprows=7,
        usecols="A:AE",
        nrows=99,
        IOcodes=IOcodes,
        value_name="qty_index",
    )
    prices = process_bea_accounts(
        os.path.join(data_folder, "raw", "bea", "II_price_97_23.xlsx"),
        "Sheet1",
        skiprows=7,
        usecols="A:AD",
        nrows=99,
        IOcodes=IOcodes,
        value_name="price_index",
    )
    quantities_hist = process_bea_accounts(
        os.path.join(data_folder, "raw", "bea", "GDPbyInd_II_1947-1997.xlsx"),
        "ChainQtyIndexes",
        skiprows=5,
        usecols="A:BA",
        nrows=102,
        IOcodes=IOcodes_hist,
        value_name="qty_index",
    )
    prices_hist = process_bea_accounts(
        os.path.join(data_folder, "raw", "bea", "GDPbyInd_II_1947-1997.xlsx"),
        "ChainPriceIndexes",
        skiprows=5,
        usecols="A:BA",
        nrows=102,
        IOcodes=IOcodes_hist,
        value_name="price_index",
    )

    df = prices.merge(quantities, on=["naics_code", "IO_Code", "year"], how="left")
    df_hist = prices_hist.merge(
        quantities_hist, on=["naics_code", "IO_Code", "year"], how="left"
    )

    return df, df_hist


def map_naics_codes(df, vars=[]):
    mappings = [
        ("11", ["111CA", "113FF"]),
        ("336", ["3361MV", "3364OT"]),
        ("311,2", ["311FT"]),
        ("313,4", ["313TT"]),
        ("315,6", ["315AL"]),
        ("44", ["441", "445", "452", "4A0"]),
        ("48", ["481", "482", "483", "484", "485", "486", "487OS", "493"]),
        ("52", ["521CI", "523", "524", "525"]),
        ("53", ["HS", "ORE", "532RL"]),
        ("54", ["5411", "5415", "5412OP"]),
        ("71", ["711AS", "713"]),
    ]
    for var in vars:
        for new_code, pats in mappings:
            pattern = "|".join(pats)
            mask = df[var].str.contains(pattern, na=False)
            df.loc[mask, var] = new_code

        # Split naics
        naics_map = {
            "311,2": ["311", "312"],
            "313,4": ["313", "314"],
            "315,6": ["315", "316"],
        }

        df[var] = df[var].map(lambda x: naics_map[x] if x in naics_map else [x])

        df = df.explode(var).reset_index(drop=True)

    return df


def construct_shea_indicators(data_folder, shea_threshold=3):
    """
    Load data from the SHEA on intermediate inputs
    """
    demand_shares, cost_shares, sales_shares = construct_bea_shares(data_folder)

    # Merge both source-dest and dest-source
    shea_full = cost_shares.merge(
        demand_shares, on=["naics_source", "naics_dest", "year"], how="left"
    )
    rev = shea_full[["naics_source", "naics_dest", "year", "DCS", "UCS"]].rename(
        columns={
            "naics_source": "naics_dest",
            "naics_dest": "naics_source",
            "DCS": "DCS_ji",
            "UCS": "UCS_ji",
        }
    )
    shea_full = shea_full.merge(
        rev, on=["naics_source", "naics_dest", "year"], how="left"
    )

    shea_full["naics_source"] = shea_full["naics_source"].astype(str)
    shea_full["naics_dest"] = shea_full["naics_dest"].astype(str)
    shea_full = shea_full[shea_full["naics_source"].str.startswith("3")]

    # Compute stats
    valid = (shea_full[["DDS", "UDS", "DCS", "UCS", "DCS_ji", "UCS_ji"]] >= 0).all(
        axis=1
    )

    stat1 = shea_full[["DDS", "UDS"]].min(axis=1) / shea_full[["DCS", "UCS"]].max(
        axis=1
    )
    stat2 = shea_full[["DDS", "UDS"]].min(axis=1) / shea_full[
        ["DCS", "UCS", "DCS_ji", "UCS_ji"]
    ].max(axis=1)

    shea_full["stat1"] = np.where((valid) & (np.abs(stat1) != np.inf), stat1, 0)
    shea_full["stat2"] = np.where((valid) & (np.abs(stat2) != np.inf), stat2, 0)

    shea_full["era"] = (shea_full.year >= 1997).astype(int)

    # Compute min by industry
    for i in [1, 2]:
        shea_full[f"Lshea{i}"] = (shea_full[f"stat{i}"] > shea_threshold).astype(int)
        # NOTE: Difference from Boem -- they groupby era as well, but this seems arbitrary
        # shea_full[f"Lshea{i}_min"] = shea_full.groupby(["naics_source", "naics_dest"])[
        #     f"Lshea{i}"
        # ].transform("min")
        shea_full[f"Lshea{i}_min"] = shea_full.groupby(
            ["naics_source", "naics_dest", "era"]
        )[f"Lshea{i}"].transform("min")

    shea_full["year"] = shea_full["year"] + 1

    # Collapse by industry
    shea_full = map_naics_codes(shea_full, vars=["naics_source"])

    shea_full = shea_full.groupby(
        ["naics_source", "naics_dest", "year"], as_index=False
    ).agg(
        Lshea1=("Lshea1", "min"),
        Lshea2=("Lshea2", "min"),
        Lshea1_min=("Lshea1_min", "min"),
        Lshea2_min=("Lshea2_min", "min"),
    )

    # Split naics
    naics_map = {
        "311,2": ["311", "312"],
        "313,4": ["313", "314"],
        "315,6": ["315", "316"],
    }
    shea_full = (
        shea_full.assign(
            naics_source=lambda d: d["naics_source"].apply(
                lambda x: naics_map[x] if x in naics_map else [x]
            )
        )
        .explode("naics_source")
        .reset_index(drop=True)
    )

    return shea_full, clean_shares(sales_shares)


def clean_shares(df, lag=1):
    df = df[df["naics_source"].str.startswith("3") | (df["naics_source"] == "GFG")]
    df = df.drop(columns=["total_output", "shipments"])

    # Create destination variable
    df["dest"] = df["naics_dest"]
    df.loc[df["naics_dest"] == "F010", "dest"] = "pce"
    df.loc[df["naics_dest"].isin(["F02S", "F02R", "F02T"]), "dest"] = "struct"
    df.loc[df["naics_dest"] == "F02E", "dest"] = "equip"
    df.loc[df["naics_dest"] == "F02N", "dest"] = "ipp"
    df.loc[df["naics_dest"] == "F040", "dest"] = "exports"
    df.loc[df["naics_dest"].str.startswith("F06", na=False), "dest"] = "defense"
    df.loc[df["naics_dest"] == "GFGD", "dest"] = "defense"

    # Generate adjustment factor for GFG
    adj = (
        df.query("naics_source == 'GFG' and naics_dest == 'F06C'")
        .loc[:, ["naics_source", "year", "sales_sh"]]
        .rename(columns={"sales_sh": "adj_factor_GFG", "naics_source": "naics_dest"})
    )

    # Merge adjustment factor
    df = df[df["naics_source"] != "GFG"]
    df = df.merge(adj, on=["naics_dest", "year"], how="left")

    # Adjust sales share for GFG
    mask = df["naics_dest"] == "GFG"
    df.loc[mask, "sales_sh"] *= df.loc[mask, "adj_factor_GFG"]
    df["dest"] = np.where(mask, "defense", df["dest"])
    df = df.drop(columns=["adj_factor_GFG"])

    # Drop if missing
    df = df[~df.dest.str.startswith("F", na=False)].copy()

    # Collapse (sum) by group
    agg = df.groupby(["naics_source", "dest", "year"], as_index=False)["sales_sh"].sum()
    agg = agg.rename(columns={"sales_sh": "Lsales_sh"})
    agg["year"] += lag

    return agg


def naics_import_shares(data_folder):
    conc = pd.read_excel(
        os.path.join(
            data_folder, "raw", "original", "schott_2008", "conc_sic87_naics97.xlsx"
        )
    )
    conc = conc[["sic87", "naics97", "ship8797"]]
    conc["sic"] = conc["sic87"].astype(int)
    conc = conc.drop(columns="sic87")

    # Trade data from 1972 to 1988
    trade = pd.read_stata(
        os.path.join(
            data_folder,
            "raw",
            "original",
            "schott_2008",
            "xm_sic87_72_105_20120424.dta",
        )
    )
    trade = trade[trade["x"] != 0][["wbcode", "year", "sic", "x"]].copy()
    trade = trade.rename(columns={"x": "val_exports"})
    trade["sic"] = trade["sic"].astype(int)
    trade = trade[trade["year"] <= 1988]
    trade = trade.merge(conc, how="left", on="sic")

    # allocate to NAICS6 via ship8797 weights
    trade["val_exports"] = trade["val_exports"] * trade["ship8797"]
    trade["naics"] = trade["naics97"].astype(str)
    trade["naics3"] = trade["naics"].str[:3]
    exp7288 = trade.groupby(["year", "naics3", "wbcode"], as_index=False)[
        "val_exports"
    ].sum()
    exp7288 = exp7288.rename(columns={"wbcode": "country_code"})

    # Trade data from 1989 to 2024
    frames = []
    for i in tqdm(range(1989, 2025)):
        path = os.path.join(
            data_folder,
            "raw",
            "original",
            "schott_2008",
            "annual",
            f"exp_detl_{i}_12n.dta",
        )
        df = pd.read_stata(path)
        df = df[["year", "all_val_yr", "naics", "cty_code"]]
        df["naics3"] = df["naics"].astype(str).str[:3]
        df = df.rename(columns={"all_val_yr": "val_exports", "cty_code": "isonumber"})
        df = df.groupby(["naics3", "year", "isonumber"], as_index=False)[
            "val_exports"
        ].sum()
        frames.append(df[["isonumber", "year", "naics3", "val_exports"]])

    exp8924 = pd.concat(frames, ignore_index=True)
    exp8924["isonumber"] = exp8924["isonumber"].astype(int)
    exp8924 = exp8924[exp8924["naics3"] != ""].copy()
    exp8924 = exp8924[exp8924["naics3"] != "."].copy()

    iso_codes = pd.read_csv(os.path.join(data_folder, "raw", "iso", "iso_vbp.csv"))
    exp8924 = exp8924.merge(
        iso_codes[["iso3", "isonumber"]],
        how="inner",
        on="isonumber",
    )

    exp8924 = exp8924.rename(columns={"iso3": "country_code"})
    exp8924 = exp8924[["year", "naics3", "country_code", "val_exports"]].copy()

    # convert 1972–88 data from millions to dollars
    exp7288["val_exports"] *= 1_000_000

    combined = pd.concat([exp7288, exp8924], ignore_index=True)
    combined = combined.rename(
        columns={
            "val_exports": "exports",
            "country_code": "importer",
        }
    )
    # drop invalid rows
    combined = combined[combined["naics3"] != ""].copy()
    combined = combined[combined["naics3"] != "."].copy()
    combined = combined.dropna(subset=["naics3", "importer", "exports", "year"])
    combined = combined.sort_values(["naics3", "importer", "year"])

    # compute total exports by NAICS3‐year
    tot = (
        combined.groupby(["naics3", "year"], as_index=False)["exports"]
        .sum()
        .rename(columns={"exports": "tot_exp"})
    )
    combined = combined.merge(tot, on=["naics3", "year"], how="left")
    combined["exp_share"] = combined["exports"] / combined["tot_exp"]

    return combined[["naics3", "year", "importer", "exp_share"]].copy()


def keep_oecd(df, data_folder):
    boem_sample = pd.read_stata(
        os.path.join(
            data_folder,
            "raw",
            "original",
            "boem_pandalai-nayar_2022/empirics/main analysis/temp_files/country_names.dta",
        )
    ).rename(columns={"country_abb": "iso"})
    df = df.merge(boem_sample, on="iso", how="inner")

    oecd_sample = pd.read_stata(
        os.path.join(
            data_folder,
            "raw",
            "original",
            "boem_pandalai-nayar_2022/empirics/main analysis/temp_files/oecd_join.dta",
        )
    )
    df = df.merge(oecd_sample, on="country", how="inner")
    df = df[df.oecd_pre2000 == 1].copy()
    df = df.drop(columns=["oecd_pre2000", "oecd_post2000", "country"])
    return df


def load_un_gdp(data_folder):
    path = os.path.join(data_folder, "raw", "unstats", "un_rgdp_natcurr.csv")
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Country/Area": "country",
            "Year": "year",
            "GDP, at constant 2015 prices - National currency": "rgdp",
        }
    )
    df["rgdp"] = pd.to_numeric(df["rgdp"], errors="coerce")
    df = df.dropna(subset=["country", "year", "rgdp"])
    df["iso"] = df.country.apply(lambda x: un_iso_map[x])

    df = df.groupby(["iso", "year"], as_index=False)["rgdp"].sum()

    # Keep countries in OECD
    df = keep_oecd(df, data_folder)

    return df


def load_un_er(data_folder):
    df = pd.read_csv(
        os.path.join(data_folder, "raw", "unstats", "un_nomexchangerates_usd.csv")
    )
    df = df.rename(
        columns={
            "Country/Area": "country",
            "Year": "year",
            "IMF based exchange rate": "er",
        }
    )
    df["er"] = pd.to_numeric(df["er"], errors="coerce")
    df["er"] = 1 / df["er"]
    df["iso"] = df.country.apply(lambda x: un_iso_map[x] if x in un_iso_map else np.nan)
    df = df.dropna(subset=["iso", "year", "er"])
    df = df.groupby(["iso", "year"], as_index=False)["er"].mean()

    # Keep countries in OECD
    df = keep_oecd(df, data_folder)

    return df


class NaicsDemandShocks:
    def __init__(self, data_folder, lag=1, shea_threshold=3):
        self.data_folder = data_folder
        self.lag = lag

        self.demand_shares, self.cost_shares, self.sales_shares = construct_bea_shares(
            data_folder
        )
        self.shea_indicators, self.sales_shares = construct_shea_indicators(
            self.data_folder, shea_threshold
        )

        self.instruments = [
            "Dln_frgn_rgdp",
            "Dln_er",
            # "Dln_M_shea_inst1",
            "Dln_M_shea_inst2",
            # "Dln_M_shea_inst1_min",
            # "Dln_M_shea_inst2_min",
        ]
        self.dependent_vars = ["Dln_Pip"]
        self.endog_vars = ["Dln_ip"]
        self.exog_vars = ["Dln_capacity", "Dln_UVCip"]

    # Private methods
    def _construct_trade_shocks(self):
        """
        Constructs trade shocks for naics level using gdp and er data
        """

        ############ Load GDP data ############
        gdp = load_un_gdp(self.data_folder)
        gdp = gdp.rename(columns={"iso": "importer"})

        # Construct growth rates
        gdp.sort_values(["importer", "year"], inplace=True)
        gdp = get_lag(
            gdp, group_cols=["importer"], shift_col="rgdp", shift_amt=self.lag
        )
        gdp["d_log_rgdp"] = np.log(gdp["rgdp"]) - np.log(gdp[f"L{self.lag}_rgdp"])
        # gdp["d_log_rgdp"] = (gdp["rgdp"] - gdp[f"L{self.lag}_rgdp"]) / gdp[
        #     f"L{self.lag}_rgdp"
        # ]

        ############ Load exchange rate data ############
        er = load_un_er(self.data_folder)
        er = er.rename(columns={"iso": "importer"})
        er.sort_values(["importer", "year"], inplace=True)
        er = get_lag(er, group_cols=["importer"], shift_col="er", shift_amt=self.lag)
        er["d_log_er"] = np.log(er["er"]) - np.log(er[f"L{self.lag}_er"])
        # er["d_log_er"] = (er["er"] - er[f"L{self.lag}_er"]) / er[f"L{self.lag}_er"]

        ############ Load export shares ############
        shares = naics_import_shares(self.data_folder)
        shares = shares[shares.naics3.str.startswith("3")].copy()
        shares["year"] += self.lag
        shares.rename(columns={"exp_share": "Lexp_share"}, inplace=True)

        # Merge in share of sales going to exports and scale by this
        exp_shares = self.sales_shares[self.sales_shares["dest"] == "exports"].copy()
        exp_shares = exp_shares.rename(columns={"naics_source": "naics3"})
        shares = shares.merge(exp_shares, on=["naics3", "year"], how="inner")
        shares["Lexp_share"] = shares["Lexp_share"] * shares["Lsales_sh"]

        ############ Merge ############
        df = gdp.merge(er, on=["importer", "year"], how="outer")
        df = df.merge(shares, on=["importer", "year"])
        df["Dln_frgn_rgdp"] = df["Lexp_share"] * df["d_log_rgdp"]
        df["Dln_er"] = df["Lexp_share"] * df["d_log_er"]
        df = (
            df.groupby(["naics3", "year"], as_index=False)[
                ["Dln_frgn_rgdp", "Dln_er", "Lexp_share"]
            ]
            .sum()
            .reset_index()
        )
        return df[["naics3", "year", "Dln_frgn_rgdp", "Dln_er", "Lexp_share"]]

    def _construct_shea_shocks(self, shea_threshold=3):
        """
        Constructs shocks for naics level using data on intermediate inputs
        """
        # Load data
        df_curr, df_hist = load_bea_industry_accounts(self.data_folder)
        df = pd.concat([df_curr, df_hist], ignore_index=True)

        # Clean data
        df = df.rename(columns={"IO_Code": "dest"})
        df = df[~df["dest"].str.startswith("G")].copy()
        df["ind_gr"] = df["dest"].astype("category").cat.codes
        df = df.sort_values(["ind_gr", "year"])
        df["Dln_M"] = df.groupby("ind_gr")["qty_index"].pct_change()
        df = df.drop(columns=["price_index", "qty_index"]).copy()

        # Load SHEA data
        shea_indicators, sales_shares = self.shea_indicators, self.sales_shares

        df = df.merge(sales_shares, on=["dest", "year"], how="inner")
        df = df.merge(
            shea_indicators,
            left_on=["naics_source", "dest", "year"],
            right_on=["naics_source", "naics_dest", "year"],
            how="inner",
        )

        for indicator_col in ["Lshea1", "Lshea2", "Lshea1_min", "Lshea2_min"]:
            new_col = f"Dln_M_{indicator_col.replace('Lshea', 'shea_inst')}"
            df[new_col] = df["Dln_M"] * df[indicator_col] * df["Lsales_sh"]

        df = df.groupby(["naics_source", "year"], as_index=False).agg(
            Dln_M_shea_inst1=("Dln_M_shea_inst1", "sum"),
            Dln_M_shea_inst2=("Dln_M_shea_inst2", "sum"),
            Dln_M_shea_inst1_min=("Dln_M_shea_inst1_min", "sum"),
            Dln_M_shea_inst2_min=("Dln_M_shea_inst2_min", "sum"),
        )

        df = df.rename(columns={"naics_source": "naics3"})

        return df

    def _compile_naics_data(self):
        """
        Compile naics data on prices, quantities, investment
        """

        # NBER CES Manufacturing data
        ces_file = os.path.join(
            self.data_folder,
            "raw",
            "nber-manufacturing-data",
            "nberces5818v1_n1997.csv",
        )
        ces = pd.read_csv(ces_file)

        ces = ces.sort_values(["naics", "year"])
        ces["vprod"] = (
            ces["vship"] + ces["invent"] - ces.groupby("naics")["invent"].shift(1)
        )
        ces["L_vprod"] = ces.groupby("naics")["vprod"].shift(1)

        ces["Dln_piship"] = np.log(ces["piship"]) - np.log(
            ces.groupby("naics")["piship"].shift(1)
        )

        # Collapse by naics3 and year
        ces["naics3"] = ces["naics"].astype(str).str[:3]

        # Take industry-size weighted average of piship
        ces["s_vprod"] = ces["vprod"] / ces.groupby(["naics3", "year"])[
            "vprod"
        ].transform("sum")
        ces["s_lvprod"] = ces["L_vprod"] / ces.groupby(["naics3", "year"])[
            "L_vprod"
        ].transform("sum")

        for col in ["Dln_piship", "piship", "piinv", "pimat"]:
            ces[col] = ces[col] * (ces["s_vprod"] + ces["s_lvprod"]) / 2

        ces["VC"] = ces[["prodw", "matcost", "energy"]].sum(axis=1)

        ces = ces.groupby(["naics3", "year"], as_index=False).agg(
            Dln_piship=("Dln_piship", "sum"),
            vprod=("vprod", "sum"),
            vship=("vship", "sum"),
            invent=("invent", "sum"),
            VC=("VC", "sum"),
            invest=("invest", "sum"),
            cap=("cap", "sum"),
            emp=("emp", "sum"),
            pay=("pay", "sum"),
            prodw=("prodw", "sum"),
            piship=("piship", "sum"),
            piinv=("piinv", "sum"),
            pimat=("pimat", "sum"),
        )

        ces.rename(columns={"cap": "capital"}, inplace=True)

        ces.loc[ces["vprod"] == 0, "vprod"] = np.nan

        # NOTE: Boehm use pct difference instead of log -- does this matter?
        ces["Dln_vprod"] = np.log(ces["vprod"]) - np.log(
            ces.groupby("naics3")["vprod"].shift(1)
        )

        # FRB Utilization data
        # TODO: Data cleaning. Currently taking Boehm cleaned version
        frb_file = os.path.join(
            self.data_folder,
            "raw",
            "original",
            "boem_pandalai-nayar_2022",
            "empirics",
            "main analysis",
            "temp_files",
            "industry_utilization_data_final.dta",
        )
        frb = pd.read_stata(frb_file)
        frb = frb[(frb["naics"].str.startswith("3"))].copy()
        frb = frb[frb["naics"].str.len() == 3].copy()
        frb = frb.rename(columns={"naics": "naics3"})
        frb = frb.sort_values(["naics3", "year"])
        frb.rename(columns={"cap": "capacity"}, inplace=True)

        # Rescale util
        frb["util"] = frb["util"] / 100

        # Get lag of utilization rate
        frb = get_lag(frb, group_cols=["naics3"], shift_col="util", shift_amt=self.lag)

        df = ces.merge(
            frb,
            on=["naics3", "year"],
            how="inner",
        )

        for col in ["ip", "util", "capital", "capacity", "VC", "invest", "emp"]:
            df[f"ln_{col}"] = np.log(df[col])
            df.loc[df[col] <= 0, f"ln_{col}"] = np.nan  # Avoid log(0) or log(negative)
            df[f"Dln_{col}"] = df[f"ln_{col}"] - df.groupby("naics3")[
                f"ln_{col}"
            ].shift(1)

        df["Dln_Pip"] = df["Dln_vprod"] - df["Dln_ip"]
        df["Dln_UVCip"] = df["Dln_VC"] - df["Dln_ip"]

        return df

    # Public methods
    def initialize_data(self):
        """
        Construct full data
        """
        trade_shocks = self._construct_trade_shocks()
        shea_shocks = self._construct_shea_shocks()
        naics_data = self._compile_naics_data()

        # Merge all
        df = naics_data.merge(trade_shocks, on=["naics3", "year"], how="left").merge(
            shea_shocks, on=["naics3", "year"], how="left"
        )

        # Keep sample where instruments are non missing
        df = df[df.year >= 1973].copy()

        # Construct FEs interacted with export shares
        for year in sorted(df["year"].unique()):
            df[f"exp_sh_{year}"] = df["Lexp_share"] * (df["year"] == year).astype(int)

        df = df.set_index(["naics3", "year"])
        self.data_nowin = df.copy()

        # Winsorize all variables at 1% and 99%
        for col in (
            self.dependent_vars + self.endog_vars + self.exog_vars + self.instruments
        ):
            ub = df[col].quantile(0.99)
            lb = df[col].quantile(0.01)
            df[col] = np.where(df[col] > ub, ub, df[col])
            df[col] = np.where(df[col] < lb, lb, df[col])

        # Demean utilization rate
        df["util_dm"] = df["L1_util"] - df.groupby(["naics3"])["L1_util"].transform(
            "mean"
        )

        # Interact utilization rate with endogenous variables
        df["util_bin"] = 1
        df.loc[df["util_dm"] > df["util_dm"].quantile(0.15), "util_bin"] = 2
        df.loc[df["util_dm"] > df["util_dm"].quantile(0.50), "util_bin"] = 3
        df.loc[df["util_dm"] > df["util_dm"].quantile(0.85), "util_bin"] = 4
        df["util_bin"] = df["util_bin"].astype("category")

        # Make dummy variables for utilization rate bins
        for level in sorted(df.util_bin.unique())[1:]:
            df[f"ub{level}"] = (df["util_bin"] == level).astype(int)

        for col in (
            self.dependent_vars + self.endog_vars + self.exog_vars + self.instruments
        ):
            for level in df.util_bin.unique():
                df[f"{col}_ub{level}"] = df[col] * (df["util_bin"] == level).astype(int)

        # De-mean all variables
        for col in (
            self.dependent_vars + self.endog_vars + self.exog_vars + self.instruments
        ):

            df[f"{col}_dm"] = df[col] - df.groupby("year")[col].transform("mean")

        self.data = df

    def run_regressions(self):
        """
        Run regressions on naics data
        """
