# %%
from utils import *
from construct_price_rent import *

data_folder = os.path.join("..", "data", "inelastic-capital-data")


def overlap_crosswalk(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    left_id: str,
    right_id: str,
    *,
    ea_crs: str = "EPSG:5070",
    predicate: str = "intersects",
    share_side: str = "left",  # "left" => share of left unit; "right" => share of right unit
    drop_slivers_m2: float | None = 0.001,
    keep_geometry: bool = False,
    one_to_one: (
        str | None
    ) = None,  # None | "left" | "right"  (take max-share match per id on that side)
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Build an overlap crosswalk between two polygon layers.

    Parameters
    ----------
    left_gdf, right_gdf : GeoDataFrame
        Polygon layers. Must have a unique id column each.
    left_id, right_id : str
        Column names of unique ids in `left_gdf` and `right_gdf`.
    ea_crs : str, default "EPSG:5070"
        Equal-area CRS used for accurate area computations. (Change to EPSG:2163 for nationwide incl. AK/HI/PR.)
    predicate : {"intersects","overlaps","contains","within","touches","crosses"}, default "intersects"
        Spatial join predicate used to prefilter candidate pairs.
    share_side : {"left","right"}, default "left"
        Which side the share is computed against:
        - "left"  => share = intersection_area / area(left polygon)
        - "right" => share = intersection_area / area(right polygon)
        Both shares are returned; this just controls which one is named `share`.
    drop_slivers_m2 : float or None, default 0.01
        If provided, drops intersections with share area < threshold.
    keep_geometry : bool, default False
        If True, returns a GeoDataFrame with the intersection geometry. Otherwise returns a plain DataFrame.
    one_to_one : None | {"left","right"}, default None
        If set, keeps only the max-share match per id on the chosen side.

    Returns
    -------
    DataFrame or GeoDataFrame
        Columns:
          - left_id, right_id
          - inter_area, left_area, right_area
          - share_of_left, share_of_right
          - share  (alias of chosen side per `share_side`)
        Plus `geometry` (intersection) if `keep_geometry=True`.

    Notes
    -----
    - Computes areas in `ea_crs` to ensure correct area measures.
    - Fixes invalid geometries via buffer(0).
    - Uses a spatial join to prefilter candidate pairs (big speed-up).
    """
    # 0) Basic checks
    if left_id not in left_gdf.columns:
        raise KeyError(f"`left_id` '{left_id}' not found in left_gdf")
    if right_id not in right_gdf.columns:
        raise KeyError(f"`right_id` '{right_id}' not found in right_gdf")
    if share_side not in {"left", "right"}:
        raise ValueError("share_side must be 'left' or 'right'")
    if one_to_one not in {None, "left", "right"}:
        raise ValueError("one_to_one must be None, 'left', or 'right'")

    # 1) Reproject to equal-area CRS
    L = left_gdf[[left_id, left_gdf.geometry.name]].copy()
    R = right_gdf[[right_id, right_gdf.geometry.name]].copy()
    L = L.to_crs(ea_crs)
    R = R.to_crs(ea_crs)

    # 2) Fix invalid geometries (common with admin boundaries)
    L.geometry = L.buffer(0)
    R.geometry = R.buffer(0)

    # 3) Spatial-join to get candidate pairs
    cand = gpd.sjoin(
        L,
        R,
        how="inner",
        predicate=predicate,
    )[[left_id, right_id]].drop_duplicates()

    if cand.empty:
        cols = [
            left_id,
            right_id,
            "inter_area",
            "left_area",
            "right_area",
            "share_of_left",
            "share_of_right",
            "share",
        ]
        return (
            gpd.GeoDataFrame(columns=cols, geometry=[]).drop(columns="geometry")
            if not keep_geometry
            else gpd.GeoDataFrame(columns=cols, geometry=[])
        )

    # 4) Attach geometries for pairwise intersection
    L_sub = cand.merge(
        L.rename(columns={left_gdf.geometry.name: "geom_left"}),
        on=left_id,
        how="left",
    )
    R_sub = cand.merge(
        R.rename(columns={right_gdf.geometry.name: "geom_right"}),
        on=right_id,
        how="left",
    )
    pairs = L_sub.merge(R_sub, on=[left_id, right_id])

    # 5) Compute intersections
    inter_geom = pairs.apply(lambda r: r.geom_left.intersection(r.geom_right), axis=1)
    inter_gdf = gpd.GeoDataFrame(
        pairs[[left_id, right_id]], geometry=inter_geom, crs=ea_crs
    )
    inter_gdf = inter_gdf[~inter_gdf.is_empty & inter_gdf.geometry.is_valid].copy()
    if inter_gdf.empty:
        cols = [
            left_id,
            right_id,
            "inter_area",
            "left_area",
            "right_area",
            "share_of_left",
            "share_of_right",
            "share",
        ]
        return (
            gpd.GeoDataFrame(columns=cols, geometry=[]).drop(columns="geometry")
            if not keep_geometry
            else gpd.GeoDataFrame(columns=cols, geometry=[])
        )

    # 6) Areas and shares
    # Left/right totals
    L_area = L.copy()
    L_area["left_area"] = L_area.area
    L_area = L_area.rename(columns={left_id: left_id}).drop(
        columns=L_area.geometry.name
    )

    R_area = R.copy()
    R_area["right_area"] = R_area.area
    R_area = R_area.rename(columns={right_id: right_id}).drop(
        columns=R_area.geometry.name
    )

    inter_gdf["inter_area"] = inter_gdf.area

    out = inter_gdf.merge(L_area, on=left_id, how="left").merge(
        R_area, on=right_id, how="left"
    )

    # Shares
    out["share_of_left"] = (out["inter_area"] / out["left_area"]).clip(0, 1)
    out["share_of_right"] = (out["inter_area"] / out["right_area"]).clip(0, 1)
    out["share"] = (
        out["share_of_left"] if share_side == "left" else out["share_of_right"]
    )

    # Optional sliver filter
    if drop_slivers_m2 is not None:
        out = out[out["share"] >= float(drop_slivers_m2)].copy()

    # 7) One-to-one collapsing (take max share per chosen side)
    if one_to_one == "left":
        out = (
            out.sort_values([left_id, "share"], ascending=[True, False])
            .groupby(left_id, as_index=False)
            .first()
        )
    elif one_to_one == "right":
        out = (
            out.sort_values([right_id, "share"], ascending=[True, False])
            .groupby(right_id, as_index=False)
            .first()
        )

    # 8) Return with or without geometry
    core_cols = [
        left_id,
        right_id,
        "inter_area",
        "left_area",
        "right_area",
        "share_of_left",
        "share_of_right",
        "share",
    ]
    if keep_geometry:
        return out[core_cols + [out.geometry.name]]
    else:
        return pd.DataFrame(out.drop(columns=out.geometry.name))


# %% Load data

# Price to rent at CBSA level
ptr_file = os.path.join(
    data_folder, "raw", "original", "campbell_davis_gallin_martin_2009", "CDGM_data.xls"
)
ptr = pd.read_excel(ptr_file)
ptr["year"] = ptr["Year:Half"].astype(int)
ptr = ptr.groupby("year", as_index=False)[
    [c for c in ptr.columns if c not in ["year", "Year:Half"]]
].mean()

cbsa_map = {
    "Chicago": "Chicago-Naperville-Elgin, IL-IN-WI",
    "Cincinnati": "Cincinnati, OH-KY-IN",
    "Cleveland": "Cleveland-Elyria, OH",
    "Detroit": "Detroit-Warren-Dearborn, MI",
    "Kansas City": "Kansas City, MO-KS",
    "Milwaukee": "Milwaukee-Waukesha, WI",
    "Minneapolis": "Minneapolis-St. Paul-Bloomington, MN-WI",
    "St. Louis": "St. Louis, MO-IL",
    "Boston": "Boston-Cambridge-Newton, MA-NH",
    "New York": "New York-Newark-Jersey City, NY-NJ-PA",
    "Philadelphia": "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD",
    "Pittsburgh": "Pittsburgh, PA",
    "Atlanta": "Atlanta-Sandy Springs-Alpharetta, GA",
    "Dallas": "Dallas-Fort Worth-Arlington, TX",
    "Houston": "Houston-The Woodlands-Sugar Land, TX",
    "Miami": "Miami-Fort Lauderdale-Pompano Beach, FL",
    "Denver": "Denver-Aurora-Lakewood, CO",
    "Honolulu": "Urban Honolulu, HI",
    "Los Angeles": "Los Angeles-Long Beach-Anaheim, CA",
    "Portland": "Portland-Vancouver-Hillsboro, OR-WA",
    "San Diego": "San Diego-Chula Vista-Carlsbad, CA",
    "San Francisco": "San Francisco-Oakland-Berkeley, CA",
    "Seattle": "Seattle-Tacoma-Bellevue, WA",
}

ptr = ptr.melt(id_vars=["year"], var_name="cbsa", value_name="rtp")
ptr = ptr.drop(ptr[~ptr["cbsa"].isin(cbsa_map.keys())].index)
ptr["cbsa"] = ptr["cbsa"].map(cbsa_map)

ptr = get_lag(ptr, group_cols=["cbsa"], shift_col="rtp", shift_amt=1)
ptr["d_rtp"] = 100 * (ptr["rtp"] - ptr["L1_rtp"])
ptr["d_log_ptr"] = 100 * (np.log(1 / ptr["rtp"]) - np.log(1 / ptr["L1_rtp"]))
ptr = get_lag(ptr, group_cols=["cbsa"], shift_col="d_log_ptr", shift_amt=1)


# %%

# Elasticity at tract level
elasticity_file = os.path.join(
    data_folder,
    "raw",
    "original",
    "baum-snow_2023",
    "sourcedata",
    "elasticities",
    "gammas_hat_all.dta",
)
elasticity = pd.read_stata(elasticity_file)
gamma_cols = [c for c in elasticity.columns if c.startswith("gamma")]
elasticity = elasticity.groupby(["ctracts2000"], as_index=False)[gamma_cols].mean()

# Number of households at tract level
tract_data_file = os.path.join(
    data_folder,
    "raw",
    "original",
    "baum-snow_2023",
    "sourcedata",
    "census_acs_19702010",
    "tracts_stf4_ncdb.dta",
)
tract_data = pd.read_stata(tract_data_file)
tract_data = tract_data.rename(
    columns={"tract_str": "ctracts2000", "shr9d": "pop_1990"}
)

elasticity = elasticity.merge(
    tract_data[["ctracts2000", "pop_1990"]], on="ctracts2000", how="left"
)

# Deregulation data
fi_folder = os.path.join(
    data_folder, "raw", "original", "fabra_imbs_2015", "20121416_1data", "data"
)

dereg = pd.read_stata(os.path.join(fi_folder, "hp_dereg_controls.dta"))
dereg = dereg.groupby(["state_n", "year"], as_index=False)[
    [
        c
        for c in dereg.columns
        if c not in ["border_name", "border", "msa", "elasticity"]
    ]
].mean()

# Loans data
hmda = pd.read_stata(os.path.join(fi_folder, "hmda.dta"))

# %% Load GIS files
# CBSA boundaries
cbsa_file = os.path.join(
    data_folder,
    "raw",
    "arcgis",
    "USA_Core_Based_Statistical_Area",
    "USA_Core_Based_Statistical_Area.shp",
)
cbsa = gpd.read_file(cbsa_file)

# State boundaries
states_file = os.path.join(
    data_folder, "raw", "arcgis", "US_State_Boundaries", "US_State_Boundaries.shp"
)
state = gpd.read_file(states_file)
state = state[state.ORDER_ADM != 0]

# Tract boundaries
tract_file = os.path.join(
    data_folder,
    "raw",
    "ipums",
    "US_tract_2000_tl10",
    "US_tract_2000_tl10.shp",
)
tract = gpd.read_file(tract_file)

# County boundaries
county_file = os.path.join(
    data_folder,
    "raw",
    "census",
    "geographies",
    "cb_2018_us_county_5m",
    "cb_2018_us_county_5m.shp",
)
county = gpd.read_file(county_file)

# Set same crs
ea_crs = "EPSG:5070"
tract = tract.to_crs(ea_crs)
cbsa = cbsa.to_crs(ea_crs)
state = state.to_crs(ea_crs)
county = county.to_crs(ea_crs)


# %% Merge
state_cbsa = overlap_crosswalk(
    state,
    cbsa,
    left_id="STATE_FIPS",
    right_id="CBSA_ID",
    share_side="right",
    drop_slivers_m2=0.01,
)
cbsa_tract = overlap_crosswalk(
    cbsa,
    tract,
    left_id="CBSA_ID",
    right_id="CTIDFP00",
    share_side="right",
    drop_slivers_m2=0.01,
)

cbsa_county = overlap_crosswalk(
    cbsa,
    county,
    left_id="CBSA_ID",
    right_id="GEOID",
    share_side="right",
    drop_slivers_m2=0.01,
)

# % Collapse elasticity/ptr/loans at CBSA-state level

### BLS data
state_cbsa = state_cbsa.merge(
    cbsa[["CBSA_ID", "NAME", "POPULATION"]],
    on="CBSA_ID",
    how="inner",
).rename(
    columns={
        "NAME": "cbsa",
        "POPULATION": "cbsa_population",
        "CBSA_ID": "cbsa_id",
        "STATE_FIPS": "state_fips",
        "inter_area": "area",
    }
)

ptr_df = ptr.merge(
    state_cbsa[["cbsa_id", "cbsa", "state_fips", "area", "cbsa_population"]],
    on="cbsa",
    how="inner",
)
ptr_df["tot_area"] = ptr_df.groupby(["cbsa_id", "year"])["area"].transform("sum")
ptr_df["sh_of_cbsa"] = ptr_df["area"] / ptr_df["tot_area"]

# Other BLS data
bls_folder = os.path.join(data_folder, "raw", "bls")
bls_cpi = pd.read_csv(os.path.join(bls_folder, "BLS_SA0.csv"))
bls_cpi_serivces = pd.read_csv(os.path.join(bls_folder, "BLS_SAS.csv"))

# Map BLS data to counties based on BLS reported mapping https://www.bls.gov/cpi/additional-resources/geographic-sample.htm


#### Elasticity
cbsa_tract = cbsa_tract.rename(
    columns={
        "CTIDFP00": "ctracts2000",
        "CBSA_ID": "cbsa_id",
        "inter_area": "area",
    }
)
elas_df = elasticity.merge(
    cbsa_tract[["ctracts2000", "area", "cbsa_id"]], on="ctracts2000", how="inner"
)
elas_df["tot_area"] = elas_df.groupby(["ctracts2000"])["area"].transform("sum")
elas_df["sh_of_tract"] = elas_df["area"] / elas_df["tot_area"]

# Collapse at cbsa level taking weighted average
elas_df["pop"] = elas_df["pop_1990"] * elas_df["sh_of_tract"]

for g in gamma_cols:
    elas_df[g] = elas_df[g] * elas_df["pop"]
elas_df = elas_df.groupby(["cbsa_id"], as_index=False).agg(
    {c: "sum" for c in gamma_cols + ["pop"]}
)
for g in gamma_cols:
    elas_df[g] = elas_df[g] / elas_df["pop"]

##### Loans
cbsa_county = cbsa_county.rename(
    columns={
        "GEOID": "fips",
        "CBSA_ID": "cbsa_id",
        "inter_area": "area",
    }
)
cbsa_county["fips"] = cbsa_county["fips"].astype(int)
loan_df = hmda.rename(columns={"county": "fips"}).merge(
    cbsa_county[["fips", "area", "cbsa_id"]], on="fips", how="inner"
)
loan_df["tot_area"] = loan_df.groupby(["cbsa_id", "year"])["area"].transform("sum")
loan_df["sh_of_cbsa"] = loan_df["area"] / loan_df["tot_area"]

cols = [c for c in loan_df.columns if c.startswith("Dl") or c.startswith("LDl")]
for c in cols:
    loan_df[c] = loan_df[c] * loan_df["sh_of_cbsa"]
loan_df = loan_df.groupby(["cbsa_id", "year"], as_index=False).agg(
    {c: "sum" for c in cols}
)


##### Merge
df = ptr_df.merge(elas_df, on=["cbsa_id"], how="inner")
df = df[
    [
        "state_fips",
        "cbsa_id",
        "cbsa",
        "year",
        "rtp",
        "d_rtp",
        "d_log_ptr",
        "L1_d_log_ptr",
        "pop",
        "sh_of_cbsa",
    ]
    + gamma_cols
].rename(columns={"pop": "pop_1990"})
df = df.merge(loan_df, on=["cbsa_id", "year"], how="inner")

df["state_n"] = df["state_fips"].astype(int)
df = df.merge(dereg, on=["state_n", "year"], how="inner")

df = df.sort_values(by=["state_fips", "cbsa_id", "year"])
df["state_cbsa"] = df["state_fips"].astype(str) + "_" + df["cbsa_id"].astype(str)

df.to_csv(os.path.join(data_folder, "working", "ptr_elas_cbsa.csv"), index=False)

# %% Run regressions
# xtivreg2 Dl_hpi (Dl_nloans_b = Linter_bra) LDl_hpi  $D_control_hp yr* [aw=w1], fe r bw(3) partial(yr*)  est store Dl_nloans_b
df = pd.read_csv(os.path.join(data_folder, "working", "ptr_elas_cbsa.csv"))

controls = ["Dl_inc", "LDl_inc", "Dl_pop", "LDl_pop", "Dl_her_v", "LDl_her_v"]
gamma_var = "gamma01a_units_FMM"

df["Ldereg"] = (df["Linter_bra"] > 0).astype(int)

# Reduced form
ols_formula = f"d_log_ptr ~ {'+'.join(controls)} +  Linter_bra + Linter_bra:{gamma_var}  | year + cbsa"
ols_model = feols(
    ols_formula, data=df, weights="sh_of_cbsa", vcov={"CRV1": "state_fips"}
)
print(ols_model.summary())

# Add lag
ols_formula = f"d_log_ptr ~ {'+'.join(controls)} +  L1_d_log_ptr + Linter_bra + Linter_bra:{gamma_var}  | year + cbsa"
ols_model = feols(
    ols_formula, data=df, weights="sh_of_cbsa", vcov={"CRV1": "state_fips"}
)
print(ols_model.summary())


# # %% IV
# df[f"Linter_bra_x_{gamma_var}"] = df["Linter_bra"] * df[gamma_var]
# df[f"Dl_nloans_b_x_{gamma_var}"] = df["Dl_nloans_b"] * df[gamma_var]
# iv_model = iv_panel_reg(
#     df,
#     dep_var="d_log_ptr",
#     exog=controls,
#     endog=["Dl_nloans_b", f"Dl_nloans_b_x_{gamma_var}"],
#     instruments=["Linter_bra", f"Linter_bra_x_{gamma_var}"],
#     fes=["year", "cbsa_id"],
#     robust=False,
#     cluster="state_fips",
# )
# print(iv_model.summary)

# %% Plot rtp for each cbsa

# tab20b color map
cbsas_unique = ptr["cbsa"].unique()


part1 = cbsas_unique[: len(cbsas_unique) // 2]
part2 = cbsas_unique[len(cbsas_unique) // 2 :]


for part in [part1, part2]:
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20b")
    colors = cmap(np.linspace(0, 1, len(part)))

    for i, c in enumerate(part):
        print(c)
        sub = ptr[ptr["cbsa"] == c]
        ax.plot(sub["year"], 100 * sub["rtp"], label=c, color=colors[i])

    # Small legend in top right
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)

    plt.show()

# %% Create data

# # %%
# # Zoom into chica
# fig, ax = plt.subplots(figsize=(8, 8))
# cbsa.plot(ax=ax)
# cbsa[cbsa.NAME.str.contains("Chicago")].plot(ax=ax, color="red")

# # # Plot state boundaries (no fill)
# states.plot(ax=ax, facecolor="none", edgecolor="black")

# tract[tract.STATEFP00 == "17"].plot(ax=ax, color="green", alpha=0.5, edgecolor="black")

# # # Set zoomed-in limits
# ax.set_xlim(-1e7, -0.9e7)
# ax.set_ylim(4.5e6, 5.5e6)

# plt.show()

# %%
