# %%
import utils
from utils import *
import usfirms.demand_shocks as dm

#### Main table: (3x for value, quantity, price) HS6 (all years), HS4 (all years), HS6 (2002-2007) -- in all correct SE, repeat for baseline + all interactions


class RegressionResults:
    def __init__(self, ds):
        self.demand_shocks = ds

    def primary_regression_table(
        self, indep_vars=["shock5_win"], dep_vars=["value", "quantity", "price"]
    ):
        models = []

        # Cols 1-3: HS6-level regressions, panel
        for var in dep_vars:
            model = regfe(
                self.HS6_df, f"d5_log_{var}_win", indep_vars, fe_vars=["year", "HS6"]
            )
            models.append(model)


def year_by_year_regressions(df, indep_var="shock5_dm", dep_var="d5_log_value_dm"):
    df = df.dropna(subset=["d5_log_value", "shock5"]).copy()
    for col in ["value", "quantity", "price"]:
        df[f"d5_log_{col}_dm"] = utils.demean_by_fixed_effects(
            df, f"d5_log_{col}_win", fe_vars=[agg_level]
        )
    df[f"shock5_dm"] = utils.demean_by_fixed_effects(
        df, f"shock5_win", fe_vars=[agg_level]
    )

    years = []
    coeffs = []
    ses = []
    for year in range(2000, 2021):
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
    plt.axhline(0)


# %%
reload(dm)
reload(utils)

# %%
data_folder = os.path.join("..", "data")

# %% Load baci data, which we will use for many things
baci = utils.load_baci_data(data_folder)
baci_us_exports = baci[baci["exporter"] == 842].copy()

####### TO DO: Use more recent HS codes

# %% Create demand shocks
ds = dm.DemandShocks(data_folder=data_folder, baci=baci)
ds.initialize_all_demand_shocks()

# %%
agg_level = "HS6"
use_prev_year = False
growth_var = "gdp"

bartik_shocks = ds.get_demand_shocks(
    agg_level=agg_level, use_prev_year=use_prev_year, growth_var=growth_var
)

prices_quantities = utils.get_product_prices(baci_us_exports, agg_level=agg_level)
df = prices_quantities.merge(bartik_shocks, on=[agg_level, "year"], how="left")

# %%
df = df.sort_values(by=[agg_level, "year"])

df["sector_mn_val"] = df.groupby([agg_level])["value"].transform("mean")

for col in ["value", "quantity", "price"]:
    df = utils.get_lag(df, [agg_level, "year"], shift_col=col)
    df[f"d_log_{col}"] = np.log(df[col]) - np.log(df[f"L1_{col}"])

    df = utils.get_lag(df, [agg_level, "year"], shift_col=col, shift_amt=5)
    df[f"d5_log_{col}"] = np.log(df[col]) - np.log(df[f"L5_{col}"])

# Aggregate shocks over 5 years
for i in range(1, 5):
    df = utils.get_lag(df, [agg_level, "year"], shift_col="shock", shift_amt=i)
shock_cols = [col for col in df.columns if "shock" in col]
df["shock5"] = df[shock_cols].sum(axis=1, min_count=len(shock_cols))

# %%
# Winsorize
for col in [
    col
    for col in df.columns
    if col.startswith("d")
    or col.startswith("shock")
    or col in ["value", "quantity", "price"]
]:
    df[f"{col}_win"] = winsorize(df[col], limits=(0.05, 0.05))

df = df.dropna(subset="d_log_value")
df = df.dropna(subset="shock")

utils.binscatter_plot(
    df, "shock5_win", "d5_log_value_win", num_bins=100, fe_vars=[agg_level, "year"]
)

###### Panel Regressions
result = utils.regfe(
    df, "d5_log_value_win", ["shock5_win"], fe_vars=["year", agg_level]
)
print(result.summary())

result = utils.regfe(
    df, "d5_log_quantity_win", ["shock5_win"], fe_vars=["year", agg_level]
)
print(result.summary())

result = utils.regfe(
    df, "d5_log_price_win", ["shock5_win"], fe_vars=["year", agg_level]
)
print(result.summary())

# %%
###### Year-by-Year Regressions
year_by_year_regressions(df, indep_var="shock5_dm", dep_var="d5_log_value_dm")
year_by_year_regressions(df, indep_var="shock5_dm", dep_var="d5_log_quantity_dm")
year_by_year_regressions(df, indep_var="shock5_dm", dep_var="d5_log_price_dm")

# %%
###### Interacted Regressions
# Interaction vars: (1) market concentration index, (2) regulation index, (3) Antras production time measure, (4) product complexity index (PCI)

# Load cross walks
naics_hs_crosswalk = utils.load_naics_hs_crosswalk(data_folder)
sic_hs_crosswalk = utils.load_sic_hs_crosswalk(data_folder)

sic_hs_crosswalk = sic_hs_crosswalk.drop_duplicates(subset=[agg_level, "sic"])

# Merge concentration ratios
concentration = pd.read_excel(
    f"{data_folder}/raw/census/economic_census/concentration92-47.xls", header=3
).rename(
    columns={
        "SIC Code": "sic",
        "4 largest companies": "CR4",
        "Herfindahl-Hirschman Index for 50 largest companies": "HHI",
    }
)
concentration = concentration[concentration.YR == 92][["sic", "CR4", "HHI"]]

concentration = concentration.merge(sic_hs_crosswalk, on="sic")
concentration = concentration.groupby([agg_level])[["CR4", "HHI"]].mean().reset_index()

df = df.merge(concentration, on=[agg_level], how="left")

# Merge regulation index
regulation_probs = pd.read_csv(
    f"{data_folder}/raw/quantgov/RegData-US_5-0/usregdata5.csv"
)
regulation_naics = pd.read_csv(
    f"{data_folder}/raw/quantgov/RegData-US_5-0/regdata_5_0_naics07_3digit.csv"
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
    naics_hs_crosswalk.drop_duplicates(subset=["naics3", "HS6"]),
    on=["naics3"],
    how="inner",
)
regulation = regulation.groupby(agg_level)["naics_restrictions"].mean().reset_index()
regulation["log_naics_restrictions"] = np.log(regulation.naics_restrictions)

df = df.merge(regulation, on=agg_level, how="left")

###### Merge production-side constraints

# Antras APP measure
antras_app = pd.read_csv(
    f"{data_folder}/raw/original/antras_tubdenov_2025/complete_ranking_usa_goods.csv"
)
antras_app = antras_app.merge(
    naics_hs_crosswalk.rename(columns={"naics": "naics6"}),
    on=["naics6"],
    how="inner",
)
antras_app = antras_app.groupby(agg_level)["invtCogsRatio"].mean().reset_index()

df = df.merge(antras_app, on=agg_level, how="left")

# Product complexity
pci = pd.read_csv(f"{data_folder}/raw/pci/{agg_level}.csv").rename(
    columns={agg_level: "description", f"{agg_level} ID": agg_level}
)
pci[agg_level] = pci[agg_level].apply(lambda x: str(x).zfill(6))
year_cols = [col for col in pci.columns if re.search(r"\d{4}", str(col))]
pci["pci"] = pci[year_cols].sum(axis=1, min_count=len(year_cols)) / len(year_cols)
df = df.merge(pci[[agg_level, pci]], on=agg_level, how="left")

# Capacity constraints

# %% Regressions

for var in ["value", "quantity", "price"]:
    indep_var = "shock5_win"
    dep_var = f"d5_log_{var}_win"

    # ##### Regulation
    # df["shock_x_restrict"] = df[indep_var] * df["log_naics_restrictions"]

    # result = utils.regfe(
    #     df, dep_var, [indep_var, "shock_x_restrict"], fe_vars=["year", agg_level]
    # )
    # print(result.summary())

    # ##### Concentration
    # df["shock_x_CR4"] = df[indep_var] * df["CR4"] / 1000
    # df["shock_x_HHI"] = df[indep_var] * df["HHI"] / 1000

    # result = utils.regfe(
    #     df, dep_var, [indep_var, "shock_x_CR4"], fe_vars=["year", agg_level]
    # )
    # print(result.summary())

    # result = utils.regfe(
    #     df, dep_var, [indep_var, "shock_x_HHI"], fe_vars=["year", agg_level]
    # )
    # print(result.summary())

    # ##### Average production period
    # df["shock_x_app"] = df[indep_var] * df["invtCogsRatio"]

    # result = utils.regfe(
    #     df, dep_var, [indep_var, "shock_x_app"], fe_vars=["year", agg_level]
    # )
    # print(result.summary())

    #####
    df["shock_x_pci"] = df[indep_var] * df["pci"]

    result = utils.regfe(
        df, dep_var, [indep_var, "shock_x_pci"], fe_vars=["year", agg_level]
    )
    print(result.summary())

# %%
