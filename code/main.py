# %%
import utils
from utils import *
import usfirms.demand_shocks as ds
import usfirms.naics_demand_shocks as nds
import usfirms.regressions as rr
import usfirms.cross_country_results as cc
import usfirms.firm_duration as fd
import usfirms.match_firms_products as mf

from dotenv import load_dotenv

load_dotenv()

# %%
reload(ds)
reload(nds)
reload(rr)
reload(cc)
reload(fd)
reload(utils)

# %%
data_folder = os.path.join("..", "data", "inelastic-capital-data")
file_path = f"{data_folder}/working/baci.p"

# %% Match firms to products
# mf.match_firms_products()

# %% Load baci data, which is main dataset for analysis
# print("Loading BACI data")
# baci = utils.load_baci_data(data_folder)
# baci = baci[baci.exporter.isin(country_codes)].copy()

# baci.to_pickle(f"{data_folder}/working/baci.p")
baci = pd.read_pickle(f"{data_folder}/working/baci.p")

# Merge baci with naics-hs crosswalk
merge_keys = load_naics_hs_crosswalk(data_folder)
baci = baci.merge(
    merge_keys[["naics", "naics5", "naics4", "naics3", "HS6"]],
    on=["HS6"],
    how="left",
)

baci_us_exports = baci[baci["exporter"] == 842].copy()

# # %% Get absolute and relative prices for products by country
# for code in ["HS6", "HS4", "naics"]:
#     df = utils.get_product_prices(baci, groupby=["exporter", code])
#     df.to_pickle(f"{data_folder}/working/baci_product_prices_{code}.p")

#     # Construct relative prices
#     df = utils.construct_relative_prices(df, agg_level=code)
#     df.to_pickle(f"{data_folder}/working/baci_relative_prices_{code}.p")


# # %% Create demand shocks
# demand_shocks = ds.DemandShocks(data_folder=data_folder, baci=baci_us_exports)
# demand_shocks.initialize_all_demand_shocks()

# # %%
# results_HS6 = rr.RegressionResults(demand_shocks)
# results_HS6.run_baseline_regressions()
# results_HS6.persistence()
# # %%
# results_HS4 = rr.RegressionResults(demand_shocks, agg_level="HS4")
# results_HS4.run_baseline_regressions()

# %%
# results_naics = rr.RegressionResults(demand_shocks, agg_level="naics")
# results_naics.run_naics_regressions()

# # %% Run country-by-country
# elas_df = cc.get_country_results(demand_shocks, data_folder)
# cc.bar_plots(elas_df)
# cc.double_bar_plots(elas_df)
# cc.elas_vs_concentration(elas_df, data_folder)
# # %%

# %%
# fd.duration_elas_regressions(data_folder, results_HS4)

# %%
NDS = nds.NaicsDemandShocks(data_folder)
NDS.initialize_data()


# %%
self = NDS
df = self.data.copy().reset_index()
df_nowin = self.data_nowin.copy()

# %%
# Elasticity estimate
model = iv_panel_reg(
    df,
    dep_var="Dln_Pip",
    exog=["Dln_capacity", "Dln_UVCip", "L1_util"]
    + [f"exp_sh_{year}" for year in sorted(df.year.unique())],
    endog=["Dln_ip"],
    instruments=["Dln_M_shea_inst2", "Dln_frgn_rgdp", "Dln_er"],
)

# %% Interact price elasticity with investment at time of shock
shocks = ["Dln_M_shea_inst2", "Dln_frgn_rgdp", "Dln_er"]
for shock in shocks:
    df[f"{shock}_x_Dln_invest"] = df[shock] * df["Dln_invest"]

# %%
for f in range(1, 4):
    print(f"Forward lag {f}")
    print("-" * 100)
    df = get_lag(df, group_cols=["naics3"], shift_col="ln_Pip", shift_amt=-f)

    m = regfe(
        df,
        dep_var=f"F{f}_ln_Pip",
        ind_vars=shocks
        + [f"{shock}_x_Dln_invest" for shock in shocks]
        + [f"exp_sh_{year}" for year in sorted(df.year.unique())],
        fe_vars=["year", "naics3"],
    )

    for shock in shocks:
        shock_interact = f"{shock}_x_Dln_invest"
        shock_coef = m.coef()[shock]
        shock_se = m.se()[shock]
        shock_interact_coef = m.coef()[shock_interact]
        shock_interact_se = m.se()[shock_interact]

        print(f"{shock}: {shock_coef:.3f} ({shock_se:.3f})")
        print(f"{shock_interact}: {shock_interact_coef:.3f} ({shock_interact_se:.3f})")
        print("\n")
# %%
for f in range(1, 4):
    print(f"Forward lag {f}")
    print("-" * 100)
    df = get_lag(df, group_cols=["naics3"], shift_col="Dln_Pip", shift_amt=-f)

    m = regfe(
        df,
        dep_var=f"F{f}_Dln_Pip",
        ind_vars=shocks
        + [f"{shock}_x_Dln_invest" for shock in shocks]
        + [f"exp_sh_{year}" for year in sorted(df.year.unique())],
        fe_vars=["year", "naics3"],
    )

    for shock in shocks:
        shock_interact = f"{shock}_x_Dln_invest"
        shock_coef = m.coef()[shock]
        shock_se = m.se()[shock]
        shock_interact_coef = m.coef()[shock_interact]
        shock_interact_se = m.se()[shock_interact]

        print(f"{shock}: {shock_coef:.3f} ({shock_se:.3f})")
        print(f"{shock_interact}: {shock_interact_coef:.3f} ({shock_interact_se:.3f})")
        print("\n")

# %%
model = iv_panel_reg(
    df_nowin,
    dep_var="Dln_Pip",
    exog=["Dln_capacity", "Dln_UVCip", "L1_util", "L1_util_x_Dln_capacity"]
    + [f"exp_sh_{year}" for year in sorted(df.year_.unique())],
    endog=["Dln_ip", "L1_util_x_Dln_ip"],
    instruments=[
        "Dln_M_shea_inst2",
        "Dln_frgn_rgdp",
        "Dln_er",
        "L1_util_x_Dln_M_shea_inst2",
        "L1_util_x_Dln_frgn_rgdp",
        "L1_util_x_Dln_er",
    ],
)

# %% Interact with utilization rate
model = iv_panel_reg(
    df,
    dep_var="Dln_Pip",
    exog=["Dln_capacity", "Dln_UVCip"]
    + [f"exp_sh_{year}" for year in sorted(df.year_.unique())],
    endog=["Dln_ip"],
    instruments=["Dln_M_shea_inst2", "Dln_frgn_rgdp", "Dln_er"],
)

# %%
for start, end in [
    # (1973, 1980),
    # (1981, 1990),
    # (1991, 2000),
    # (2001, 2007),
    # (2008, 2018),
    (1973, 1980),
    (1980, 1997),
    (1997, 2007),
    (2007, 2018),
]:
    print(f"Results for {start}-{end}")
    sub = df[(df.year_ >= start) & (df.year_ <= end)]
    model = iv_panel_reg(
        sub,
        dep_var="Dln_Pip",
        exog=["Dln_capacity", "Dln_UVCip"]
        + [f"exp_sh_{year}" for year in sorted(sub.year_.unique())],
        endog=["Dln_ip"],
        instruments=["Dln_M_shea_inst2", "Dln_frgn_rgdp", "Dln_er"],
    )
    print(f"Dln_ip: {model.params['Dln_ip']:.3f} ({model.std_errors['Dln_ip']:.3f})\n")
    # print(model.summary)

# %% Interact with utilization rate
sub = df.copy()

exog = (
    [f"exp_sh_{year}" for year in sorted(sub.year_.unique())]
    + ["Dln_UVCip"]
    + ["ub2", "ub3", "ub4"]
)
for col in ["Dln_capacity"]:
    exog.extend([c for c in df.columns if re.search(rf"{col}_ub\d", c)])

endog = []
for col in self.endog_vars:
    endog.extend([c for c in df.columns if re.search(rf"{col}_ub\d", c)])

instruments = []
for col in ["Dln_frgn_rgdp", "Dln_M_shea_inst2"]:
    instruments.extend([c for c in df.columns if re.search(rf"{col}_ub\d", c)])


iv_panel_reg(sub, dep_var="Dln_Pip", exog=exog, endog=endog, instruments=instruments)

# %% Get elasticity for each industry
naics3_codes = sorted(df.index.get_level_values("naics3").unique())
elas = []
se = []

for n in naics3_codes:
    sub = df[df.index.get_level_values("naics3") == n]
    instruments = ["Dln_frgn_rgdp_dm", "Dln_er_dm"]
    if sub.Dln_M_shea_inst2.mean() != 0:
        instruments.append("Dln_M_shea_inst2")

    # instruments = ["Dln_frgn_rgdp_dm", "Dln_er_dm", "Dln_M_shea_inst2"]
    # if sub.Dln_M_shea_inst2.mean() == 0:
    #     elas.append(np.nan)
    #     se.append(np.nan)
    #     continue

    model = iv_panel_reg(
        sub,
        dep_var="Dln_Pip_dm",
        exog=["Dln_capacity_dm", "Dln_UVCip_dm", "Lexp_share"],
        endog=["Dln_ip_dm"],
        instruments=instruments,
        time_fe=False,
        group_fe=False,
    )
    print(n)
    print(
        f"Dln_ip: {model.params['Dln_ip_dm']:.3f} ({model.std_errors['Dln_ip_dm']:.3f})\n"
    )

    elas.append(model.params["Dln_ip_dm"])
    se.append(model.std_errors["Dln_ip_dm"])

elas_df = pd.DataFrame({"naics3": naics3_codes, "elas": elas, "se": se})
elas_df = elas_df[np.abs(elas_df["elas"]) > elas_df["se"]].copy()
elas_df = elas_df.sort_values("elas", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots()

# Plot coefficients with 95% confidence intervals (1.96 * standard error)
ax.errorbar(
    elas_df.naics3, elas_df["elas"], yerr=1.96 * elas_df["se"], fmt="o", capsize=5
)

# Label axes
ax.set_xlabel("Naics")
ax.set_ylabel("Inverse Elasticity Estimate")

# Add a horizontal line at zero for reference
ax.axhline(0, linestyle="--")

# Improve layout for label readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% Plot investment rates for each industry
fig, ax = plt.subplots(figsize=(10, 6))
for n in naics3_codes:
    sub = df[(df.index.get_level_values("naics3") == n) & (df.year_ <= 2016)]
    ax.plot(sub.year_, sub.invest / sub.vprod, label=n)

ax.set_xlabel("Year")
ax.set_ylabel("Investment As Share of Production Value")
ax.legend(title="Naics3", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
for n in naics3_codes:
    sub = df[(df.index.get_level_values("naics3") == n) & (df.year_ <= 2016)]
    ax.plot(sub.year_, sub.invest / sub.capital, label=n)

ax.set_xlabel("Year")
ax.set_ylabel("Investment As Share of Capital Stock")
ax.legend(title="Naics3", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# %%
