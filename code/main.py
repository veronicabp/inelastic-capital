# %%
import utils
from utils import *
import usfirms.demand_shocks as ds
import usfirms.regressions as rr
import usfirms.cross_country_results as cc
import usfirms.firm_duration as fd
import usfirms.match_firms_products as mf

from dotenv import load_dotenv

load_dotenv()

# %%
reload(ds)
reload(rr)
reload(utils)
reload(cc)
reload(fd)

# %%
data_folder = os.path.join("..", "data")

# %% Match firms to products
mf.match_firms_products()

# # %%
# fd.estimate_industry_duration(data_folder)

# # %% Load baci data, which is main dataset for analysis
# print("Loading BACI data")
# # baci = utils.load_baci_data(data_folder)
# # baci = baci[baci.exporter.isin(country_codes)].copy()

# # baci.to_pickle(f"{data_folder}/working/baci.p")
# baci = pd.read_pickle(f"{data_folder}/working/baci.p")

# baci_us_exports = baci[baci["exporter"] == 842].copy()

# # # %% Get absolute and relative prices for products by country
# # for code in ["HS6", "HS4"]:
# #     df = utils.get_product_prices(baci, groupby=["exporter", code])
# #     df.to_pickle(f"{data_folder}/working/baci_product_prices_{code}.p")

# #     # Construct relative prices
# #     df = utils.construct_relative_prices(df, agg_level=code)
# #     df.to_pickle(f"{data_folder}/working/baci_relative_prices_{code}.p")


# # %% Create demand shocks

# demand_shocks = ds.DemandShocks(data_folder=data_folder, baci=baci_us_exports)
# demand_shocks.initialize_all_demand_shocks()

# # %%
# results_HS6 = rr.RegressionResults(demand_shocks)
# results_HS6.run_baseline_regressions()
# # results_HS6.persistence()
# # %%
# results_HS4 = rr.RegressionResults(demand_shocks, agg_level="HS4")
# results_HS4.run_baseline_regressions()

# # %% Run country-by-country
# elas_df = cc.get_country_results(demand_shocks, data_folder)
# cc.bar_plots(elas_df)
# cc.double_bar_plots(elas_df)
# cc.elas_vs_concentration(elas_df, data_folder)
# # %%
