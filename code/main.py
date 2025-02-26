# %%
import utils
from utils import *
import usfirms.demand_shocks as ds
import usfirms.regressions as rr


def year_by_year_regressions(df, indep_var="shock5_dm", dep_var="d5_log_value_dm"):
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
reload(ds)
reload(rr)
reload(utils)

# %%
data_folder = os.path.join("..", "data")

# %% Load baci data, which we will use for many things
baci = utils.load_baci_data(data_folder)
baci_us_exports = baci[baci["exporter"] == 842].copy()

####### TO DO: Use more recent HS codes

# %% Create demand shocks
demand_shocks = ds.DemandShocks(data_folder=data_folder, baci=baci)
demand_shocks.initialize_all_demand_shocks()

# %%
reload(rr)
results = rr.RegressionResults(demand_shocks)
results.panel_regression_table()

# %%
###### Year-by-Year Regressions
year_by_year_regressions(df, indep_var="shock5_dm", dep_var="d5_log_value_dm")
year_by_year_regressions(df, indep_var="shock5_dm", dep_var="d5_log_quantity_dm")
year_by_year_regressions(df, indep_var="shock5_dm", dep_var="d5_log_price_dm")


# %%
