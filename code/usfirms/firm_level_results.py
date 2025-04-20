from utils import *
import utils
import usfirms.demand_shocks as ds

data_folder = os.path.join("..", "data")

params = {
    "agg_level": ["naics"],  # , "HS2", "naics"
    "growth_var": ["gdp"],  # "imports"
    "use_prev_year": [True],
    "share_var": ["quantity"],
}
demand_shocks_naics = ds.DemandShocks(
    data_folder=data_folder, baci=baci_us_exports, parameters=params
)
demand_shocks_naics.initialize_all_demand_shocks()
shocks = demand_shocks_naics.get_demand_shocks(agg_level="naics")

for col in ["capx", "xrd"]:
    df = pd.read_csv(f"{data_folder}/raw/wrds/Compustat_Fundamentals_Yearly_all.csv")
    df = df.rename(columns={"fyear": "year"})

    df["sale_avg"] = df.groupby("gvkey")["sale"].transform("mean")
    df = df[df.sale_avg > 0].copy()

    df = df.merge(shocks, on=["naics", "year"], how="inner")
    df = df.sort_values(by=["gvkey", "year"])

    df[f"log_cum_{col}"] = np.log(
        df[col]
        + df.groupby("gvkey")[col].shift(1)
        + df.groupby("gvkey")[col].shift(2)
        + df.groupby("gvkey")[col].shift(3)
        + df.groupby("gvkey")[col].shift(4)
    )

    df = df[np.abs(df[f"log_cum_{col}"]) != np.inf].copy()

    df[f"log_cum_{col}_win"] = winsorize(df[f"log_cum_{col}"], limits=(0.05, 0.05))
    df[f"shock_win"] = winsorize(df[f"shock"], limits=(0.05, 0.05))

    df = df.set_index(["gvkey", "year"])

    utils.binscatter_plot(
        df,
        "shock_win",
        f"log_cum_{col}_win",
        num_bins=100,
        time_fe=True,
        group_fe=True,
        weights="sale_avg",
    )


for col in ["sale", "at", "emp"]:
    df = pd.read_csv(f"{data_folder}/raw/wrds/Compustat_Fundamentals_Yearly_all.csv")
    df = df.rename(columns={"fyear": "year"})

    df["sale_avg"] = df.groupby("gvkey")["sale"].transform("mean")
    df = df[df.sale_avg > 0].copy()

    df = df.merge(shocks, on=["naics", "year"], how="inner")
    df = df.sort_values(by=["gvkey", "year"])

    df[f"d5_log_{col}"] = np.log(df[col]) - np.log(df.groupby("gvkey")[col].shift(5))

    df = df[np.abs(df[f"d5_log_{col}"]) != np.inf].copy()

    df[f"d5_log_{col}_win"] = winsorize(df[f"d5_log_{col}"], limits=(0.05, 0.05))
    df[f"shock_win"] = winsorize(df[f"shock"], limits=(0.05, 0.05))

    df = df.set_index(["gvkey", "year"])

    utils.binscatter_plot(
        df,
        "shock_win",
        f"d5_log_{col}_win",
        num_bins=100,
        time_fe=True,
        group_fe=True,
        weights="sale_avg",
    )
