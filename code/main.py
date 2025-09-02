# %%
import utils
from utils import *
from ushousing.construct_price_rent import geolytics_price_rent
from dotenv import load_dotenv

load_dotenv()


# %%
reload(utils)

# %%
data_folder = os.path.join("..", "data", "inelastic-capital-data")


# %% Look at effect on price to rent ratio (using census data)

folder = os.path.join(data_folder, "raw", "ipums")
df = pd.read_csv(os.path.join(folder, "nhgis0014_csv", "nhgis0014_ds82_1950_tract.csv"))

# %%
for file in os.listdir(os.path.join(folder, "nhgis0014_csv")):
    if file.endswith(f".txt") and "tract" in file:
        # Read in the text file
        with open(os.path.join(folder, "nhgis0014_csv", file), "r") as f:
            lines = f.readlines()

        print(lines[0])
        print(lines[1])
        cont = True
        for line in lines:
            if "Data Type" in line:
                cont = False
            if "Citation and Use" in line:
                cont = True

            if cont:
                continue
            print(line)
        print("\n\n\n")

# %%
df = geolytics_price_rent(data_folder)
fi_folder = os.path.join(
    data_folder, "raw", "original", "fabra_imbs_2015", "20121416_1data", "data"
)
hp_dereg = pd.read_stata(os.path.join(fi_folder, "hp_dereg_controls.dta"))
hp_dereg = hp_dereg[["state_n", "year", "inter_bra"]].drop_duplicates()

# Reshape wide (based on year)
dereg_wide = hp_dereg.pivot(
    index=["state_n"], columns="year", values="inter_bra"
).reset_index()
dereg_wide["reg_change"] = dereg_wide[2005] - dereg_wide[1994]

df["state_n"] = df["state"].astype(int)
df = df.merge(dereg_wide[["state_n", "reg_change"]], on=["state_n"], how="inner")

df["d_rtp"] = 100 * ((1 / df["ptr2010"]) - (1 / df["ptr1990"]))
df["d_log_ptr"] = 100 * (np.log(df["ptr2010"]) - np.log(df["ptr1990"]))

# %%
formula = "d_log_ptr ~ reg_change * gamma01a_units_FMM"
model = feols(
    formula, data=df[df.pop1990_ncdb > 0], vcov="hetero", weights="pop1990_ncdb"
)
model.summary()
# %%
