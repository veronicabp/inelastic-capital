# sam sandbox for inelastic capital RHS variable creation 
# conda environment: inelastic_capital


# what I needed to pip, conda install that I didn't have for replication ----------
# ipykernel (vscode will do automatically)
# notebook -- this was a pain, should manually install in conda env
# pip install linearmodels # or conda install? 
# conda install matplotlib 
# pip install pyfixest 
# conda install scipy 
# conda install tqdm
# pip install stargazer 
# conda install requests 

# what I pip/conda installed myself ---------- 
# conda install xlrd

#%%
import os
os.chdir('C:/Users/illge/Princeton Dropbox/Sam Barnett/inelastic_capital/code')
print(os.getcwd())

#%% Run once to get code folder on PATH
import sys
sys.path.append("C:/Users/illge/Princeton Dropbox/Sam Barnett/inelastic_capital/code") 
print(sys.version); print(sys.executable); print(sys.path)

#%% Manually fix the GDP data csv 
# file = os.path.join(
#     data_folder,
#     "raw",
#     "worldbank",
#     "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_88",
#     "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_88.csv",
# )
# df = pd.read_csv(file)

# df.rename(columns={
#     'Series Code': 'Indicator Code',
#     'Series Name': 'Indicator Name'
# }, inplace=True)
# # Add two empty rows at the top
# empty = pd.DataFrame([[""] * len(df.columns), [""] * len(df.columns)], columns=df.columns)
# df_with_empty = pd.concat([empty, df], ignore_index=True)

# df_with_empty.columns = [col.split(' ')[0] if 'YR' in col else col for col in df.columns]
# # Write back to CSV
# df_with_empty.to_csv(file, index=False)

# Final manual step, in Excel: put the two empty rows above the header in the csv file

# Also manually fixed BACI country codes in Excel  

#%% Set "data" folder 
import utils
from utils import *

reload(utils)

folder = "C:/Users/illge/Princeton Dropbox/Sam Barnett/inelastic_capital"
data_folder = os.path.join("..", "data")





#%% Get GDP, Exchange rate, and HS-NAICS crosswalk data
gdp_data = load_gdp_data(data_folder)

exchange_rate_data = load_exchange_rate_data(data_folder)   
#USD / LCU
#can multiply a value in local currency by exchange_rate to convert it into USD.

naics_hs_crosswalk = load_naics_hs_crosswalk(data_folder)




# %% Get FRB data on capacity, utilization
import pandas as pd

def split_whitespace_column(df, col, n):
    """
    Split df[col] (a whitespace-separated string) into exactly n columns.
    """
    # 1) strip surrounding quotes
    s = df[col].astype(str).str.strip('"')
    # 2) split on any run of whitespace, expand into DataFrame
    parts = s.str.split(r"\s+", expand=True)
    # 3) pad (or truncate) to exactly n columns
    parts = parts.reindex(columns=range(n), fill_value="")
    # 4) rename
    parts.columns = [f"{col}_{i+1}" for i in range(n)]
    # 5) drop original and concat
    df2 = pd.concat([df.drop(columns=[col]), parts], axis=1)
    return df2

fcapacity_path = os.path.join(data_folder, "raw", "frb", "fred_capacity.txt")
futilization_path = os.path.join(data_folder, "raw", "frb", "fred_utilization.txt")

# read in the capacity data ----------
df = pd.read_csv(fcapacity_path, sep="\t", header=0) 
df = split_whitespace_column(df, "B50001: Total index", 14)

#rename B50001: Total index_2 to year 
df.rename(columns={"B50001: Total index_2": "year"}, inplace=True)
#drop any row where year is not a number
df = df[pd.to_numeric(df["year"], errors="coerce").notnull()]
#convert year to int
df["year"] = df["year"].astype(int)

#rename B50001: Total index_1 to NAICS code
df.rename(columns={"B50001: Total index_1": "naics_code"}, inplace=True)

#rename BS50001: Total index_`n' to capacity_`n-2'
for n in range(3, 15):
    df.rename(columns={f"B50001: Total index_{n}": f"capacity_{n-2}"}, inplace=True)
#convert capacity_1 to capacity_12 to numeric
for n in range(1, 13):
    df[f"capacity_{n}"] = df[f"capacity_{n}"].astype(float)

#average capacity_1 to capacity_12 in a new column called mean_capacity
df["mean_capacity"] = df[[f"capacity_{i}" for i in range(1, 13)]].mean(axis=1)
# #drop capacity_1 to capacity_12
df.drop(columns=[f"capacity_{i}" for i in range(1, 13)], inplace=True)
frb_capacity = df 

# read in the utilization data ----------
df = pd.read_csv(futilization_path, sep="\t", header=0)
df = split_whitespace_column(df, "B50001: Total index", 14)

#rename B50001: Total index_2 to year
df.rename(columns={"B50001: Total index_2": "year"}, inplace=True)
#drop any row where year is not a number
df = df[pd.to_numeric(df["year"], errors="coerce").notnull()]
#convert year to int
df["year"] = df["year"].astype(int)

#rename B50001: Total index_1 to NAICS code
df.rename(columns={"B50001: Total index_1": "naics_code"}, inplace=True)
#rename BS50001: Total index_`n' to utilization_`n-2'
for n in range(3, 15):
    df.rename(columns={f"B50001: Total index_{n}": f"utilization_{n-2}"}, inplace=True)
#convert utilization_1 to utilization_12 to numeric
for n in range(1, 13):
    df[f"utilization_{n}"] = df[f"utilization_{n}"].astype(float)

#average utilization_1 to utilization_12 in a new column called mean_utilization
df["mean_utilization"] = df[[f"utilization_{i}" for i in range(1, 13)]].mean(axis=1)
# #drop utilization_1 to utilization_12
df.drop(columns=[f"utilization_{i}" for i in range(1, 13)], inplace=True)
frb_utilization = df

#Merge the two dataframes on year and naics_code
capacity_utilization = pd.merge(frb_capacity, frb_utilization, on=["year", "naics_code"])




#%% Import NAICS codes used by Boehm etal 2022 and get their 21 3-digit manufacturing codes
naics_path = os.path.join(data_folder, "raw", "census", "2-digit_2012_Codes.xls")
naics_codes = pd.read_excel(naics_path, sheet_name="tbl_2012_title_description_coun")

#rename 2012 NAICS US Code to naics_code
naics_codes.rename(columns={"2012 NAICS US   Code": "naics_code"}, inplace=True)

#keep only length-3 naics_codes 
naics_codes = naics_codes[naics_codes["naics_code"].astype(str).str.len() == 3]

#merge capacity_utilization with naics_codes on naics_code; first strip naics_code in capcity_utilization of non-numeric characters
capacity_utilization["naics_code"] = capacity_utilization["naics_code"].astype(str).str.extract(r"(\d+)")
capacity_utilization = pd.merge(capacity_utilization, naics_codes, on="naics_code", how="left")

#keep if length of naics_code is 3 and starts with 3, nonmissing naics_code
capacity_utilization = capacity_utilization[
    (capacity_utilization["naics_code"].astype(str).str.len() == 3) &
    (capacity_utilization["naics_code"].astype(str).str.startswith("3")) &
    (capacity_utilization["naics_code"].notnull())
]

#print levels of naics_code in capacity_utilization
print(capacity_utilization["naics_code"].unique()) # 21 levels, as expected 

# demean capacity and utilization by naics_code
capacity_utilization["mean_capacity_demeaned"] = capacity_utilization.groupby("naics_code")["mean_capacity"].transform(lambda x: x - x.mean())
capacity_utilization["mean_utilization_demeaned"] = capacity_utilization.groupby("naics_code")["mean_utilization"].transform(lambda x: x - x.mean())




# %% FRB industrial production index 'INDPRO', by year 
indpro_path = os.path.join(data_folder, "raw", "frb", "INDPRO.csv")
df = pd.read_csv(indpro_path, sep=",", header=0) 
#save first four digits of observtion date as year
df["year"] = df["observation_date"].str[:4].astype(int)
#collapse (mean) by year 
df = (
    df
    .groupby("year", as_index=False)["INDPRO"]
    .mean()
    .rename(columns={"INDPRO": "mean_indpro"})
)




# %% BEA data for Shea instrument






# %%
