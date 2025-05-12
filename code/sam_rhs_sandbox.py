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

ungdp_path = os.path.join(data_folder, "raw", "unstats", "un_gdp_constantLCU2015.csv")
ungdp_data = pd.read_csv(ungdp_path, sep=",", header=0)

# FRB ---------------------------------------------------
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
frb_capacity = df.copy() 

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
frb_utilization = df.copy()

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




#%% Exports data from Schott ----------------------------------------

iso_df = load_iso_codes(data_folder) #new utils function
print(iso_df.head())

# get sic/naics concordance and weights 
sic_path = os.path.join(data_folder, "raw", "original", "conc_sic87_naics97.xlsx")
sic_naics = pd.read_excel(sic_path, sheet_name="Data")
sic_naics = sic_naics[["sic87", "naics97", "ship8797"]]
sic_naics["sic87"] = sic_naics["sic87"].astype(str)

# pre-1988 exports data from .dta ----------
path = os.path.join(data_folder, "raw", "original", "xm_sic87_72_105_20120424.dta")
df = pd.read_stata(path, convert_categoricals=False)

# keep wbcode year sic x 
df = df[["wbcode", "year", "sic", "x"]]
# rename x val_exports, tostring sic and keep pre-1988
df.rename(columns={"x": "val_exports"}, inplace=True)
df = df[df["val_exports"] != 0]
df["sic"] = df["sic"].astype(int).astype(str)
df = df[df["sic"] != "nan"]
df = df[df["year"] <= 1988]

#merge df many:many sic87 using sic_naics 
df = pd.merge(df, sic_naics, left_on="sic", right_on="sic87", how="left")
df["val_exports"] = df["val_exports"] * df["ship8797"] # weight by weights following Boehm 2022
df["naics"] = df["naics97"].astype(str)
df["naics3"] = df["naics"].str[:3]
pre_1988_exports = (
    df
    .groupby(["year", "naics3", "wbcode"], as_index=False)["val_exports"]
    .sum()
)
#convert from millions to USD 
pre_1988_exports["val_exports"] = pre_1988_exports["val_exports"] * 1e6

# post-1988 exports data from .dta ----------
# for each year 1989 to 2024, read in the file exp_detl_yearly_`year'_12n.dta
# collapse by naics3 
# and append to df
df = pd.DataFrame()   

for year in range(89, 123):
    path = os.path.join(
        data_folder, "raw", "original", "annual_schott_legacy", f"exp_detl_yearly_{year}n", f"exp_detl_yearly_{year}n.dta"
    )
    df_temp = pd.read_stata(path, convert_categoricals=False)
    df_temp = df_temp[["year", "all_val_yr", "naics", "cty_code"]]
    df_temp.rename(columns={"all_val_yr": "val_exports"}, inplace=True)
    df_temp["naics"] = df_temp["naics"].astype(str)
    df_temp["naics3"] = df_temp["naics"].str[:3]
    df_temp = (
        df_temp
        .groupby(["year", "naics3", "cty_code"], as_index=False)["val_exports"]
        .sum()
    )
    # append to df
    df = pd.concat([df, df_temp], ignore_index=True)

#rename cty_code isonumber
df.rename(columns={"cty_code": "isonumber"}, inplace=True)
# merge m:1 isonumber using temp_files\isocodes.dta
df["isonumber"] = df["isonumber"].astype(int).astype(str)
df["isonumber"] = df["isonumber"].str.strip()
df = pd.merge(df, iso_df, on="isonumber", how="left")
# keep year naics3 val_exports wbcode
df = df[["year", "naics3", "val_exports", "wbcode"]]

post_1988_exports = df.copy()

#append pre_1988_exports to post_1988_exports
exports = pd.concat([pre_1988_exports, post_1988_exports], ignore_index=True)

# rename wbcode country_abb, rename naics3 naics, rename val_exports exports
exports.rename(columns={"wbcode": "country_abb", "naics3": "naics", "val_exports": "exports"}, inplace=True)
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
#now we have export shares we need 
#check exp_share sums to 1 within naics and year
exports_check = exports.groupby(["naics", "year"])["exp_share"].sum().reset_index()
exports_check = exports_check[exports_check["exp_share"] != 1]
print(exports_check) # everything = 1 up to rounding error




# %%  BEA I/O Tables ----------------------------------------
leg_indgdp_path = os.path.join(data_folder, "raw", "bea", "GDPbyInd_II_1947-1997.xlsx")
leg_usetables_path = os.path.join(data_folder, "raw", "bea", "use_tables_hist.xlsx")

price_path = os.path.join(data_folder, "raw", "bea", "II_price_97_23.xlsx") 
qty_path = os.path.join(data_folder, "raw", "bea", "II_qty_97_23.xlsx")
usetables_path = os.path.join(data_folder, "raw", "bea", "IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx")

#%% Use tables -> four Shea vars to build J^Shea
# for each year 1963 to 2023, just read in the appropriate sheet of use tables
import sys, os, numpy as np, pandas as pd

hist_xls = os.path.join(data_folder, "raw", "bea", "use_tables_hist.xlsx")
post_xls = os.path.join(
    data_folder, "raw", "bea",
    "IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx"
)

out_dir  = os.path.join(data_folder, "temp_files")
os.makedirs(out_dir, exist_ok=True)

# ------------- 1968-2023 -------------
for year in range(1968, 2024):
    sheet = str(year)

    if year <= 1996:
        # read header row and IO block for historical file. fill NAs with 0 as Boehm (2022).
        names = pd.read_excel(hist_xls, sheet_name=sheet, header=None, usecols="C:BO", skiprows=5, nrows=1, engine="openpyxl", na_values="...").to_numpy().flatten()
        IO = pd.read_excel(hist_xls, sheet_name=sheet, header=None, usecols="C:BO", skiprows=7, nrows=65, engine="openpyxl", na_values="...").fillna(0).to_numpy()
        FD = pd.read_excel(hist_xls, sheet_name=sheet, header=None, usecols="CG", skiprows=7, nrows=65, engine="openpyxl", na_values="...").to_numpy().flatten()
        VA = pd.read_excel(hist_xls, sheet_name=sheet, header=None, usecols="C:BO", skiprows=75, nrows=1, engine="openpyxl", na_values="...").to_numpy().flatten()
        EXP = pd.read_excel(hist_xls, sheet_name=sheet, header=None, usecols="BV", skiprows=7, nrows=65, engine="openpyxl", na_values="...").fillna(0).to_numpy() 
        #GFG scaling for historical data only. Starts as 75 rows and 86 cols but we just need a scalar, the value of one cell / another. This is start of a01_construct_sample.do 
        GFGSCL = pd.read_excel(hist_xls, sheet_name=sheet, header=0, usecols="A:CH", skiprows=6, nrows=70, engine="openpyxl", na_values="...").fillna(0)
        # make GFGSCL the value of (row: GFG, column National Defense: Consumption expenditures) / (row: GFG, column Total Commodity Output)
        GFGSCL = GFGSCL["National defense: Consumption expenditures"][GFGSCL["IOCode"] == "GFG"] / GFGSCL["Total Commodity Output"][GFGSCL["IOCode"] == "GFG"]

    if year >= 1997:
        # read header row and IO block for post-1997 file. fill NAs with 0 as Boehm (2022).
        names = pd.read_excel(post_xls, sheet_name=sheet, header=None, usecols="C:BU", skiprows=5, nrows=1, engine="openpyxl", na_values="...").to_numpy().flatten()
        IO = pd.read_excel(post_xls, sheet_name=sheet, header=None, usecols="C:BU", skiprows=7, nrows=71, engine="openpyxl", na_values="...").fillna(0).to_numpy()
        FD = pd.read_excel(post_xls, sheet_name=sheet, header=None, usecols="CQ", skiprows=7, nrows=71, engine="openpyxl", na_values="...").to_numpy().flatten()
        VA = pd.read_excel(post_xls, sheet_name=sheet, header=None, usecols="C:BU", skiprows=84, nrows=1, engine="openpyxl", na_values="...").to_numpy().flatten()
        EXP = pd.read_excel(post_xls, sheet_name=sheet, header=None, usecols="CC", skiprows=7, nrows=71, engine="openpyxl", na_values="...").fillna(0).to_numpy() 
    # clean and compute totals
    IO[np.isnan(IO)] = 0
    total = IO.sum(axis=1) + FD
    II    = len(total)

    # demand-side shares ----------
    C_p = IO / total                        # cost shares -- this is Gamma^c' in Boehm 2022. 
    tmp = np.outer(1/total, FD)             # outer product of column vectors 1/py and FD
    UDS = np.linalg.inv(np.eye(II) - C_p) * tmp # hademard 
    DDS = IO / (total - FD)[:, None]
    #write DDS and UDS to a combined csv file 
    DDS_df = pd.DataFrame(DDS, columns=names, index=names) 
    UDS_df = pd.DataFrame(UDS, columns=names, index=names)
    
    #reshape 
    DDS_df = DDS_df.stack().reset_index()
    DDS_df.columns = ["naics_source","naics_dest","DDS"]

    UDS_df = UDS_df.stack().reset_index()
    UDS_df.columns = ["naics_source", "naics_dest", "UDS"]

    #merge and save
    combined_df = pd.merge(DDS_df, UDS_df, on=["naics_source", "naics_dest"])
    combined_df["year"] = year
    combined_df.to_csv(os.path.join(out_dir, f"demand_shares_{year}.csv"), index=False)

    # cost-side shares ----------
    total = IO.sum(axis=0) + VA
    C_p   = IO.T / total #Gamma^s in Boehm 2022. 
    tmp   = np.outer(1/total, VA) # outer product of column vectors 1/py and VA
    UCS   = (np.linalg.inv(np.eye(II) - C_p) * tmp).T #hademard. we transpose because source and dest are flipped vs above 
    DCS   = (IO / (total - VA)) 
    #write DCS and UCS to a combined csv file 
    DCS_df = pd.DataFrame(DCS, columns=names, index=names)
    UCS_df = pd.DataFrame(UCS, columns=names, index=names)
    # Sji_df = pd.DataFrame(C_p, columns=names, index=names) #i's sales shares to j
    IOT_df = pd.DataFrame(IO.T, columns=names, index=names) #later, will use in i's sales shares to j
    totalsq = np.outer(total, np.ones(II))
    total_df = pd.DataFrame(totalsq, columns=names, index=names) #later, will use in i's sales shares to j
    #exports is single col (naics_source)
    EXP_df = pd.DataFrame(EXP, index=names)
    EXP_df.rename(columns={0: "exp"}, inplace=True)
    EXP_df = EXP_df.reset_index()
    EXP_df.rename(columns={"index": "naics_source"}, inplace=True)

    #reshape 
    DCS_df = DCS_df.stack().reset_index()
    DCS_df.columns = ["naics_source","naics_dest","DCS"] 

    UCS_df = UCS_df.stack().reset_index()
    UCS_df.columns = ["naics_source", "naics_dest", "UCS"]

    IOT_df = IOT_df.stack().reset_index()
    IOT_df.columns = ["naics_dest", "naics_source", "IOT"] #flip indices to get Sji, i's sales shares to j, later
    IOT_df["IOT"][IOT_df["naics_source"]=="GFG"] = IOT_df["IOT"][IOT_df["naics_source"]=="GFG"] * GFGSCL  #rescale by GFGSCL    

    total_df = total_df.stack().reset_index()
    total_df.columns = ["naics_dest", "naics_source", "total"] #flip indices to get Sji, i's sales shares to j, later

    combined_df = pd.merge(DCS_df, UCS_df, on=["naics_source", "naics_dest"])
    combined_df = pd.merge(combined_df, IOT_df, on=["naics_source", "naics_dest"])
    combined_df = pd.merge(combined_df, total_df, on=["naics_source", "naics_dest"])
    combined_df = pd.merge(combined_df, EXP_df, on=["naics_source"])
    combined_df.to_csv(os.path.join(out_dir, f"cost_shares_{year}.csv"), index=False)

print("Finished. CSVs are in:", out_dir)



# %% Compare sheets of use_tables_BOEHM.xlsx and "IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx" year by year 

# (1) A plot of the average absolute percent difference by year
# (2) A plot of the standard deviation of the percent difference by year
# (3) A histogram of the percent difference (aggregate + by year)

import sys, os, numpy as np, pandas as pd

hist_xls = os.path.join(data_folder, "raw", "bea", "use_tables_hist.xlsx")
post_xls = os.path.join(
    data_folder, "raw", "bea",
    "IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx"
)
post_xls_boehm = os.path.join(
    data_folder, "raw", "bea",
    "use_tables_BOEHM.xlsx"
)

out_dir  = os.path.join(data_folder, "temp_files")
os.makedirs(out_dir, exist_ok=True)

# ------------- 1997-2016 -------------
graphdf = []
#empty df IOdiff_long to hold all the IO differences
IOdiff_long = pd.DataFrame()

for year in range(1997, 2017):
    sheet = str(year)

    # read header row and IO block for historical file
    IO_me = pd.read_excel(post_xls, sheet_name=sheet, header=None, usecols="C:BO", skiprows=7, nrows=71, engine="openpyxl", na_values="...").to_numpy()
    IO_boehm = pd.read_excel(post_xls_boehm, sheet_name=sheet, header=None, usecols="C:BO", skiprows=7, nrows=71, engine="openpyxl", na_values="...").to_numpy()

    # IO_me - IO_boehm 
    IO_diff = (IO_me - IO_boehm) / IO_boehm * 100
    # take the mean of the absolute value of the difference across all rows and columns, ignorning NaN values
    #replace inf with NaN
    IO_diff[np.isinf(IO_diff)] = np.nan
    mean_diff = np.nanmean(np.abs(IO_diff))
    #median of the absolute value of the difference across all rows and columns, ignorning NaN values
    median_diff = np.nanmedian(np.abs(IO_diff))

    #std of the absolute value of the difference across all rows and columns, ignorning NaN values
    std_diff = np.nanstd(np.abs(IO_diff))

    #fill graphdf with year, mean_diff, median_diff, std_diff
    graphdf.append([year, mean_diff, median_diff, std_diff])

    #convert IO_diff to a dataframe
    IO_diff = pd.DataFrame(IO_diff)
    IO_diff["year"] = year
    #append IO_diff to empty dataframe IOdiff_long
    IOdiff_long = pd.concat([IOdiff_long, IO_diff], ignore_index=True)
    
graphdf = pd.DataFrame(graphdf, columns=["year", "mean_diff", "median_diff", "std_diff"])

# (1) A plot of the average absolute percent difference by year
# (2) A plot of the standard deviation of the percent difference by year
import matplotlib.pyplot as plt
import seaborn as sns   

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6)) 
plt.plot(graphdf["year"], graphdf["mean_diff"], label="Mean Absolute Percent Difference", color="blue")
plt.plot(graphdf["year"], graphdf["median_diff"], label="Median Absolute Percent Difference", color="orange")   

plt.plot(graphdf["year"], graphdf["std_diff"], label="Standard Deviation of Absolute Percent Difference", color="green")
plt.title("Average Absolute Percent Difference by Year")
plt.xlabel("Year")
plt.ylabel("Average Absolute Percent Difference")
plt.legend()
plt.xticks(graphdf["year"], rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "avg_abs_percent_diff.png"), dpi=300)

#now plotting just the mean_diff and median_diff
plt.figure(figsize=(10, 6)) 
plt.plot(graphdf["year"], graphdf["mean_diff"], label="Mean Absolute Percent Difference", color="blue")
plt.plot(graphdf["year"], graphdf["median_diff"], label="Median Absolute Percent Difference", color="orange")

plt.title("Average Absolute Percent Difference by Year")
plt.xlabel("Year")
plt.ylabel("Average Absolute Percent Difference")
plt.legend()
plt.xticks(graphdf["year"], rotation=45)
plt.tight_layout()
#make y-axis start at 0     
plt.ylim(0, 150)
plt.savefig(os.path.join(out_dir, "avg_abs_percent_diff_mean_median.png"), dpi=300)


# (3) A histogram of the percent difference (aggregate + by year)
IOdiff_long = IOdiff_long.drop(columns=["year"])
IOdiff_long = IOdiff_long.melt(var_name="naics", value_name="IO_diff") #note: this label not =naics
IOdiff_long = IOdiff_long[IOdiff_long["IO_diff"].notnull()]
IOdiff_long = IOdiff_long[IOdiff_long["IO_diff"] <= 1000] # there are several outliers
IOdiff_long["IO_diff"] = np.abs(IOdiff_long["IO_diff"])
#plot histogram of IO_diff

plt.figure(figsize=(10, 6))
plt.hist(IOdiff_long["IO_diff"], bins=50, color="blue", alpha=0.7)  

plt.title("Histogram of Absolute Percent Difference")
plt.xlabel("Absolute Percent Difference")

plt.ylabel("Frequency")
plt.xticks(rotation=45)

plt.tight_layout()
#save to out_dir    
plt.savefig(os.path.join(out_dir, "abs_percent_diff_histogram.png"), dpi=300)





# %% BEA Intermediates input use time series -----------------------------------
# (Chain Qty Index, "M" in Boehm 2022 Shea instrument)

IOcodes = pd.read_excel(
    os.path.join(data_folder, "raw", "bea", "IO_codes.xlsx"),
    sheet_name="Sheet1",
    header=5,
    usecols="A:B",
    nrows=71,
    engine="openpyxl",
) 
IOcodes["Description"] = IOcodes["Description"].str.strip()

IOcodes_hist = pd.read_excel(
    os.path.join(data_folder, "raw", "bea", "IO_codes_aggregated.xlsx"),
    sheet_name="Sheet1",
    header=5,
    usecols="A:B",
    nrows=65,
    engine="openpyxl",
)
IOcodes_hist["Description"] = IOcodes_hist["Description"].str.strip()

#1997-2024 quantities index ----------
quantities = pd.read_excel(
    os.path.join(data_folder, "raw", "bea", "II_qty_97_24.xlsx"),
    sheet_name="Sheet1",
    skiprows=7,
    header=0,
    usecols="A:AE",
    nrows=99,
    engine="openpyxl",
)
quantities = quantities.drop(columns=["Line", "Unnamed: 2"])
quantities.rename(columns={"Unnamed: 1": "Description"}, inplace=True)

quantities["Description"] = quantities["Description"].str.strip()
#check description column for duplicates
duplicates_hist = quantities[quantities.duplicated(subset=["Description"], keep=False)]

quantities = clean_GDP_by_ind(quantities)

quantities = pd.merge(quantities, IOcodes, on="Description", how="inner")
quantities.drop(columns=["Description"], inplace=True)
quantities["naics_code"] = quantities["IO_Code"]

# condense naics codes
mappings = [
    ("11",   ["111CA", "113FF"]),                     # Agriculture, forestry, and fishing
    ("336",  ["3361MV", "3364OT"]),                   # Transportation equipment
    ("311,2",["311FT"]),                              # Food/beverages
    ("313,4",["313TT"]),                              # Textiles
    ("315,6",["315AL"]),                              # Apparel/leather
    ("44",   ["441", "445", "452", "4A0"]),            # Retail
    ("48",   ["481","482","483","484","485","486","487OS","493"]),  # Transp & warehousing
    ("52",   ["521CI","523","524","525"]),             # Finance
    ("53",   ["HS","ORE","532RL"]),                   # Real estate & rental/leasing
    ("54",   ["5411","5415","5412OP"]),                # Legal/professional services
    ("71",   ["711AS","713"]),                        # Performing arts/entertainment
]

for new_code, patterns in mappings:
    regex = "|".join(patterns)            # e.g. "111CA|113FF"
    mask = quantities["naics_code"].str.contains(regex, na=False)
    quantities.loc[mask, "naics_code"] = new_code

quantities = quantities.replace("...", 0)

#reshape long to time series
quantities = quantities.melt(id_vars=["naics_code", "IO_Code"], var_name="year", value_name="qty_index")


#1947-1997 quantities index ----------
quantities_hist = pd.read_excel(
    os.path.join(data_folder, "raw", "bea", "GDPbyInd_II_1947-1997.xlsx"),
    sheet_name="ChainQtyIndexes",
    skiprows=5,
    header=0,
    usecols="A:BA",
    nrows=102,
    engine="openpyxl",
)

quantities_hist = quantities_hist.drop(columns=["Line"])
quantities_hist.rename(columns={"Unnamed: 1": "Description"}, inplace=True)

quantities_hist["Description"] = quantities_hist["Description"].str.strip()
#check description column for duplicates
duplicates_hist = quantities_hist[quantities_hist.duplicated(subset=["Description"], keep=False)]

quantities_hist = clean_GDP_by_ind(quantities_hist)

quantities_hist = pd.merge(quantities_hist, IOcodes, on="Description", how="inner")
quantities_hist.drop(columns=["Description"], inplace=True)
quantities_hist["naics_code"] = quantities_hist["IO_Code"]

#condense naics codes
mappings = [
    ("11",   ["111CA", "113FF"]),                     # Agriculture, forestry, and fishing
    ("336",  ["3361MV", "3364OT"]),                   # Transportation equipment
    ("311,2",["311FT"]),                              # Food/beverages/clothing/textiles
    ("313,4",["313TT"]),                              # Textiles
    ("315,6",["315AL"]),                              # Apparel/leather
    ("44",   ["441", "445", "452", "4A0"]),            # Retail
    ("48",   ["481","482","483","484","485","486","487OS","493"]),  # Transp & warehousing
    ("52",   ["521CI","523","524","525"]),             # Finance
    ("53",   ["HS","ORE","532RL"]),                   # Real estate & rental/leasing
    ("54",   ["5411","5415","5412OP"]),                # Legal/professional services
    ("71",   ["711AS","713"]),                        # Performing arts/entertainment
]

for new_code, patterns in mappings:
    regex = "|".join(patterns)                        # e.g. "111CA|113FF"
    mask = quantities_hist["naics_code"].str.contains(regex, na=False)
    quantities_hist.loc[mask, "naics_code"] = new_code

quantities_hist = quantities_hist.replace("...", 0)

#reshape long to time series
quantities_hist = quantities_hist.melt(id_vars=["naics_code", "IO_Code"], var_name="year", value_name="qty_index")

#append quantities to non-1997-quantities_hist 
quantities_hist = quantities_hist[quantities_hist["year"] != "1997"].copy()
quantities_full = pd.concat([quantities_hist, quantities], ignore_index=True)
quantities_full = quantities_full.rename(columns={"naics_code": "naics_dest"})



# %% Compare the quantities in our time series data with Boehm's 
# use "processed data" since it just takes the raw values; easier to merge 

#make quant_BOEHM and quant_sam, two series indexed to same year (2017)
#read stata dataset intermediate_ts_BOEHM.dta
intermediate_ts_path = os.path.join(data_folder, "raw", "bea", "intermediate_ts_BOEHM.dta")
intermediate_ts = pd.read_stata(intermediate_ts_path, convert_categoricals=False)

# save new dataframe with only naics_code year quantity_index and only rows year == 2017
intermediate_ts_ind = intermediate_ts[["IO_Code", "year", "quantity_index"]]
intermediate_ts_ind = intermediate_ts_ind[intermediate_ts["year"] == 2017]
#drop year column
intermediate_ts_ind.drop(columns=["year"], inplace=True)
intermediate_ts_ind.rename(columns={"quantity_index": "qty_2017"}, inplace=True)

#merge intermediate_ts_ind with intermediate_ts on naics_code
intermediate_ts = pd.merge(intermediate_ts, intermediate_ts_ind, on=["IO_Code"], how="left")
intermediate_ts["qty_index_BOEHM"] = intermediate_ts["quantity_index"] / intermediate_ts["qty_2017"] * 100

#quantites_sam  = quantities, rows where year <= 2017
quantities["year"] = quantities["year"].astype(int)
quantities_sam = quantities[quantities["year"] <= 2017].copy()

quantities_sam = pd.merge(intermediate_ts, quantities_sam, on=["IO_Code", "year"], how="inner")

quantities_sam["BOEHM_min_sam"] = (quantities_sam["qty_index_BOEHM"] - quantities_sam["qty_index"]) / quantities_sam["qty_index"] * 100

#get time series by year: average absolute BOEHM_min_sam by year, SD of BOEHM_min_sam by year
quantities_sam["year"] = quantities_sam["year"].astype(int)
quantities_sam["BOEHM_min_sam_abs"] = np.abs(quantities_sam["BOEHM_min_sam"].astype(float))
# collapse by year
quantities_sam = (
    quantities_sam
    .groupby("year", as_index=False)[["BOEHM_min_sam", "BOEHM_min_sam_abs"]]
    .agg(["mean", "median", "std"])
)

#plot BOEHM_min_sam mean, median, std by year
plt.figure(figsize=(10, 6))
plt.plot(quantities_sam["year"], quantities_sam["BOEHM_min_sam"]["mean"], label="Mean Percent Difference", color="blue")
plt.plot(quantities_sam["year"], quantities_sam["BOEHM_min_sam"]["median"], label="Median Percent Difference", color="orange")
plt.plot(quantities_sam["year"], quantities_sam["BOEHM_min_sam"]["std"], label="Standard Deviation of Percent Difference", color="green")
plt.title("Percent Difference by Year")
plt.xlabel("Year")
plt.ylabel("Percent Difference")
plt.legend()
plt.xticks(quantities_sam["year"], rotation=45) 
plt.tight_layout()
#save to out_dir
plt.savefig(os.path.join(out_dir, "avg_percent_diff_quantities_QbyInd.png"), dpi=300)

#plot BOEHM_min_sam_abs mean, median, std by year
plt.figure(figsize=(10, 6))
plt.plot(quantities_sam["year"], quantities_sam["BOEHM_min_sam_abs"]["mean"], label="Mean Absolute Percent Difference", color="blue")
plt.plot(quantities_sam["year"], quantities_sam["BOEHM_min_sam_abs"]["median"], label="Median Absolute Percent Difference", color="orange")
plt.plot(quantities_sam["year"], quantities_sam["BOEHM_min_sam_abs"]["std"], label="Standard Deviation of Absolute Percent Difference", color="green")
plt.title("Abs. Percent Difference by Year")
plt.xlabel("Year")
plt.ylabel("Abs. Percent Difference")
plt.legend()
plt.xticks(quantities_sam["year"], rotation=45)
plt.tight_layout()
#save to out_dir    
plt.savefig(os.path.join(out_dir, "avg_abs_percent_diff_QbyInd.png"), dpi=300)




# %% Building the actual Shea instrument 
shea_threshold = 3

#Cost Shares 
#load in all the cost shares and append them. 
cost_shares = pd.DataFrame()
for year in range(1968, 2024):
    cost_shares_temp = pd.read_csv(os.path.join(out_dir, f"cost_shares_{year}.csv"))
    cost_shares_temp["year"] = year
    cost_shares = pd.concat([cost_shares, cost_shares_temp], ignore_index=True)

#Demand Shares
#load in all the demand shares and append them.
demand_shares = pd.DataFrame()
for year in range(1968, 2024):
    demand_shares_temp = pd.read_csv(os.path.join(out_dir, f"demand_shares_{year}.csv"))
    demand_shares_temp["year"] = year
    demand_shares = pd.concat([demand_shares, demand_shares_temp], ignore_index=True)

#the four stats we need are: dcs_ij, ucs_ij, dds_ji, uds_ji 
#then take for each industry i: J^Shea_{it} = j: min{dds_{jit}, uds_{jit}} / max{ucs_{ijt}, dcs_{ijt}, ucs_{jit}, dcs_{jit}} > 3 
shea_full = pd.merge(cost_shares, demand_shares, on=["naics_source", "naics_dest", "year"], how='left')

#the below processing ignores Sji; this follows Boehm
shareslist = ["DCS", "UCS", "DDS", "UDS"]
for share in shareslist:
    shea_full[share][shea_full[share].isnull()] = 0  

#A) we have dds_ji, uds_ji, dcs_ij, ucs_ij; we need dcs_ji, ucs_ji
shea_C_ji = shea_full.copy()
shea_C_ji.rename(columns={"naics_source": "naics_dest", "naics_dest": "naics_source"}, inplace=True)
shea_C_ji.drop(columns=["DDS", "UDS", "IOT", "total", "exp"], inplace=True)
shea_C_ji.rename(columns={"DCS": "DCS_ji", "UCS": "UCS_ji"}, inplace=True)
#merge the two dataframes on naics_source and naics_dest and year
shea_full = pd.merge(shea_full, shea_C_ji, on=["naics_source", "naics_dest", "year"], how='left')

# keep if substr(naics_source,1,1) == "3"
shea_full = shea_full[shea_full["naics_source"].str.startswith("3")]

#stat2 = min(dds,uds)/max(dcs,ucs,dcs_sfd,ucs_sfd). Boehm et al also construct simpler "original Shea" ver, but we care about this one
shea_full["stat2"] = np.minimum(shea_full["DDS"], shea_full["UDS"]) / np.maximum.reduce([shea_full["DCS"], shea_full["UCS"], shea_full["DCS_ji"], shea_full["UCS_ji"]])

#replace as 0 if any of the values are negative.  
shea_full["stat2"] = np.where(
    (shea_full["stat2"] < 0) | (shea_full["DDS"] < 0) | (shea_full["UDS"] < 0) | (shea_full["DCS"] < 0) | (shea_full["UCS"] < 0) | (shea_full["DCS_ji"] < 0) | (shea_full["UCS_ji"] < 0), 0, shea_full["stat2"]
)

shea_full["LJShea"] = (shea_full["stat2"] > shea_threshold).astype(int) #Defining JShea

#replace year = year + 1 # so now we have JShea_{it-1} 
shea_full["year"] = shea_full["year"].astype(int) + 1

#B) fix NAICS codes ala "M" construction, line 608
#condense naics codes
mappings = [
    ("11",   ["111CA", "113FF"]),                     # Agriculture, forestry, and fishing
    ("336",  ["3361MV", "3364OT"]),                   # Transportation equipment
    ("311,2",["311FT"]),                              # Food/beverages/clothing/textiles
    ("313,4",["313TT"]),                              # Textiles
    ("315,6",["315AL"]),                              # Apparel/leather
    ("44",   ["441", "445", "452", "4A0"]),            # Retail
    ("48",   ["481","482","483","484","485","486","487OS","493"]),  # Transp & warehousing
    ("52",   ["521CI","523","524","525"]),             # Finance
    ("53",   ["HS","ORE","532RL"]),                   # Real estate & rental/leasing
    ("54",   ["5411","5415","5412OP"]),                # Legal/professional services
    ("71",   ["711AS","713"]),                        # Performing arts/entertainment
]

for new_code, patterns in mappings:
    regex = "|".join(patterns)                        # e.g. "111CA|113FF"
    mask = shea_full["naics_source"].str.contains(regex, na=False)
    shea_full.loc[mask, "naics_source"] = new_code


#C) collapse min within each naics_code, naics_dest, and year; S_ji comes back # 72k -> 68k
shea_full = (
    shea_full
    .groupby(["naics_source", "naics_dest", "year"], as_index=False)
    .agg({"LJShea": "min", "IOT": "sum", "total": "sum", "exp": "sum"})
)

#D) split "311,2" "313,4", and "315,6" into separate rows: 311, 312, 313, 314, 315, 316, copying all other columns
split_mapping = {
    "311,2": ["311", "312"],
    "313,4": ["313", "314"],
    "315,6": ["315", "316"]
}

'''
The below code:
1.Defines a split_mapping dictionary for the naics_source codes you want to split and their corresponding new codes.
2.Creates a temporary column naics_source_temp. For each row:
-If naics_source is one of the keys in split_mapping (e.g., "311,2"), naics_source_temp gets the list of new codes (e.g., ["311", "312"]).
-Otherwise, naics_source_temp gets the original naics_source value wrapped in a list (e.g., ["some_other_code"]) to ensure explode works correctly for all rows.
3.shea_full.explode("naics_source_temp") then transforms each row with a list in naics_source_temp into multiple rows, one for each item in the list, duplicating all other column values.
4.The original naics_source column is updated with the new, split values from naics_source_temp.
'''
# Apply the mapping: if a naics_source is in split_mapping, it becomes a list. Otherwise, wrap it in a list.
shea_full["naics_source_temp"] = shea_full["naics_source"].apply(lambda x: split_mapping.get(x, [x]))
# Explode the DataFrame on the temporary column
shea_full = shea_full.explode("naics_source_temp")
# Assign the exploded values back to 'naics_source' and drop the temporary column
shea_full["naics_source"] = shea_full["naics_source_temp"]
shea_full.drop(columns=["naics_source_temp"], inplace=True)

shea_full["sales_sh"] = shea_full["IOT"] / shea_full["total"] # sales shares sj,i,t of industry i to buyer j; adj as in a01_construct_sample.do
shea_full["exp_sh"] = shea_full["exp"] / shea_full["total"] # export shares sj,i,t of industry i to buyer j
#Now we have Indic{i \in JShea_{i,t-1}}, sales_sh s_{j,i,t-1}, exp_sh s_{j,i,t-1} (used in the WID and exchange rate instruments)

# collapse not necssary given differences in construction (I built without final use categories)
shea_full.rename(columns={"sales_sh": "Lsales_sh"}, inplace=True) # since we alr did year = year + 1
shea_full.rename(columns={"exp_sh": "Lexp_sh"}, inplace=True) 

#CHECK: count unique levels of naics_source in shea_full
unique_naics_source = shea_full["naics_source"].nunique()
print("Unique naics_source in shea_full:", unique_naics_source) #21, good
# In replication procedure, we are at end of a04_construct_sample.do and did some scaling at top of a01_construct_sample.do -----
# Make Boehm's Dln_M_shea_inst. shea full has sales shares, export shares, and JShea; quantities full has M 

# start with quantities_full: drop everything where naics_dest first char is G 
quantities_forshea = quantities_full[quantities_full["naics_dest"].str.startswith("3")]
#drop naics dest
quantities_forshea = quantities_forshea.drop(columns=["naics_dest"])
#rename IO_code to naics_dest
quantities_forshea = quantities_forshea.rename(columns={"IO_Code": "naics_dest"})
#drop if year <= 1963
quantities_forshea["year"] = quantities_forshea["year"].astype(int)
quantities_forshea = quantities_forshea[quantities_forshea["year"] > 1963]
#by naics_dest: generate percent change in qty_index
quantities_forshea["Dln_M"] = quantities_forshea.groupby("naics_dest")["qty_index"].pct_change() * 100

#In Boehm 2022 code: Only 19 "destination" industries for their Shea inst, but 21 sources. 
#I think this is just imputing the "quantities" for those industries; assumption is that changes in M for those industries are equal.
shea_calc = pd.merge(shea_full, quantities_forshea, on=["naics_dest", "year"], how="inner")

shea_calc["Dln_M_shea_inst2"] = shea_calc["Lsales_sh"] * shea_calc["Dln_M"] * shea_calc["LJShea"] 
#collapse (sum) by naics_source and year
shea_calc["total_by_naicsyear"] = 1
shea_calc = (
    shea_calc
    .groupby(["naics_source", "year"], as_index=False)
    .agg({"Dln_M_shea_inst2": "sum", "LJShea": "sum", "total_by_naicsyear": "sum"})
) #Shea instrument is Dln_M_shea_inst2 in shea_calc

#CHECK: share of partner-year observations that are LJShea for each naics_source
#collapse (sum) by naics_source and year: LJShea and total_by_naicsyear
shJS_by_i = (
    shea_calc
    .groupby(["naics_source"], as_index=False)
    .agg({"LJShea": "sum", "total_by_naicsyear": "sum"})
)
shJS_by_i["LJShea"] = shJS_by_i["LJShea"] / shJS_by_i["total_by_naicsyear"] # share of partner-year observations that are LJShea for each naics_source
#8 with no partner-year obs that are LJShea, compared with 6 in Boehm 2022.


#%% Export instrument 
#first: export shares 

#exports: keep only naics where first character is 3
exports = exports[exports["naics"].str.startswith("3")]
#rename exp_share Lexp_share
exports.rename(columns={"exp_share": "Lexp_share", "naics": "naics_source"}, inplace=True)
#year = year + 1
exports["year"] = exports["year"].astype(int) + 1

exports_calc = pd.merge(exports, shea_full, on=["naics_source", "year"], how="inner")
#drop naics_year
exports_calc = exports_calc.drop(columns=["naics_year"])

# %% Main sample 

#Everything is winsorized at 1% and 99%
#Everything *that enters interaction terms* is demeaned. 




# %%
