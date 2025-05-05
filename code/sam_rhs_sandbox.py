# sam sandbox for inelastic capital RHS variable creation 
# conda environment: inelastic_capital


# what I needed to pip, conda install that I didn't have 
# ipykernel (vscode will do automatically)
# notebook -- this was a pain, should manually install in conda env
# pip install linearmodels # or conda install? 
# conda install matplotlib 
# pip install pyfixest 
# conda install scipy 
# conda install tqdm
# pip install stargazer 
# conda install requests 


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

#%% Get GDP, Exchange rate, and HS-NAICS crosswalk data
import utils
from utils import *

reload(utils)

folder = "C:/Users/illge/Princeton Dropbox/Sam Barnett/inelastic_capital"
data_folder = os.path.join("..", "data")

gdp_data = load_gdp_data(data_folder)

exchange_rate_data = load_exchange_rate_data(data_folder)   
#USD / LCU
#can multiply a value in local currency by exchange_rate to convert it into USD.

naics_hs_crosswalk = load_naics_hs_crosswalk(data_folder)

# %% 
