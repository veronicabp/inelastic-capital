# %%
from utils import *

# Download data from census of manufactures
BASE_URL = "https://api.census.gov/data/timeseries/asm/industry"
variables = [
    "CEXBLD",
    "CEXMCH",
    "CEXTOT",
    "CSTELEC",
    "CSTMTOT",
    "DPRTOT",
    "EMP",
    "HOURS",
    "INVFINB",
    "INVMATB",
    "INVFINE",
    "PAYANN",
    "PCHEXSO",
    "PCHDAPR",
    "RCPTOT",
    "RPBLD",
    "VALADD",
]

naics = pd.read_csv(f"{data_folder}/raw/naics.csv")
naics["code"] = naics["code"].astype(str)
naics = naics[naics["code"].str[0] == "3"].copy()

naics_digits = 4
naics = naics[naics.code.str.len() == naics_digits].copy()

output = {"naics": [], "year": [], "variable": [], "value": []}
naics_codes = naics.code.unique()
years = range(2002, 2017)
for code, year in tqdm(
    product(naics_codes, years), total=len(naics_codes) * len(years)
):

    params = {
        "get": ",".join(variables),
        "for": "us:*",
        "time": year,
        "NAICS": code,
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        for i, key in enumerate(data[0]):
            if key in variables:
                output["naics"].append(code)
                output["year"].append(year)
                output["variable"].append(key)
                output["value"].append(data[1][i])

    else:
        print(
            f"Error for NAICS {code}, year {year}: {response.status_code} - {response.text}"
        )

df = pd.DataFrame(output)
df.to_csv(os.path.join(data_folder, "raw", f"asm_naics{naics_digits}.csv"), index=False)

# %%
