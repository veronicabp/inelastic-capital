from utils import *


def construct_tract_price(data_folder):
    return


# Construct
def geolytics_price_rent(data_folder):
    folder = os.path.join(
        data_folder,
        "raw",
        "original",
        "baum-snow_2023",
        "sourcedata",
    )
    df = pd.read_stata(
        os.path.join(
            folder,
            "census_acs_19702010",
            "tracts_stf4_ncdb.dta",
        )
    )

    for year in [1970, 1980, 1990, 2000, 2010]:
        df[f"ptr{year}"] = df[f"avhval{year}"] / (12 * df[f"avgrent{year}"])

    for year in [1990, 2000]:
        df[f"ptr_med{year}"] = df[f"medhval{year}"] / (12 * df[f"medgrent{year}"])

    df = df[
        ["tract_str", "pop1990_ncdb"] + [c for c in df.columns if c.startswith("ptr")]
    ].copy()
    df["state"] = df["tract_str"].str[:2]
    df["county"] = df["tract_str"].str[2:5]
    df["tract"] = df["tract_str"].str[5:]

    df = df.rename(columns={"tract_str": "ctracts2000"})
    elas = pd.read_stata(os.path.join(folder, "elasticities", "gammas_hat_all.dta"))
    df = df.merge(elas, on=["ctracts2000"], how="inner")

    return df


def bls_cbsa_rent(data_folder):
    return
