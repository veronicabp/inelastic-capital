# %%
from utils import *


def run_gpt(model="gpt-4o", context_prompt="", prompt="", get_full_response=False):
    api_key = os.environ["MY_CHATGPT_API_KEY"]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    messages = []

    # Add context prompt if it exists
    if len(context_prompt) > 0:
        message = {
            "role": "developer",
            "content": [{"type": "text", "text": context_prompt}],
        }
        messages.append(message)

    # Add main prompt
    message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }
    messages.append(message)

    # print(f"{len(messages)} Messages:", messages)

    payload = {
        "model": model,
        "messages": messages,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    # print(response.json())

    if "choices" not in response.json():
        print(response.json())
        return "ERROR"

    if get_full_response:
        return response

    else:
        return response.json()["choices"][0]["message"]["content"]


def get_prompt(firm_name, df, HS_num):
    # main_prompt = f"You are studying the firm '{firm_name}'. Please output a list of {HS_num}-digit Harmonized System codes associated with the primary products that the firm '{firm_name}' produces, formatted as a Python list (e.g. [{'X'*HS_num}, {'Y'*HS_num}, ...]). If you do not know what this firm specializes in, or the firm is in the service industry and does not produce products, output an empty Python list, []. Do not output anything else."
    main_prompt = f"You are studying the firm '{firm_name}'. Please output a list of {HS_num}-digit Harmonized System codes from the below list associated with the primary products that the firm '{firm_name}' produces, formatted as a Python list (e.g. [{'X'*HS_num}, {'Y'*HS_num}, ...]). If none apply, output an empty list, []. Do not output anything else."

    context_prompt = f"The Harmonized System {HS_num}-digit product codes and descriptions of interest are:\n"
    for i, row in df.iterrows():
        context_prompt += f"{row.code}: {row.description}\n"
    return main_prompt + "\n" + context_prompt


def match_firms_products():

    compustat_file = os.path.join(
        data_folder, "raw", "wrds", "CCM_fundamentals_all.dta"
    )
    compustat = pd.read_stata(compustat_file)

    # Keep only sectors that produce goods
    compustat["naics2"] = compustat["naics"].apply(lambda x: x[:2])
    compustat = compustat[
        ~compustat.naics2.isin(
            ["48", "49", "52", "53", "55", "56", "61", "72", "81", "92"]
        )
    ]

    # Keep sample years
    compustat = compustat[(compustat.fyearq >= 2000) & (compustat.fyearq <= 2007)]

    # Collapse by company
    compustat["n_obs"] = compustat.groupby("GVKEY")["datadate"].transform("count")
    compustat = compustat[["GVKEY", "conml", "n_obs"]].drop_duplicates(subset=["GVKEY"])
    compustat = compustat.rename(columns={"GVKEY": "gvkey", "conml": "company_name"})
    compustat = compustat.sort_values(by="gvkey")

    # Keep observations with at least four years
    compustat[compustat.n_obs > 4 * 4]

    csv_file = os.path.join(data_folder, "working", "firm_hscodes.csv")
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(
            columns=["gvkey", "firm_name", "HS4_codes", "model", "description"]
        )

    # model = "o1-preview"
    model = "o1-mini"
    for i, row in compustat.iterrows():

        firm_name = row.company_name
        if firm_name in df.firm_name.unique():
            continue

        prompt = f"You are a researcher matching Harmonized System (HS) codes to US firms. In particular, you are interested in determining the four-digit HS codes associated with the products produced by {firm_name}. Please provide a description of which codes are most appropriate. At the end of your response, output a list of the codes formatted as a Python list (e.g.) [XXXX, YYYY, ...]. If the firm only produces services and does not produce products, output an empty list []. If you do not have any information about this firm, output 'NA' and nothing else."

        print(firm_name)
        print("=" * len(firm_name))

        result = run_gpt(model=model, prompt=prompt).replace("```", "")

        if result == "ERROR":
            break

        s = re.search(r"\[.+\]$", result)

        print(result)
        print("\n\n")

        if s:
            HS4_code = s.group()

        else:
            HS4_code = "[]"

        new_row = pd.DataFrame(
            [[row.gvkey, firm_name, HS4_code, model, result]],
            columns=["gvkey", "firm_name", "HS4_codes", "model", "description"],
        )

        df = pd.concat([df, new_row]).copy()
        df.to_csv(csv_file, index=False)

    # # Load product code descriptions
    # product_codes = pd.read_csv(
    #     os.path.join(data_folder, "raw", "baci", "product_codes_HS92_V202301.csv")
    # )

    # product_codes["HS2"] = product_codes["code"].apply(lambda x: x[:2])
    # product_codes["HS4"] = product_codes["code"].apply(lambda x: x[:4])

    # product_codes["HS4_description"] = product_codes["description"].apply(
    #     lambda x: x.split(":")[0]
    # )

    # HS4 = (
    #     product_codes.groupby(["HS4", "HS2"])["HS4_description"]
    #     .apply(
    #         lambda x: ", ".join(pd.unique(x))
    #     )  # join desc values with comma and space
    #     .reset_index()
    # ).rename(columns={"HS4_description": "description", "HS4": "code"})

    # HS2 = pd.read_csv(os.path.join(data_folder, "working", "HS2.csv")).rename(
    #     columns={"HS2": "code"}
    # )

    # # First, identify the firm's 2 digit code

    # firm_name = "CF & I Steel Corp"
    # prompt = get_prompt(firm_name, HS2, 2)
    # result1 = run_gpt(prompt=prompt)

    # prompt = get_prompt(
    #     firm_name, HS4[HS4.HS2 == "72"].drop_duplicates(subset="description"), 4
    # )
    # print(prompt)
    # result2 = run_gpt(prompt=prompt)


# %%
