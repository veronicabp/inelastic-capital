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

    file = os.path.join(data_folder, "raw", "wrds", "CRSP_firm_descriptions.csv")
    compustat = pd.read_csv(file)

    # Keep only sectors that produce goods
    compustat["naics2"] = compustat["naics"].astype(str).apply(lambda x: x[:2])
    compustat = compustat[
        ~compustat.naics2.isin(
            ["48", "49", "52", "53", "55", "56", "61", "72", "81", "92"]
        )
    ]

    # Collapse by company
    compustat["n_obs"] = compustat.groupby("LPERMNO")["datadate"].transform("count")
    compustat = compustat[["LPERMNO", "conml", "busdesc", "n_obs"]].drop_duplicates(
        subset=["LPERMNO"]
    )
    compustat = compustat.rename(columns={"LPERMNO": "permno", "conml": "company_name"})
    compustat = compustat.sort_values(by="permno")
    compustat = compustat.reset_index(drop=True)

    # Keep observations with at least four years
    compustat[compustat.n_obs > 4 * 4]

    csv_file = os.path.join(data_folder, "working", "firm_hscodes.csv")
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(
            columns=["permno", "firm_name", "HS4_codes", "model", "description"]
        )

    # model = "o1-preview"
    # model = "o1-mini"
    model = "gpt-4o"

    for i, row in compustat.iterrows():

        firm_name = row.company_name
        firm_desc = row.busdesc
        if firm_name in df.firm_name.unique():
            continue

        prompt = f"You are a researcher matching Harmonized System (HS) codes to US firms. In particular, you are interested in determining the four-digit HS codes associated with the products produced by {firm_name}. Here is a brief description of this firm: {firm_desc}. Based on this description and what you know already about {firm_name}, please explain which codes are most appropriate. At the end of your response, output a list of the codes formatted as a Python list (e.g.) [XXXX, YYYY, ...]. If the firm primarily produces services and does not produce products, output an empty list []. If you do not have sufficient information about this firm, output 'NA' and nothing else."

        print(f"[{i}/{len(compustat)}]: {firm_name}")
        print("=" * len(firm_name))
        # print(prompt, "\n\n")

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
            [[row.permno, firm_name, HS4_code, model, result]],
            columns=["permno", "firm_name", "HS4_codes", "model", "description"],
        )

        df = pd.concat([df, new_row]).copy()
        df.to_csv(csv_file, index=False)
