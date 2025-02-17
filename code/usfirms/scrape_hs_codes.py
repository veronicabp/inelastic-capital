from utils import *


def scrape_hs_chapters():

    # If your HTML is online, use requests to fetch it:
    url = "https://www.usitc.gov/tata/hts/archive/9200/1992_basic_index.htm"
    response = requests.get(url)
    html_content = response.text

    soup = BeautifulSoup(html_content, "html.parser")

    # Find all table rows
    rows = soup.find_all("tr")

    chapter_nums = set()
    chapter_data = []

    for row in rows:
        # Get all <td> elements in this row
        tds = row.find_all("td")

        # We’re specifically looking for rows that have two <td> cells:
        #   1) The first cell (with colspan="2") has a link whose text starts with "Chapter"
        #   2) The second cell has the description
        if len(tds) == 2:
            link = tds[0].find("a")
            if link and link.text.strip().startswith("Chapter"):
                chapter_number = link.text.strip()  # e.g. "Chapter 1"
                description = tds[1].text.strip()  # e.g. "Live animals"

                # Store them in a list (chapter_number, description)
                chapter_number = chapter_number.replace("Chapter ", "")
                if chapter_number not in chapter_nums:
                    chapter_data.append([chapter_number, description])
                    chapter_nums.add(chapter_number)

    # Write the data to a CSV file
    file = os.path.join(data_folder, "working", "HS2.csv")
    df = pd.DataFrame(chapter_data, columns=["HS2", "description"])
    df.to_csv(file, index=False)

    print("Scraping complete. Data written to chapters.csv")
