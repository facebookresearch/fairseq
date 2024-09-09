import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy
import re
from bs4 import BeautifulSoup

def get_ethnologue_info(iso):
    fl = f"/checkpoint/vineelkpratap/data/ethnologue/{iso}.txt"
    if not os.path.exists(fl):
        return "NULL",  "NULL",  "NULL",  "NULL",  "NULL",  "NULL",  "NULL"
    with open(f"/checkpoint/vineelkpratap/data/ethnologue/{iso}.txt") as f:
        soup = BeautifulSoup(f.read(), features="lxml")
#     print(soup.text)
    name = soup.find("h1",id= "page-title").text
    all_divs = soup.find_all("div")
    alternate_names = "NULL"
    c1 = "NULL"
    c2 = "NULL"
    pop = "NULL"
    status = "NULL"
    writing = "NULL"
    pop2 = "NULL"
    for i, ad in enumerate(all_divs):
        if ad.text.lower() == "alternate names":
            alternate_names = all_divs[i+1].text.strip()
        elif ad.text.lower() == "classification":
            tree = all_divs[i+1].text.split(",")
            if len(tree) > 0:
                c1 = tree[0].strip()
            if len(tree) > 1:
                c2 = tree[1].strip()
        elif ad.text.lower() == "user population":
            pop = all_divs[i+1].text.strip().replace("\n", " ")
            all_pops = re.findall(r'\d+', pop.replace(",", "").replace(".", " "))
            all_pops.append(0)
            final_pop = max([int(ap) for ap in all_pops])
            pop2 = final_pop
        elif ad.text.lower() == "language status":
            status = all_divs[i+1].text.strip().replace("\n", " ").split("We use an asterisk")[0]
        elif ad.text.lower() == "writing":
            writing = all_divs[i+1].text.strip().replace("\n", " ")
    # return iso, name, alternate_names, c1, c2, writing, status, pop, pop2

    # if writing != "NULL":
    #     toks = writing.split(".")
    #     for t in toks:
    #         if "primary usage" in t:
    #             return t.strip()

    return writing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--lang', type=str)
    args = parser.parse_args()

    print(get_ethnologue_info(args.lang))