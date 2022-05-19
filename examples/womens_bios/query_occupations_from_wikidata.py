import sys
from SPARQLWrapper import SPARQLWrapper, JSON

endpoint_url = "https://query.wikidata.org/sparql"

with open("/your/urls/here") as f:
    data = f.readlines()
urls = [i.strip() for i in data]

def get_results(endpoint_url, URL):
    query = f"""SELECT ?uriLabel ?occupation ?occupationLabel ?dob ?dobLabel WHERE {{
    <{URL}> schema:about ?uri .
    ?uri  wdt:P106 ?occupation .         
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}"""
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

all_occupations = []
for URL in urls:
    results = get_results(endpoint_url, URL)
    occupations = []
    for result in results["results"]["bindings"]:
        occupations.append(result['occupationLabel']['value'])
    all_occupations.append(result['uriLabel']['value'] + ", " + ", ".join(occupations))
    
assert(len(all_occupations) == len(urls))

with open("/your/file/output/here", "w") as o:
    for line in all_occupations:
        o.write(line.strip() + "\n")