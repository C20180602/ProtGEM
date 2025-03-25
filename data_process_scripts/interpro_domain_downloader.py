'''
修改自InterPro官网上的代码
用于读取InterPro上的查找结果 export.tsv 并根据结果下载所有蛋白质的结构域信息
'''
# standard library modules
import sys, errno, re, json, ssl, os
from urllib import request
from urllib.error import HTTPError
from time import sleep

# 设置工作路径为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

def parse_items(items):
    if type(items)==list:
        return ",".join(items)
    return ""
def parse_member_databases(dbs):
    if type(dbs)==dict:
        return ";".join([f"{db}:{','.join(dbs[db])}" for db in dbs.keys()])
    return ""
def parse_go_terms(gos):
    if type(gos)==list:
        return ",".join([go["identifier"] for go in gos])
    return ""
def parse_locations(locations):
    if type(locations)==list:
        return ",".join(
        [",".join([f"{fragment['start']}..{fragment['end']}" 
                    for fragment in location["fragments"]
                    ])
        for location in locations
        ])
    return ""
def parse_group_column(values, selector):
    return ",".join([parse_column(value, selector) for value in values])

def parse_column(value, selector):
    if value is None:
        return ""
    elif "member_databases" in selector:
        return parse_member_databases(value)
    elif "go_terms" in selector: 
        return parse_go_terms(value)
    elif "children" in selector: 
        return parse_items(value)
    elif "locations" in selector:
        return parse_locations(value)
    return str(value)

def download_to_file(url, file_path):
    #disable SSL verification to avoid config issues
    context = ssl._create_unverified_context()

    next = url
    last_page = False

    
    attempts = 0
    while next:
        try:
            req = request.Request(next, headers={"Accept": "application/json"})
            res = request.urlopen(req, context=context)
            # If the API times out due a long running query
            if res.status == 408:
                # wait just over a minute
                sleep(61)
                # then continue this loop with the same URL
                continue
            elif res.status == 204:
                #no data so leave loop
                break
            payload = json.loads(res.read().decode())
            next = payload["next"]
            attempts = 0
            if not next:
                last_page = True
        except HTTPError as e:
            if e.code == 408:
                sleep(61)
                continue
            else:
                # If there is a different HTTP error, it wil re-try 3 times before failing
                if attempts < 3:
                    attempts += 1
                    sleep(61)
                    continue
                else:
                    sys.stderr.write("LAST URL: " + next)
                    raise e
        with open(file_path,"w+") as f:
            for i, item in enumerate(payload["results"]):
                f.write(parse_column(item["metadata"]["accession"], 'metadata.accession') + "\t")
                f.write(parse_column(item["metadata"]["name"], 'metadata.name') + "\t")
                f.write(parse_column(item["metadata"]["source_database"], 'metadata.source_database') + "\t")
                f.write(parse_column(item["metadata"]["type"], 'metadata.type') + "\t")
                f.write(parse_column(item["metadata"]["integrated"], 'metadata.integrated') + "\t")
                f.write(parse_column(item["metadata"]["member_databases"], 'metadata.member_databases') + "\t")
                f.write(parse_column(item["metadata"]["go_terms"], 'metadata.go_terms') + "\t")
                f.write(parse_column(item["proteins"][0]["accession"], 'proteins[0].accession') + "\t")
                f.write(parse_column(item["proteins"][0]["protein_length"], 'proteins[0].protein_length') + "\t")
                f.write(parse_column(item["proteins"][0]["entry_protein_locations"], 'proteins[0].entry_protein_locations') + "\t")
                f.write("\n")
        
    # Don't overload the server, give it time before asking for more
    sleep(1)

# 设置物种名称，大写
species_name = "CAFA3_train"
# 先读取之前已经完成的进度
exist_file_list = set()
already_exist_file = os.listdir(os.path.join("raw_data",species_name,"domain"))
for file in already_exist_file:
    if file.endswith(".tsv"):
        exist_file_list.add(file.split(".")[0])

with open(os.path.join("raw_data",species_name,"seq","seq.fasta")) as f:
    line = f.readline()
    cnt = 0
    while line:
        if line.startswith(">"):
            line = line.strip()
            cnt += 1
            print(cnt)
            if cnt<=62579:
                line = f.readline()
                continue 
            protein_id = line.split(">")[1]
            # 如果之前完成了，就跳过该条信息
            if protein_id in exist_file_list:
                print(protein_id, " exists")
                line = f.readline()
                continue
            url = f"https://www.ebi.ac.uk:443/interpro/api/entry/InterPro/protein/reviewed/{protein_id}/?page_size=200"
            download_to_file(url,os.path.join("raw_data",species_name,"domain", protein_id+'.tsv'))
        line = f.readline()
