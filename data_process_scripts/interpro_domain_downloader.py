'''
修改自InterPro官网上的代码
用于读取InterPro上的查找结果 export.tsv 并根据结果下载所有蛋白质的结构域信息
'''
# standard library modules
import sys, errno, re, json, ssl, os
import click
from urllib import request
from urllib.error import HTTPError
from time import sleep
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

@click.command()
@click.option('-s', '--species-name', type=click.STRING)
def main(species_name):
    # 设置工作路径为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    # 先读取之前已经完成的进度
    exist_file_list = set()
    already_exist_file = os.listdir(os.path.join("raw_data",species_name,"domain"))
    for file in already_exist_file:
        if file.endswith(".tsv"):
            exist_file_list.add(file.split(".")[0])

    # 读取需要处理的pid列表
    with open(os.path.join("raw_data",species_name,"pid_list.json"),"r") as f:
        pid_list = json.load(f)
    # 先收集所有需要下载的任务
    download_tasks = []
    for pid in tqdm(pid_list):
        if pid not in exist_file_list:
            url = f"https://www.ebi.ac.uk:443/interpro/api/entry/InterPro/protein/reviewed/{pid}/?page_size=200"
            file_path = os.path.join("raw_data",species_name,"domain", pid+'.tsv')
            download_tasks.append((url, file_path))

    # 使用线程池并发下载
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_to_file, url, file_path) for url, file_path in download_tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading domains"):
            pass
    sleep(10.0)

if __name__ == "__main__":
    main()