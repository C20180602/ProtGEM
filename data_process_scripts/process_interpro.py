'''
用于处理interpro数据到GO标签的映射
'''

import os
import json
# 设置工作路径为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

ipr2go_map = {}
with open(os.path.join("raw_data","interpro_entry.txt"),"r") as f:
    f.readline() # 忽略第一行
    line = f.readline()
    while line:
        interpro = line.split("\t")[0]
        ipr2go_map[interpro]=[]
        line = f.readline()

with open(os.path.join("raw_data","interpro2go.txt"),"r") as f:
    line = f.readline()
    while line:
        if line.startswith("!"):
            line = f.readline()
            continue
        line = line.strip()
        interpro = line.split(" ")[0].split(":")[1]
        go = line.split(" ")[-1]
        if interpro not in ipr2go_map:
            ipr2go_map[interpro]=[]
        ipr2go_map[interpro].append(go)
        line = f.readline()

with open(os.path.join("processed_data","interpro2go.json"), 'w') as f:
    json.dump(ipr2go_map, f, indent=4)

with open(os.path.join("processed_data","interpro_id.json"), 'w') as f:
    json.dump(list(ipr2go_map.keys()), f, indent=4)

