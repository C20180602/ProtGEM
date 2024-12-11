import os
import json
# 设置工作目录为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("../..")

with open(os.path.join("processed_data","go_terms.json"),"r") as f:
    go_terms = json.load(f)
with open(os.path.join("pretrain","biobert","go_def.txt"),"w") as fout:
    max_len = 0
    for info in go_terms:
        max_len = max(max_len,len(info["def"].split(" ")))
        fout.write(info["def"]+"\n")
    print(max_len)