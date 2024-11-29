'''
GO数据处理第一步
用于过滤原始go-basic.obo文件的冗余信息
'''

import os
import json
unique_keys = {'id', 'name', 'namespace', 'def'}
multi_keys = {'is_a', 'relationship'}

# 查询替代节点对应的编号
def find(terms_map, x):
    if x not in terms_map:
        return -1
    if isinstance(terms_map[x],int):
        return terms_map[x]
    terms_map[x] = find(terms_map, terms_map[x])
    return terms_map[x]

def parse_obo(filepath, output_dir):
    terms = []
    terms_cnt = 0
    terms_map = {}
    with open(filepath,"r") as file:
        line = file.readline()
        while line:
            line = line.strip()
            
            if line.startswith("[Term]"):
                info = {}
                is_obsolete = False
                line = file.readline()
                while line:
                    line = line.strip()
                    
                    if line == "":
                        break
                    key = line.split(': ')[0]
                    value = ''.join(line.split(': ')[1:])
                    # 分类讨论
                    if key == 'is_obsolete' and value == 'true':
                        is_obsolete = True
                    if key == 'alt_id':
                        # 把可能使用GO id映射到当前的id位置上，保持统一（也许会用到）
                        terms_map[value] = info['id']
                    if key in unique_keys:
                        if key == 'def':
                            value = value.split('"')[1]
                        info[key] = value
                    if key in multi_keys:
                        # 如果是relationship的标签，则取Relation-ID作为关键词
                        if key == 'relationship':
                            key = value.split(' ')[0]
                            value = value.split(' ')[1]
                        if key not in info.keys():
                            info[key] = []
                        value = value.split(' ')[0]
                        info[key].append(value)
                    
                    line = file.readline()
                # 只加入非过时标签
                if not is_obsolete:
                    terms.append(info)
                    terms_map[info['id']] = terms_cnt
                    terms_cnt += 1
                
            line = file.readline()

    terms_id = {}
    for x in terms_map:
        terms_map[x] = find(terms_map,x)
        if terms_map[x] != -1:
            terms_id[x] = terms_map[x]
    
    terms_path = os.path.join(output_dir,"go_terms.json")
    terms_id_path = os.path.join(output_dir,"go_id.json")
    with open(terms_path,'w') as f:
        json.dump(terms,f,indent=4)
    with open(terms_id_path,'w') as f:
        json.dump(terms_id,f,indent=4)

go_basic_path = os.path.join('.','raw_data','go-basic.obo')
go_output_dir = os.path.join('.','processed_data')

parse_obo(go_basic_path,go_output_dir)