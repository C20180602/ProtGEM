'''
用于解压当文件夹下的所有压缩pdb文件
'''
import os

# 设置工作路径为项目文件夹
os.chdir("..")

# 切换路径，进入需要处理的文件夹
data_path = os.path.join('raw_data','HUMAN','pdb')
os.chdir(data_path)
for file_path in os.listdir('.'):
    if file_path.endswith(".pdb.gz"):
        # gzip文件解压后，压缩包会消失
        os.system(f"gzip -d {file_path}")