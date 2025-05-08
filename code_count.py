import os
dir_list = ['data_process_scripts','model','tmp_scripts','pretrain']
total_cnt = 0
for dir_name in dir_list:
    for path, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(path,file),"r") as f:
                    lines = f.readlines()
                    total_cnt += len(lines)
for file in os.listdir('.'):
    if file.endswith(".py"):
        with open(file,"r") as f:
            lines = f.readlines()
            total_cnt += len(lines)
print(total_cnt)