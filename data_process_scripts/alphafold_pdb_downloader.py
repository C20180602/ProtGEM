import os
import json
import subprocess
import click
from time import sleep

@click.command()
@click.option('-s', '--species-name', type=click.STRING)
def main(species_name):
    st = set()
    file_list = os.listdir(os.path.join("raw_data",species_name,"pdb"))
    for filename in file_list:
        if filename.startswith("AF"):
            st.add(filename.split("-")[1])

    urls = []
    with open(os.path.join("raw_data",species_name,"pid_list.json"),"r") as f:
        pid_list = json.load(f)
        for pid in pid_list:
            url = "https://alphafold.ebi.ac.uk/files/AF-"+pid+"-F1-model_v4.pdb\n"
            urls.append(url)

    for i,url in enumerate(urls):
        if i < len(file_list):
            continue
        pid = url.split("/")[-1].split("-")[1]
        if pid not in st:
            print(pid)
            output_file_path = os.path.join("raw_data",species_name,"pdb",pid+".pdb")
            try:
                result = subprocess.run(
                    ["wget", "-O", output_file_path, url.strip()],
                    check=True,  # 如果返回值不为 0，将引发异常
                    stdout=subprocess.PIPE,  # 捕获标准输出
                    stderr=subprocess.PIPE   # 捕获标准错误
                )
                print(f"文件下载成功，返回值：{result.returncode}")
            except subprocess.CalledProcessError as e:
                print(f"下载失败，返回值：{e.returncode}")
                print(f"错误信息：{e.stderr.decode()}")
                os.system("rm %s"%(output_file_path))
            sleep(1)

if __name__ == "__main__":
    main()