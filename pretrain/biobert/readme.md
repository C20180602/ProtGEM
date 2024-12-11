首先进入pretrain/biobert文件夹

运行go_term2txt.py，把GO标签的描述依次提取到go_def.txt中

如果对huggingface.co的连接不支持，可以先通过浏览器下载对应文件保存在pretrain/biobert/dmis-lab/biobert-large-cased-v1.1文件夹下

下载地址https://huggingface.co/dmis-lab/biobert-large-cased-v1.1/tree/main

然后运行run_embedding.py，获取embedding
```shell
export MAX_LENGTH=384
export DATA_PATH=go_def.txt
export OUTPUT_PATH=go_def.h5
export BATCH_SIZE=64

python run_embedding.py \
    --model_name_or_path dmis-lab/biobert-large-cased-v1.1 \
    --max_seq_length  ${MAX_LENGTH} \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --batch_size ${BATCH_SIZE} \
    --pooling mean
```

在dataset文件夹中建立GO文件夹

运行load_embedding.py，将h5py格式的embedding转化为tensor使用pkl保存