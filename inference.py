'''
本地GO预测网页demo, 但版本较老, 不一定能运行
仅使用PDB数据, 通过InterproScan获取Interpro标签
只对单个蛋白质进行预测
若需要批量GO预测, 请构建数据集, 并使用batch_inference_go.py
若需要批量PPI预测, 请构建数据集, 先用tran_ppi.py训练模型, 再使用batch_inference_ppi.py
'''

import gradio as gr
import pickle as pkl
import networkx as nx
import Bio.PDB.PDBParser
from pathlib import Path
from gradio_molecule3d import Molecule3D
from data_process_scripts.build_pdb_graph import get_graph_info
from model.predictor import ProteinGoModel, ProteinGoModelv2
from model.dataset import probablity_propagate
import torch as th
import esm
import dgl
import pandas as pd

from utils import *

logger = init_logger("logs/website.log")

three2one= {
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 
    'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 
    'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'
}

def get_seq_info(pdbfilepath):
    parser = Bio.PDB.PDBParser()
    struct = parser.get_structure(0,pdbfilepath)
    # 只取第一个模型
    model = struct.get_models().__next__()
    # i表示当前氨基酸的序号
    seq = []
    for res in model.get_residues():
        acid = three2one[res.get_resname()]
        seq.append(acid)
    return "".join(seq)

def get_esm_embedding(seq):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    converter = alphabet.get_batch_converter()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    labels, strs, tokens = converter([("test",seq)])
    seq_emb = model(tokens, repr_layers=[33])['representations'][33]
    seq_emb = seq_emb.squeeze(0)[1:-1]
    return seq_emb

def predict_go_pdb(dataset_name, model_name, model, filepath, progress=gr.Progress()):
    device = torch.device("cuda:0")
    
    ont = model.split("_")[0]
    model_path = Path("workspace", dataset_name, model_name, "model_param", model)
    config_dir = Path("workspace", dataset_name, model_name, "config")
    cache_dir = Path("workspace", dataset_name, model_name, "cache")
    balanced_go_list_path = Path(cache_dir,f"{ont}_balanced_go_list.json")
    print(model_path)
    print(filepath)
    
    # --------------------------------------------------------
    progress(0.0, "reading sequence & getting ESM embedding")
    seq = get_seq_info(filepath)
    if len(seq) > 1022:
        return "Too long sequence!"
    progress(0.05, "reading sequence & getting ESM embedding")
    seq_emb = get_esm_embedding(seq)
    seq_emb = seq_emb.unsqueeze(0).to(device)
    
    # --------------------------------------------------------
    progress(0.1, "building graph")
    graph = get_graph_info(filepath)
    batched_graph = dgl.batch([graph]).to(device)
    
    # --------------------------------------------------------
    progress(0.2, "calling interproscan API (usually takes about 3 minutes)")
    task_id = str(Path(filepath).parent.name)[:8]
    iprscan_task_fasta_path = Path("tmp_data", task_id+".fasta")
    iprscan_result = Path("tmp_data",task_id)
    with open(iprscan_task_fasta_path, "w") as f:
        f.write(">test\n")
        f.write(seq+"\n")
    os.system(f'python iprscan5.py --email zendskia@gmail.com --sequence {str(iprscan_task_fasta_path)} --outfile {str(iprscan_result)} --outformat tsv --quiet')
    
    # --------------------------------------------------------
    progress(0.3, "processing InterPro information")
    with open(Path("processed_data","interpro_id.json"), 'r') as f:
        interpro_list = json.load(f)
        vocab_interpro = {interpro:i for i,interpro in enumerate(interpro_list)}
    domain_list = []
    domain_id_list = []
    with open(Path("tmp_data",task_id+".tsv.tsv"),"r") as f:
        line = f.readline()
        while line:
            interpro_id = line.split("\t")[11]
            start = int(line.split("\t")[6])
            end = int(line.split("\t")[7])
            if interpro_id in vocab_interpro:
                domain_list.append([vocab_interpro[interpro_id],start,end])
                domain_id_list.append(vocab_interpro[interpro_id])
            line = f.readline()
    
    glob_dm_id = th.tensor(domain_id_list,device=device).unsqueeze(0)
    domain_info = [glob_dm_id, None, [len(seq)], [domain_list]]

    # --------------------------------------------------------
    progress(0.4, "initializing model")
    # 模型参数
    config_files = os.listdir(config_dir)
    pred_param = ""
    for config_file in config_files:
        if "GoEncoder" in config_file:
            with open(Path(config_dir,config_file),"r") as f:
                gm_param = json.load(f)
        elif "ProteinEncoder" in config_file:
            with open(Path(config_dir,config_file),"r") as f:
                pm_param = json.load(f)
        elif "Predictor" in config_file:
            with open(Path(config_dir,config_file),"r") as f:
                pred_param = json.load(f)
    if pred_param != "":
        pg_model = ProteinGoModel(pm_param, gm_param, **pred_param)
    else:
        pg_model = ProteinGoModelv2(pm_param, gm_param)
    pg_model.to(device)
    pg_model.eval()
    
    # 其他必要文件
    with open(balanced_go_list_path,"r") as f:
        balanced_go_list = json.load(f)
    with open(Path("dataset","go_graph_nx.pkl"),"rb") as f:
        go_graph_nx = pkl.load(f)
    with open(Path("dataset","go_term_dis.json"),"r") as f:
        go_term_dis = json.load(f)
    with open(Path("processed_data","go_terms.json"),"r") as f:
        go_terms = json.load(f)
    balanced_go_map_inv = {go:i for i,go in enumerate(balanced_go_list)}
    topo_order = list(nx.topological_sort(go_graph_nx))
    go_idx = torch.tensor(balanced_go_list).to(device)
    
    # --------------------------------------------------------
    progress(0.6, "predicting GO terms")
    predict_matrix = pg_model(seq_emb, batched_graph, domain_info, go_idx)
    predict_matrix = predict_matrix.transpose(1,0)
    predict_matrix = probablity_propagate(predict_matrix, go_graph_nx, topo_order, balanced_go_map_inv)
    predict_matrix = predict_matrix.transpose(1,0)
    pred_list = []
    for i, pred_score in enumerate(predict_matrix[0]):
        pred_list.append((pred_score.item(), go_terms[balanced_go_list[i]]['id'], go_terms[balanced_go_list[i]]['name'][:64], go_term_dis[str(balanced_go_list[i])]))
    
    # 按照预测分数排序
    pred_list = list(sorted(pred_list,reverse=True))
    final_go_list = []
    for i, pred in enumerate(pred_list):
        final_go_list.append([i+1,pred[1],pred[2],pred[0],pred[3]])
    df = pd.DataFrame(final_go_list,columns=['Rank', 'ID', 'Name', 'Score', 'Term Layer(The Larger, the more specific)'])
    # --------------------------------------------------------
    progress(1.0, "Finish!")
    return "predict success", df
    
def update_radio2(dataset_name):
    return gr.update(choices=os.listdir(Path("workspace", dataset_name)))

def update_radio3(dataset_name, model_name):
    return gr.update(choices=os.listdir(Path("workspace", dataset_name, model_name, "model_param")))

if __name__ == "__main__":
    molecule_reps = [
        {
        "model": 0,
        "chain": "",
        "resid": "",
        "style": "stick",
        "color": "whiteCarbon",
        "residue_range": "",
        "around": 0,
        "byres": False,
        "visible": False
        }
    ]
    with gr.Blocks(theme=gr.themes.Soft(),css="static/css/gradio.css") as index:
        radio1 = gr.Radio(os.listdir("workspace"),label='Choose dataset')
        radio2 = gr.Radio([],label='Choose model name')
        radio3 = gr.Radio([],label='Choose ontology type')
        
        radio1.change(fn=update_radio2,inputs=radio1,outputs=radio2)
        radio2.change(fn=update_radio3,inputs=[radio1,radio2],outputs=radio3)
        
        Mol = Molecule3D(label='PDB file of the protein',reps=molecule_reps)
        
        text_output = gr.Textbox(label="输出")
        table_output = gr.Dataframe(label='Predict result')
        
        gr.Button("提交").click(fn=predict_go_pdb, inputs=[radio1, radio2, radio3, Mol], outputs=[text_output,table_output])
        

    from fastapi import FastAPI
    import uvicorn as uvi
    app = FastAPI()
    app = gr.mount_gradio_app(app, index, path="/")
    uvi.run(app,host='127.0.0.1',port=8080)
