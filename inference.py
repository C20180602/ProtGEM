import gradio as gr
from pathlib import Path
from gradio_molecule3d import Molecule3D

from utils import *

logger = init_logger("logs/website.log")

def predict_go_pdb(filepath):
    # Only Linux
    logger.info('upload address:{}'.format(filepath))
    
def update_radio2(dataset_name):
    return gr.update(choices=os.listdir(Path("workspace", dataset_name)))

def update_radio3(dataset_name, model_name):
    return gr.update(choices=os.listdir(Path("workspace", dataset_name, model_name, "model_param")))

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
    
    output = gr.Textbox(label="输出")
    
    gr.Button("提交").click(fn=predict_go_pdb, inputs=[radio1, radio2, radio3, Mol], outputs=output)
    

from fastapi import FastAPI
import uvicorn as uvi
app = FastAPI()
app = gr.mount_gradio_app(app, index, path="/")
uvi.run(app,host='127.0.0.1',port=8080)
