import os
import sys
import click
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from register import *
from utils import load_config_clone

@click.command()
@click.option('-o', '--ont', type=click.Choice(['bp', 'cc', 'mf', 'all']), default='mf')
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2']), default='v1.1')
def main(ont, species_name, model_name, model_type):
    workspace_dir = Path("workspace",species_name,model_name)
    log_dir = Path(workspace_dir,"log")
    config_dir = Path(workspace_dir,"config")
    model_param_dir = Path(workspace_dir,"model_param")

    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    gm_param = load_config_clone(default_gm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    pg_model, _ = model_init(model_type, pm_param, gm_param, pred_param)
    pg_model.load_model(Path(model_param_dir,"{}_model.pt".format(ont)))
    pg_model.pm.save_model(Path(model_param_dir,"{}_pm_model.pt".format(ont)))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    main()