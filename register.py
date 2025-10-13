from pathlib import Path
from model.predictor import ProteinGoModel, ProteinGoModelv2, PPIModel, ContrastLoss
from model.enhanced_predictor import EnhancedProteinGoModel, EnhancedProteinGoModelv2
from ablation.seq_ablation import ProteinGoModel_seq_abl
from ablation.struct_ablation import ProteinGoModel_struct_abl
from ablation.domain_ablation import ProteinGoModel_domain_abl
from model.dataset import my_collate_fn, my_collate_fn_v2, ppi_collate_fn

# 默认配置文件路径
default_pm_conf = {
    "v1.1":Path("config","ProteinEncoder.conf"),
    "ppi_v1":Path("config","ProteinEncoder.conf"),
    "v2":Path("config","ProteinEncoder2.conf"),
    "seq_abl":Path("config","ProteinEncoder.conf"),
    "struct_abl":Path("config","ProteinEncoder.conf"),
    "domain_abl":Path("config","ProteinEncoder.conf"),
    "enhanced_v1":Path("config","EnhancedProteinEncoder.conf"),
    "enhanced_v2":Path("config","EnhancedProteinEncoder.conf")
}
default_gm_conf = {
    "v1.1":Path("config","GoEncoder.conf"),
    "v2":Path("config","GoEncoder2.conf"),
    "seq_abl":Path("config","GoEncoder.conf"),
    "struct_abl":Path("config","GoEncoder.conf"),
    "domain_abl":Path("config","GoEncoder.conf"),
    "enhanced_v1":Path("config","EnhancedGoEncoder.conf"),
    "enhanced_v2":Path("config","EnhancedGoEncoder.conf")
}
default_pred_conf = {
    "v1.1":Path("config","Predictor.conf"),
    "ppi_v1":Path("config","Predictor_ppi.conf"),
    "v2":Path("config","Predictor.conf"),
    "seq_abl":Path("config","Predictor.conf"),
    "struct_abl":Path("config","Predictor.conf"),
    "domain_abl":Path("config","Predictor.conf"),
    "enhanced_v1":Path("config","EnhancedPredictor.conf"),
    "enhanced_v2":Path("config","EnhancedPredictor.conf")
}

# 注册模型
def model_init(model_type, pm_param, gm_param, pred_param):
    if model_type == "v1.1":
        pg_model = ProteinGoModel(pm_param, gm_param, **pred_param)
        local_collate_fn = my_collate_fn
    elif model_type == "ppi_v1":
        pg_model = PPIModel(pm_param, **pred_param)
        local_collate_fn = ppi_collate_fn
    elif model_type == "v2":
        pg_model = ProteinGoModelv2(pm_param, gm_param)
        local_collate_fn = my_collate_fn_v2
    elif model_type == "seq_abl":
        pg_model = ProteinGoModel_seq_abl(pm_param, gm_param)
        local_collate_fn = my_collate_fn
    elif model_type == "struct_abl":
        pg_model = ProteinGoModel_struct_abl(pm_param, gm_param)
        local_collate_fn = my_collate_fn
    elif model_type == "domain_abl":
        pg_model = ProteinGoModel_domain_abl(pm_param, gm_param)
        local_collate_fn = my_collate_fn
    elif model_type == "enhanced_v1":
        pg_model = EnhancedProteinGoModel(pm_param, gm_param, pred_param.get('enhanced_config'))
        local_collate_fn = my_collate_fn
    elif model_type == "enhanced_v2":
        pg_model = EnhancedProteinGoModelv2(pm_param, gm_param, pred_param.get('enhanced_config'))
        local_collate_fn = my_collate_fn_v2
    else:
        return None, None
    
    return pg_model, local_collate_fn