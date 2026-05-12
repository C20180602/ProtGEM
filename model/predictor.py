import torch
from torch import nn
import torch.nn.functional as F
from model.protein_model import ProteinEncoder, ProteinEncoderv2, ProteinEncoderv3
from model.go_model import GoEncoder

class ProteinGoModel(nn.Module):
    '''
    蛋白质和GO标签的联合模型
    '''
    def __init__(self, pm_param, gm_param, layer_cnt=1, hid_dim=256):
        super(ProteinGoModel, self).__init__()
        self.pm = ProteinEncoder(**pm_param)
        self.gm = GoEncoder(**gm_param)
        
        assert pm_param['final_out_dim'] == gm_param['emb_dim']
        self.in_dim = pm_param['final_out_dim']
        self.hid_dim = hid_dim
        self.MLP = nn.Sequential()
        for i in range(layer_cnt):
            if i==0:
                self.MLP.append(nn.Linear(self.in_dim, self.hid_dim))
            else:
                self.MLP.append(nn.Linear(self.hid_dim, self.hid_dim))
            self.MLP.append(nn.LayerNorm(self.hid_dim))
            self.MLP.append(nn.ELU())
        self.MLP.append(nn.Linear(self.hid_dim, 1))
        self.MLP.append(nn.Sigmoid())
        
    def forward(self, seq_embs, batched_graph, domain_info, go_idx, mask=None, graph_mask=None, ablation_feat = [], pert_mode=False, get_protein_feat=False):
        if get_protein_feat:
            protein_emb, protein_emb_token = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask, ablation_feat, pert_mode, True)
        else:
            protein_emb = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask, ablation_feat, pert_mode)
        go_emb = self.gm(go_idx)
        # protein_emb: [batch_size, hid_dim]
        # go_emb: [go_nums, hid_dim]
        protein_emb = protein_emb.unsqueeze(1)
        go_emb = go_emb.unsqueeze(0)
        predict_matrix = go_emb * protein_emb
        # predict_matrix: [batch_size, go_nums, hid_dim]
        predict_matrix = self.MLP(predict_matrix)
        predict_matrix = predict_matrix.squeeze(-1)
        if get_protein_feat:
            return predict_matrix, protein_emb, protein_emb_token
        else:
            return predict_matrix
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class ReconstructDecoder(nn.Module):
    '''
    重构器，输入为[batch_size, hidden_dim(final_out_dim)] 和 [batch_size, seq_len, token_hid_dim]
    输出重构序列和重构坐标[batch_size, seq_len, seq_vocal_size] 和 [batch_size, seq_len, 3]
    '''
    def __init__(self, hidden_dim, token_hid_dim, num_tokens=21, **kwargs):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        max_seq_len = kwargs.get('max_seq_len', 2100)  # 最大2048
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)

        # 把位置和glob拼一起，融合
        self.query_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, token_hid_dim)
        )
        
        # 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_hid_dim,
            nhead=kwargs.get('num_heads', 2),
            dim_feedforward=token_hid_dim,
            dropout=kwargs.get('dropout', 0.1),
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=kwargs.get('num_layers', 1)  # 较少的层数
        )
        
        # 预测头
        self.token_head = nn.Linear(token_hid_dim, num_tokens)
        self.coord_head = nn.Sequential(
            nn.Linear(token_hid_dim, token_hid_dim // 2),
            nn.GELU(),
            nn.Linear(token_hid_dim // 2, 3),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, global_feature, encoder_hidden, lens, src_mask=None):
        batch_size, seq_len, _ = encoder_hidden.shape
        device = encoder_hidden.device
        
        # 1、投影全局特征
        global_proj = self.global_projection(global_feature)  # [batch, hidden]
        
        # 2、生成位置编码
        positions = torch.arange(seq_len, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.position_embeddings(positions)  # [batch, seq, hidden]
        
        # 3、生成查询（融合全局特征和位置信息）
        global_expanded = global_proj.expand(-1, seq_len, -1)
        query_input = torch.cat([global_expanded, pos_embeds], dim=-1)
        tgt_queries = self.query_generator(query_input)
        tgt_queries = self.dropout(tgt_queries)
        
        # 4、Transformer解码器
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        decoder_output = self.transformer_decoder(
            tgt=tgt_queries,
            memory=encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=src_mask
        )
        
        # 5、输出
        token_logits = self.token_head(decoder_output)
        coordinates = self.coord_head(decoder_output)
        coords_logits = []
        for i, len in enumerate(lens):
            coords_logits.append(coordinates[i,:len,:])
        
        return token_logits, coords_logits

class ProteinGoModelv2(nn.Module):
    '''
    蛋白质和GO标签的联合模型
    '''
    def __init__(self, pm_param, gm_param):
        super(ProteinGoModelv2, self).__init__()
        self.pm = ProteinEncoderv2(**pm_param)
        self.gm = GoEncoder(**gm_param)
        
        assert pm_param['final_out_dim'] == gm_param['emb_dim']
        self.in_dim = pm_param['final_out_dim']
        self.hid_dim = 256
        self.MLP = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ELU(),
            nn.Linear(self.hid_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq_embs, batched_graph, domain_info, go_idx, mask=None, graph_mask=None, ablation_feat = []):
        protein_emb = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask, ablation_feat)
        go_emb = self.gm(go_idx)
        # protein_emb: [batch_size, hid_dim]
        # go_emb: [go_nums, hid_dim]
        protein_emb = protein_emb.unsqueeze(1)
        go_emb = go_emb.unsqueeze(0)
        predict_matrix = go_emb * protein_emb
        # predict_matrix: [batch_size, go_nums, hid_dim]
        predict_matrix = self.MLP(predict_matrix)
        predict_matrix = predict_matrix.squeeze(-1)
        return predict_matrix
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        
class ProteinGoModelv3(nn.Module):
    '''
    蛋白质和GO标签的联合模型
    '''
    def __init__(self, pm_param, gm_param, layer_cnt=1, hid_dim=256):
        super(ProteinGoModelv3, self).__init__()
        self.pm = ProteinEncoderv3(**pm_param)
        self.gm = GoEncoder(**gm_param)
        
        assert pm_param['final_out_dim'] == gm_param['emb_dim']
        self.in_dim = pm_param['final_out_dim']
        self.hid_dim = hid_dim
        self.MLP = nn.Sequential()
        for i in range(layer_cnt):
            if i==0:
                self.MLP.append(nn.Linear(self.in_dim, self.hid_dim))
            else:
                self.MLP.append(nn.Linear(self.hid_dim, self.hid_dim))
            self.MLP.append(nn.LayerNorm(self.hid_dim))
            self.MLP.append(nn.ELU())
        self.MLP.append(nn.Linear(self.hid_dim, 1))
        self.MLP.append(nn.Sigmoid())
        
    def forward(self, seq_embs, batched_graph, domain_info, go_idx, mask=None, graph_mask=None, ablation_feat = []):
        protein_emb = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask, ablation_feat)
        go_emb = self.gm(go_idx)
        # protein_emb: [batch_size, hid_dim]
        # go_emb: [go_nums, hid_dim]
        protein_emb = protein_emb.unsqueeze(1)
        go_emb = go_emb.unsqueeze(0)
        predict_matrix = go_emb * protein_emb
        # predict_matrix: [batch_size, go_nums, hid_dim]
        predict_matrix = self.MLP(predict_matrix)
        predict_matrix = predict_matrix.squeeze(-1)
        return predict_matrix
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class PPIModel(nn.Module):
    '''
    PPI预测模型
    '''
    def __init__(self, pm_param, layer_cnt=1, hid_dim=256):
        super(PPIModel, self).__init__()
        self.pm = ProteinEncoder(**pm_param)
        
        self.in_dim = pm_param['final_out_dim']
        self.hid_dim = hid_dim
        self.MLP = nn.Sequential()
        for i in range(layer_cnt):
            if i==0:
                self.MLP.append(nn.Linear(self.in_dim*2, self.hid_dim))
                # self.MLP.append(nn.Linear(self.in_dim, self.hid_dim))
            else:
                self.MLP.append(nn.Linear(self.hid_dim, self.hid_dim))
            self.MLP.append(nn.LayerNorm(self.hid_dim))
            self.MLP.append(nn.ELU())
        self.MLP.append(nn.Linear(self.hid_dim, 1))
        self.MLP.append(nn.Sigmoid())
        
    def forward(self, seq_embs1, batched_graph1, domain_info1,
                      seq_embs2, batched_graph2, domain_info2,
                      mask1=None, graph_mask1=None, mask2=None, graph_mask2=None, ablation_feat = []):
        
        protein_emb1 = self.pm(seq_embs1, batched_graph1, domain_info1, mask1, graph_mask1, ablation_feat)
        protein_emb2 = self.pm(seq_embs2, batched_graph2, domain_info2, mask2, graph_mask2, ablation_feat)
        # protein_emb: [batch_size, hid_dim]
        protein_emb = torch.cat([protein_emb1 , protein_emb2],dim=1)
        # protein_emb = protein_emb1 * protein_emb2
        predict_matrix = self.MLP(protein_emb)
        predict_matrix = predict_matrix.squeeze(-1)
        return predict_matrix
        # predict_matrix: [batch_size]
    
    def upstream_infer(self, seq_embs, batched_graph, domain_info, ablation_feat = []):
        protein_emb = self.pm(seq_embs, batched_graph, domain_info, ablation_feat=ablation_feat)
        return protein_emb
    
    def downstream_infer(self, emb1, emb2):
        protein_emb = torch.cat([emb1 , emb2],dim=1)
        # protein_emb = emb1 * emb2
        predict_matrix = self.MLP(protein_emb)
        predict_matrix = predict_matrix.squeeze(-1)
        return predict_matrix
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))