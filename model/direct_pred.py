import torch
from torch import nn
from .loss import lambdaLoss
import torch.nn.functional as F
from collections import defaultdict

class DirectPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._init_params(config)

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.model_embedding = nn.Embedding(self.n_models, self.hidden_size)
        self.score_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_size*2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )
        self.rank_embedding = nn.Embedding(100, self.hidden_size)

        self.encoder_layer = nn.TransformerEncoderLayer(
                d_model = self.hidden_size*4,
                nhead = self.n_heads,
                dim_feedforward = self.inner_size,
                layer_norm_eps = self.layer_norm_eps,
                dropout=0.1,
                activation = self.hidden_act,
                batch_first=True
            )
        
        self.encoder = nn.TransformerEncoder(
                encoder_layer = self.encoder_layer,
                num_layers = self.num_layers
        )

        self.output_layer = nn.Linear(self.hidden_size*5, 1)

        self.loss_func = lambdaLoss

    def _init_params(self, config):
        self.n_items = config["n_items"]
        self.n_models = config["n_models"]
        self.top_k = config["top_k"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.n_heads = config["n_heads"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.hidden_act = config["hidden_act"]
        self.num_layers = config["num_layers"]
        
    def create_labels(self, pos_indices, batch_size, seq_len=100):
        device = pos_indices.device if isinstance(pos_indices, torch.Tensor) else 'cpu'
        labels = torch.zeros(batch_size, seq_len, 1, device=device)
        batch_indices = torch.arange(batch_size, device=device)
        labels[batch_indices, pos_indices, 0] = 1
        return labels

    def forward(self, lists):
        device = next(self.parameters()).device
        model_id_map = {"list1": 0, "list2": 1, "list3": 2}
        list_keys = list(lists.keys())
        n_lists = len(list_keys)
    
        batch_size, list_len = lists["list1"]["item_ids"].shape

        final_list = []
         
        for lst_key in lists:
            items = lists[lst_key]["item_ids"].to(device)
            scores = lists[lst_key]["scores"].to(device)
        
            scores = torch.tanh(scores / 10) * 10
            scores = (scores - scores.mean()) / (scores.std() + 1e-6)

            ranks = lists[lst_key]["ranks"].to(device)
            
            item_embs = self.item_embedding(items)
            model_idx = model_id_map[lst_key]
            model_indices = torch.full((batch_size,), model_idx, device=device, dtype=torch.long)
            model_embs = self.model_embedding(model_indices).unsqueeze(1).expand(-1, list_len, -1)
    
            score_embs = self.score_embedding(scores.unsqueeze(-1))
    
            rank_embs = self.rank_embedding(ranks)
    
            final_embs = torch.cat((model_embs, item_embs, score_embs, rank_embs), dim=2)

            out = self.encoder(final_embs)

            final_cat = torch.cat([score_embs, out], dim=-1)
            alpha = self.output_layer(final_cat)
   

            final_scores = torch.sigmoid(alpha)
            
            if torch.isnan(final_scores).any():
                print("NaNs in final_scores", final_scores)
            final_list.append(final_scores)
        stacked = torch.stack(final_list, dim=0)

        return stacked.sum(dim=0).squeeze(-1)

    def calculate_loss(self, interaction):
        device = next(self.parameters()).device
        scores = self.forward(interaction)
        labels = interaction["list1"]["labels"].to(device)
        n = (labels != 0).sum(dim=1)
        loss = self.loss_func(scores, labels)
        return loss

