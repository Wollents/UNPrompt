from torch import nn
import torch.nn.functional as F
import random
import torch
import dgl
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dgl
from model import SimplePrompt, GPFplusAtt
import numpy as np
from tqdm import tqdm


class Prompts(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_layers=2, dropout_rate=0, activation='ReLU', num_hops=4, num_prompts=5, **kwargs):
        super(Prompts, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = num_hops
        if num_layers == 0:
            return
        self.layers.append(nn.Linear(in_feats, h_feats))
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Linear(h_feats, h_feats))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.cross_attn = CrossAttn(h_feats * num_hops)

        if num_prompts < 2:
            self.prompts_weight = SimplePrompt(in_feats)
        else:
            self.prompts_weight = GPFplusAtt(in_feats, num_prompts)

    def add(self, x: Tensor):
        return self.prompts_weight.add(x)

    def get_subgraph_loss(self, X, graph, model):

        y = graph.ano_labels

        positive_indices = torch.nonzero((y == 1)).squeeze(1).tolist()
        all_negative_indices = torch.nonzero((y == 0)).squeeze(1).tolist()

        negative_indices = random.sample(all_negative_indices, len(positive_indices))
        negative_subgraph_indices = self.generate_rwr_subgraph(self.get_dgl_graph(graph), subgraph_size=4, start_idx=negative_indices)
        positive_subgraph_indices = self.generate_rwr_subgraph(self.get_dgl_graph(graph), subgraph_size=4, start_idx=positive_indices)

        negative_subgraph_readout = self.get_subgraph_embedding(model, negative_subgraph_indices, graph)
        postive_subgraph_readout = self.get_subgraph_embedding(model, positive_subgraph_indices, graph)

        p_embed = X[positive_indices]
        n_embed = X[negative_indices]

        yp = torch.ones([len(negative_indices)]).to(y.device)
        yn = -torch.ones([len(positive_indices)]).to(y.device)
        torch.nn.CosineEmbeddingLoss
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, 1), if y_i == 0
        loss_qn = F.cosine_embedding_loss(n_embed, negative_subgraph_readout, yp)
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, -1), if y_i == 1
        loss_qp = F.cosine_embedding_loss(p_embed, postive_subgraph_readout, yn)

        loss = torch.mean(loss_qp + loss_qn)

        return loss

    def get_subgraph_score(self, X, prompt_mask, graph, model, multi_round=3, batch_size=1024):
        multi_round_ano_score, ori_index = self.get_subgraph_score_for_multi_round(X, prompt_mask, graph, model, multi_round=3, batch_size=1024)
        return np.mean(multi_round_ano_score[:, ori_index], axis=0)

    def get_subgraph_score_for_multi_round(self, X, prompt_mask, graph, model, multi_round=3, batch_size=1024):
        all_index = torch.nonzero(prompt_mask == False).squeeze(1).tolist()
        if batch_size is None:
            batch_size = len(all_index)

        batch_num = len(all_index) // batch_size

        multi_round_ano_score = np.zeros((multi_round, graph.num_nodes))

        tbar = tqdm(range(multi_round), desc="Getting subgraph score")
        for round in tbar:
            random.shuffle(all_index)
            # batch_num = 2 batch_idx = 0
            dgl_graph = self.get_dgl_graph(graph)
            for batch_idx in range(batch_num):
                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    query_indices = all_index[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    query_indices = all_index[batch_idx * batch_size:]

                subgraph_indices = self.generate_rwr_subgraph(dgl_graph, 4, query_indices, X.device)
                subgraph_emb = self.get_subgraph_embedding(model, subgraph_indices, graph)
                # H_q
                query_embed = X[query_indices]

                diff = query_embed - subgraph_emb
                # score
                query_score = torch.sqrt(torch.sum(diff ** 2, dim=1))
                multi_round_ano_score[round, query_indices] = query_score.detach().cpu().numpy()
            tbar.set_postfix(test_dataset=graph.name)
            tbar.update()

        torch.cuda.empty_cache()
        ori_index = torch.nonzero(prompt_mask == False).squeeze(1).tolist()
        return multi_round_ano_score, ori_index

    def get_dgl_graph(self, data):
        if isinstance(data, Data):
            if data.edge_index is not None:
                row, col = data.edge_index
            elif 'adj' in data:
                row, col, _ = data.adj.coo()
            elif 'adj_t' in data:
                row, col, _ = data.adj_t.t().coo()
            else:
                row, col = [], []

            g = dgl.graph((row, col), num_nodes=data.num_nodes)
        return g

    def generate_rwr_subgraph(self, dgl_graph, subgraph_size, start_idx=None, device="cuda:0"):
        """Generate subgraph with RWR algorithm."""
        if start_idx is None:
            start_idx = list(range(dgl_graph.number_of_nodes()))
        if isinstance(start_idx, Tensor):
            start_idx = torch.Tensor(start_idx, device=device)
        dgl_graph = dgl_graph.to(device)
        reduced_size = subgraph_size
        trace, _ = dgl.sampling.random_walk(dgl_graph, start_idx, restart_prob=0.5, length=subgraph_size*2)
        # traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*2)
        subv = [row[row != -1].tolist() for row in trace]

        for i in range(len(subv)):
            retry_time = 0
            while len(subv[i]) < reduced_size:
                cur_trace, _ = dgl.sampling.random_walk(dgl_graph, start_idx[i], restart_prob=0.3, length=subgraph_size*4)
                subv[i] = cur_trace[cur_trace != -1].tolist()
                retry_time += 1
                if (len(subv[i]) <= reduced_size) and (retry_time > 10):
                    subv[i] = (subv[i] * reduced_size)
            # print(subv[i])
            random.shuffle(subv[i][1:])
            subv[i] = subv[i][:reduced_size]

        return subv

    def get_subgraph_embedding(self, model, indices, graph):

        readout_list = []
        for indice in indices:
            x = graph.x[indice]
            x[0, :] = 0
            adj = graph.adj[indice][:, indice]
            subgraph_embedding_list = model(Data(x=x, adj=adj))
            subgraph_embedding = self(subgraph_embedding_list)

            readout = torch.mean(subgraph_embedding, 0)
            readout_list.append(readout)

        return torch.stack(readout_list)

    def forward(self, x_list):
        # Z^{[l]} = MLP(X^{[l]}
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
            if i != len(self.layers) - 1:
                x_list = [self.act(x) for x in x_list]
        residual_list = []
        # Z^{[0]}
        first_element = x_list[0]
        for h_i in x_list[1:]:
            # R^{[l]} = Z^{[l]}-Z^{[0]}
            dif = h_i - first_element
            residual_list.append(dif)
        # H = [R^{[1]} || ... || R^{[L]}]
        residual_embed = torch.hstack(residual_list)
        return residual_embed


class CrossAttn(nn.Module):
    def __init__(self, embedding_dim):
        super(CrossAttn, self).__init__()
        self.embedding_dim = embedding_dim

        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)

    def cross_attention(self, query_X, support_X):
        Q = self.Wq(query_X)  # query
        K = self.Wk(support_X)  # key
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(
            torch.tensor(self.embedding_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_query_embeddings = torch.matmul(attention_weights, support_X)
        return weighted_query_embeddings

    def find_neighbor_indices(self, ori_indices, edge_index):
        neighbor_indices = set()
        for indice in ori_indices:
            start = torch.nonzero(edge_index[0] == indice)
            arg_indices = edge_index[1][start].squeeze(1).tolist()
            neighbor_indices.update(arg_indices)
        return list(neighbor_indices)

    def get_train_loss(self, X, graph, num_prompt):
        y = graph.ano_labels

        positive_indices = torch.nonzero((y == 1)).squeeze(1).tolist()
        all_negative_indices = torch.nonzero((y == 0)).squeeze(1).tolist()

        negative_indices = random.sample(all_negative_indices, len(positive_indices))
        negative_neighbor_indices = self.find_neighbor_indices(negative_indices, graph.edge_index)
        # H_q_i, y_i == 1
        query_p_embed = X[positive_indices]
        # H_q_i, y_i == 0
        query_n_embed = X[negative_indices]
        # H_q
        query_embed = torch.vstack([query_p_embed, query_n_embed])

        remaining_negative_indices = list(set(all_negative_indices) - set(negative_indices) - set(negative_neighbor_indices))

        if len(remaining_negative_indices) + len(negative_neighbor_indices) < num_prompt:
            raise ValueError(f"Not enough remaining negative indices to select {num_prompt} support nodes.")

        support_indices = None
        # 这里可以采样，应该采样邻居的样本
        if len(remaining_negative_indices) < num_prompt/2:
            support_indices = remaining_negative_indices + random.sample(negative_neighbor_indices, num_prompt - len(remaining_negative_indices))
        else:
            support_indices = random.sample(remaining_negative_indices, num_prompt//2) + random.sample(negative_neighbor_indices, num_prompt//2)

        support_indices = torch.tensor(support_indices).to(y.device)
        # H_k
        support_embed = X[support_indices]

        # the updated query node embeddings
        # \tilde{H_q}
        query_tilde_embeds = self.cross_attention(query_embed, support_embed)
        # tilde_p_embeds: \tilde{H_q_i}, y_i == 1; tilde_n_embeds: \tilde{H_q_i}, y_i == 0;
        tilde_p_embeds, tilde_n_embeds = query_tilde_embeds[:len(positive_indices)], query_tilde_embeds[
            len(positive_indices):]

        yp = torch.ones([len(negative_indices)]).to(y.device)
        yn = -torch.ones([len(positive_indices)]).to(y.device)
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, 1), if y_i == 0
        loss_qn = F.cosine_embedding_loss(query_n_embed, tilde_n_embeds, yp)
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, -1), if y_i == 1
        loss_qp = F.cosine_embedding_loss(query_p_embed, tilde_p_embeds, yn)
        loss = torch.mean(loss_qp + loss_qn)
        return loss

    def get_test_score(self, X, prompt_mask, y):
        # prompt node indices
        negative_indices = torch.nonzero((prompt_mask == True) & (y == 0)).squeeze(1).tolist()
        n_support_embed = X[negative_indices]
        # query node indices
        query_indices = torch.nonzero(prompt_mask == False).squeeze(1).tolist()
        # H_q
        query_embed = X[query_indices]
        # \tilde{H_q}
        query_tilde_embed = self.cross_attention(query_embed, n_support_embed)
        # dis(H_q, \tilde{H_q})
        diff = query_embed - query_tilde_embed
        # score
        query_score = torch.sqrt(torch.sum(diff ** 2, dim=1))

        return query_score
