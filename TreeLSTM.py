import math
from os import DirEntry

import numpy as np
import torch
import torch.nn as nn
from HyperQO.ImportantConfig import Config

config = Config()


class MSEVAR(nn.Module):
    def __init__(self, var_weight):
        super(MSEVAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, multi_value, target, var):
        var_wei = (self.var_weight * var).reshape(-1, 1)
        loss1 = torch.mul(torch.exp(-var_wei), (multi_value - target) ** 2)
        loss2 = var_wei
        loss3 = 0
        loss = (loss1 + loss2 + loss3)
        return loss.mean()


class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.fc_left = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_right = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_input = nn.Linear(input_size, 5 * hidden_size)
        elementwise_affine = False
        self.layer_norm_input = nn.LayerNorm(5 * hidden_size, elementwise_affine=elementwise_affine)
        self.layer_norm_left = nn.LayerNorm(5 * hidden_size, elementwise_affine=elementwise_affine)
        self.layer_norm_right = nn.LayerNorm(5 * hidden_size, elementwise_affine=elementwise_affine)
        self.layer_norm_c = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine)

    def forward(self, h_left, c_left, h_right, c_right, feature):
        lstm_in = self.layer_norm_left(self.fc_left(h_left))
        lstm_in += self.layer_norm_right(self.fc_right(h_right))
        lstm_in += self.layer_norm_input(self.fc_input(feature))
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * c_left +
             f2.sigmoid() * c_right)
        c = self.layer_norm_c(c)
        h = o.sigmoid() * c.tanh()
        return h, c

    def zero_h_c(self, input_dim=1):
        return torch.zeros(input_dim, self.hidden_size, device=config.device), torch.zeros(input_dim, self.hidden_size,
                                                                                           device=config.device)


class Head(nn.Module):
    def __init__(self, hidden_size):
        super(Head, self).__init__()
        self.hidden_size = hidden_size
        self.head_layer = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, 1),
                                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.head_layer(x)
        return out


class SPINN(nn.Module):

    def __init__(self, head_num, input_size, hidden_size, table_num, sql_size, attention_dim):
        super(SPINN, self).__init__()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.table_num = table_num
        self.input_size = input_size
        self.sql_size = sql_size
        self.tree_lstm = TreeLSTM(input_size=input_size, hidden_size=hidden_size)
        self.sql_layer = nn.Linear(sql_size, hidden_size)
        self.linear_M = nn.Linear(hidden_size, hidden_size)

        self.head_layer = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                        nn.Dropout(p=0.4),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, 1),
                                        nn.Sigmoid(),
                                        )
        self.table_embeddings = nn.Embedding(table_num, hidden_size)  # 2 * max_column_in_table * size)

        self.heads = nn.ModuleList([Head(self.hidden_size) for _ in range(self.head_num + 1)])
        self.relu = nn.ReLU()
        self.attention = SelfAttentionEncoderWithPositionalEmbedding(input_dim=hidden_size, attention_dim=attention_dim)
        self.H = []  # used to save all processed node for implementing attention mechanism
        self.wf = nn.Parameter(torch.zeros(1, hidden_size, hidden_size).uniform_(-0.001, 0.001))  # 初始化为较小的值

    def leaf(self, alias_id):
        table_embedding = self.table_embeddings(alias_id)
        return table_embedding, torch.zeros(table_embedding.shape, device=config.device, dtype=torch.float32)

    def input_feature(self, feature):
        return torch.tensor(np.array(feature), device=config.device, dtype=torch.float32).reshape(-1, self.input_size)

    def sql_feature(self, feature):
        return torch.tensor(np.array(feature), device=config.device, dtype=torch.float32).reshape(1, -1)

    def target_vec(self, target):
        print('target :%s',target)
        return torch.tensor([target], device=config.device, dtype=torch.float32).reshape(1, -1)

    def tree_node(self, h_left, c_left, h_right, c_right, feature):
        if feature.dim() == 2:
            feature = feature.squeeze(0)
        h, c = self.tree_lstm(h_left, c_left, h_right, c_right, feature)
        self.H.append(h)
        M, alphas = self.attention.forward(self.H, relative_position=int(feature[-4]))
        M = self.relu(M).squeeze(0)
        # # 使用一个学习到的权重矩阵对 attention_weighted_M 进行处理
        # weighted_attention_weighted_M = self.linear_M(M)
        if feature[-4] == 0:
            # 已生成 root 节点的向量化表示，清空 all processed node
            self.H = []
        return M, c

    def logits(self, encoding, sql_feature):
        sql_hidden = self.relu(self.sql_layer(sql_feature))
        out_encoding = torch.cat([encoding, sql_hidden], dim=1)
        out = self.head_layer(out_encoding)
        return out

    def zero_hc(self, input_dim=1):
        return (torch.zeros(input_dim, self.hidden_size, device=config.device),
                torch.zeros(input_dim, self.hidden_size, device=config.device))


class SelfAttentionEncoderWithPositionalEmbedding(nn.Module):
    def __init__(self, input_dim, attention_dim, dropout_rate=0.1, max_sequence_length=100):
        super(SelfAttentionEncoderWithPositionalEmbedding, self).__init__()

        self.attention_units = attention_dim
        self.hops = 1
        self.dropout = nn.Dropout(dropout_rate)

        # Linear layers
        self.linear_attention = nn.Linear(input_dim, self.attention_units, bias=False)
        self.linear_weights = nn.Linear(self.attention_units, self.hops, bias=False)

        # Positional embedding
        self.positional_embedding = self._generate_positional_embedding(input_dim, max_sequence_length)

        # Initialization
        self._init_weights()

        # Activation functions
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def _init_weights(self, init_range=0.1):
        # Weight initialization
        nn.init.uniform_(self.linear_attention.weight, -init_range, init_range)
        nn.init.uniform_(self.linear_weights.weight, -init_range, init_range)

    def _generate_positional_embedding(self, input_dim, max_sequence_length):
        position = torch.arange(0, max_sequence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * -(math.log(10000.0) / input_dim))
        pos_embedding = torch.zeros((max_sequence_length, input_dim))
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding.unsqueeze(0)  # [1, max_sequence_length, hidden_dim]

    def forward(self, input_data, relative_position):
        # Input data shape: [batch_size, seq_len, hidden_dim]
        input_data = torch.cat(input_data, dim=0)  # 将列表中的张量合并成一个张量
        # Add positional embedding
        input_data[-1] = input_data[-1] + self.positional_embedding[:, relative_position, :]

        # Adjust shape for attention computation
        # input_data = torch.stack(input_data, dim=0)  # [batch_size, seq_len, hidden_dim]
        input_data = input_data.unsqueeze(0)  # [1, batch_size, seq_len, hidden_dim]
        compressed_embeddings = input_data.view(input_data.size(1), -1)  # [batch_size * seq_len, hidden_dim]

        # Attention computation
        hbar = self.tanh(self.linear_attention(self.dropout(compressed_embeddings)))
        alphas = self.linear_weights(hbar).view(1, input_data.size(1), -1)
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [1, hops, seq_len]

        # Softmax and reshape attention weights
        alphas = self.softmax(alphas.view(-1, input_data.size(1)))  # [1 * hops, seq_len]
        alphas = alphas.view(input_data.size(0), self.hops, input_data.size(1))  # [batch_size, hops, seq_len]

        # Weighted sum to get the context vector
        context_vector = torch.bmm(alphas, input_data)  # [batch_size, hops, hidden_dim]

        return context_vector, alphas
