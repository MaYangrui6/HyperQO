import random
import torch
import sys
from ImportantConfig import Config
from sql2fea import TreeBuilder, value_extractor
from NET import TreeNet
from sql2fea import Sql2Vec
from TreeLSTM import SPINN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from PGUtils import pgrunner
import numpy as np
import pandas as pd
from sql_feature.workload_embedder import PredicateEmbedderDoc2Vec

sys.path.append('/home/ubuntu/project/HyperQO')

config = Config()
# sys.stdout = open(config.log_file, "w")
random.seed(0)

if __name__ == "__main__":
    with open(config.queries_file) as f:
        import json

        queries = json.load(f)

    tree_builder = TreeBuilder()
    sql2vec = Sql2Vec()
    # 这里的 input_size 必须为偶数！
    value_network = SPINN(head_num=config.head_num, input_size=36, hidden_size=config.hidden_size, table_num=50,
                          sql_size=config.sql_size, attention_dim=30).to(config.device)
    for name, param in value_network.named_parameters():
        from torch.nn import init

        if len(param.shape) == 2:
            init.xavier_normal(param)
        else:
            init.uniform(param)

    treenet_model = TreeNet(tree_builder, value_network)

    mask = (torch.rand(1, config.head_num, device=config.device) < 0.9).long()

    train = pd.read_csv('./information/train.csv', index_col=0)
    queries = train['query'].values

    workload_embedder = PredicateEmbedderDoc2Vec(queries, 20, pgrunner)

    train.head()

    x = torch.tensor(train.index)
    y = torch.tensor(train['cost_reduction_ratio'].values)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 例如，均方误差损失
    optimizer = treenet_model.optimizer  # 例如，Adam 优化器

    Batch_Size = 32
    torch_dataset = Data.TensorDataset(x, y)

    # 训练循环
    for epoch in range(1):  # 例如，训练多个 epochs
        loader = Data.DataLoader(dataset=torch_dataset,
                                 batch_size=Batch_Size,
                                 shuffle=True)
        for batch_x, batch_y in loader:
            optimizer.zero_grad()  # 每个批次前先清零梯度
            batch_loss = 0
            for num in range(Batch_Size):
                sql = queries[batch_x[num]]
                target_value = batch_y[num]
                plan_json = pgrunner.getCostPlanJson(sql)
                sql_vec = workload_embedder.get_embedding([sql])

                # 计算损失
                loss, pred_val = treenet_model.train(plan_json, sql_vec, target_value, mask,
                                                                         is_train=True)
                print("train loss, pred_val : {} - {}".format(loss, pred_val))
                # loss = treenet_model.optimize()

                batch_loss += loss  # 累积批次损失

            print("batch loss : {}".format(batch_loss / Batch_Size))