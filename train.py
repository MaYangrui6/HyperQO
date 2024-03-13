import os
import random
from datetime import datetime

import torch
import sys

sys.path.append('/home/ubuntu/project/mayang/HyperQO/')
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
from sklearn.model_selection import train_test_split

config = Config()
# sys.stdout = open(config.log_file, "w")
random.seed(0)
current_dir = os.path.dirname(__file__)
import ast

if __name__ == "__main__":
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

    train = pd.read_csv(current_dir + '/information/train.csv', index_col=0)
    queries = train['query'].values

    plans_json = train["plan_json"].values

    workload_embedder_path = os.path.join("./information/", "embedder.pth")
    workload_embedder = PredicateEmbedderDoc2Vec(queries[:], plans_json, 20, database_runner=pgrunner, file_name=workload_embedder_path)

    train.head()

    x = torch.tensor(train.index)
    y = torch.tensor(train['cost_reduction_ratio'].values)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 例如，均方误差损失
    optimizer = treenet_model.optimizer  # 例如，Adam 优化器

    Batch_Size = 512
    torch_dataset = Data.TensorDataset(x, y)
    # train_set, val_set = train_test_split(torch_dataset, test_size=0.2, shuffle=True)

    run_cnt = 1

    list_loss = []
    list_batch_loss = []
    # 训练循环
    batch_num = 0
    for epoch in range(3):  # 例如，训练多个 epochs
        loader = Data.DataLoader(dataset=torch_dataset,
                                 batch_size=Batch_Size,
                                 shuffle=True,
                                 drop_last=False)
        for batch_x, batch_y in loader:
            actual_batch_size = len(batch_x)
            batch_loss = 0
            # training process
            for num in range(actual_batch_size):
                sql = queries[batch_x[num]]
                target_value = batch_y[num]
                plan_json = pgrunner.getCostPlanJson(sql)
                sql_vec = workload_embedder.get_embedding([sql])

                # 计算损失
                loss, pred_val = treenet_model.train(plan_json, sql_vec, target_value, is_train=True)
                batch_loss += loss
                list_loss.append(loss)
                print(
                    "training count {} : train loss : {}, pred_val : {}, target_value : {},  diff : {}".format(run_cnt,
                                                                                                               loss,
                                                                                                               pred_val,
                                                                                                               target_value,
                                                                                                               abs(pred_val - target_value)))
                run_cnt += 1
            print("training average loss : {}".format(batch_loss / actual_batch_size))
            optimize_loss = treenet_model.optimize()
            list_batch_loss.append(batch_loss / actual_batch_size)
            print("optimize batch loss : {}".format(optimize_loss))

    # 创建保存模型的文件夹
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(current_dir, "models", current_time)
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(save_dir, "model_value_network.pth")
    torch.save(treenet_model.value_network.state_dict(), model_path)

    # 保存训练结果
    res = pd.DataFrame()
    res['loss'] = [float(x) for x in list_loss]
    res.to_csv(os.path.join(save_dir, "training_result.csv"))

    batch = pd.DataFrame()
    batch['training batch loss'] = [float(x) for x in list_batch_loss]
    batch.to_csv(os.path.join(save_dir, "batch_result.csv"))