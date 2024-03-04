import os
import random
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


    train = pd.read_csv('/home/ubuntu/project/mayang/HyperQO/information/train.csv', index_col=0)
    queries = train['query'].values

    workload_embedder_path = os.path.join("./information/", "embedder.pth")
    workload_embedder = PredicateEmbedderDoc2Vec(queries[:100], 20, pgrunner, file_name=workload_embedder_path)

    train.head()

    x = torch.tensor(train.index)
    y = torch.tensor(train['cost_reduction_ratio'].values)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 例如，均方误差损失
    optimizer = treenet_model.optimizer  # 例如，Adam 优化器

    Batch_Size = 32
    torch_dataset = Data.TensorDataset(x, y)
    train_set, val_set = train_test_split(torch_dataset, test_size=0.2, shuffle=True)

    run_cnt = 1
    list_pred = []
    list_loss = []
    list_pred_val = []
    list_loss_val = []
    list_batch_loss = []
    list_batch_loss_val = []
    # 训练循环
    batch_num = 0
    for epoch in range(1):  # 例如，训练多个 epochs
        loader = Data.DataLoader(dataset=train_set,
                                 batch_size=Batch_Size,
                                 shuffle=True)
        loader_val = Data.DataLoader(dataset=val_set,
                                     batch_size=Batch_Size // 4,
                                     shuffle=True)
        loader_val = [x for x in loader_val]
        for batch_x, batch_y in loader:
            optimizer.zero_grad()  # 每个批次前先清零梯度
            batch_loss = 0
            batch_loss_val = 0
            # training process
            for num in range(Batch_Size):
                sql = queries[batch_x[num]]
                target_value = batch_y[num]
                plan_json = pgrunner.getCostPlanJson(sql)
                sql_vec = workload_embedder.get_embedding([sql])

                # 计算损失
                loss, pred_val = treenet_model.train(plan_json, sql_vec, target_value, is_train=True)
                list_loss.append(loss)
                list_pred.append(pred_val)
                print(
                    "training count {} : train loss : {}, pred_val : {}, target_value : {},  diff : {}".format(run_cnt,
                                                                                                               loss,
                                                                                                               pred_val,
                                                                                                               target_value,
                                                                                                               abs(pred_val - target_value)))
                batch_loss += loss  # 累积批次损失
                run_cnt += 1
            list_batch_loss.append(batch_loss)
            print("batch loss : {}".format(batch_loss / Batch_Size))
            # val process  4:1的比例进行验证
            for num in range(Batch_Size // 4):
                # valid process
                batch_x_val, batch_y_val = loader_val[batch_num]
                sql = queries[batch_x_val[num]]
                target_value = batch_y_val[num]
                plan_json = pgrunner.getCostPlanJson(sql)
                sql_vec = workload_embedder.get_embedding([sql])

                # 计算损失
                loss, pred_val = treenet_model.train(plan_json, sql_vec, target_value, is_train=False)
                list_loss_val.append(loss)
                list_pred_val.append(pred_val)
                print("valid epo : {}, valid loss : {}, pred_val : {}, target_value : {},  diff : {}".format(batch_num,
                                                                                                             loss,
                                                                                                             pred_val,
                                                                                                             target_value,
                                                                                                             abs(pred_val - target_value)))
                batch_loss_val += loss
            list_batch_loss_val.append(batch_loss)
            print("valid batch loss : {}".format(batch_loss_val / (Batch_Size // 4)))
            batch_num += 1  # 记录批次batch
    #保存模型
    torch.save(treenet_model.value_network.state_dict(), '/home/ubuntu/project/mayang/HyperQO/information/model_value_network.pth')
    res = pd.DataFrame()
    res['loss'] = [float(x) for x in list_loss]
    res['pred'] = [float(x) for x in list_pred]
    res.to_csv('/home/ubuntu/project/mayang/HyperQO/information/training_result.csv')
    batch = pd.DataFrame()
    batch['training batch loss'] = [float(x) for x in list_batch_loss]
    batch['valid batch liss'] = [float(x) for x in list_batch_loss_val]
    batch.to_csv('/home/ubuntu/project/mayang/HyperQO/information/batch_result.csv')