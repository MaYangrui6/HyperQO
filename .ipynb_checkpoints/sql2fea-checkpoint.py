import sys
sys.path.append(".")

from JOBParser import TargetTable,FromTable,Comparison
# max_column_in_table = 15
import torch
import torch
import torch.nn as nn
from itertools import count
import numpy as np
from PGUtils import pgrunner
from JOBParser import TargetTable,FromTable,Comparison
from ImportantConfig import Config
import pandas as pd
config = Config()
def zero_hc(input_dim = 1):
    return torch.zeros(input_dim,config.hidden_size,device = config.device),torch.zeros(input_dim,config.hidden_size,device = config.device)
column_id = {}
def getColumnId(column):
    if not column in column_id:
        column_id[column] = len(column_id)
    return column_id[column]
class Sql2Vec:
    def __init__(self,):
        pass
    def to_vec(self,sql):
        return np.array([1]),set(['kt1'])


import json
# 从 JSON 文件加载字典operater_counts
with open('./information/operater_counts.json', 'r') as json_file:
    operater_counts = json.load(json_file)

JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan","CTE Scan","Bitmap Heap Scan","Subquery Scan"]
ALL_TYPES=[k for k,_ in operater_counts.items()]

limit_operators = ['Limit']
aggregate_operators = ['Aggregate', 'Group', 'WindowAgg','Hash']
merge_and_join_operators = ['Gather Merge', 'Merge Join', 'Nested Loop', 'Hash Join','Merge Append']
sort_and_scan_operators = ['Sort', 'Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan', 'Bitmap Index Scan']
gather_and_materialize_operators = ['Gather', 'Materialize']
subquery_and_cte_operators = ['Subquery Scan', 'CTE Scan']
set_operation_and_others = ['SetOp', 'Append', 'Result', 'Unique']

table_statistics=pd.read_csv('./information/table_statistics.csv').iloc[70:,:]   #前70条系统表数据
table_rows=pd.read_csv('./information/table_row_counts.csv') #前70条系统表数据
Column_to_NullFraction_dict=dict(zip(table_statistics['Column'], table_statistics['Null Fraction']))
Column_to_DistinctValues_dict=dict(zip(table_statistics['Column'], table_statistics['Distinct Values']))
Column_list=table_statistics['Column'].values
Column_to_Table=dict(zip(table_statistics['Column'], table_statistics['Table']))
table_row_counts=dict(zip(table_rows['Table'], table_rows['Rows']))
col_row_counts = {}
# 遍历 Column_to_Table 字典中的每一对键值对
for col, table in Column_to_Table.items():
    # 获取列对应的表的行数
    if table in table_row_counts:
        row_count = table_row_counts[table]
        # 将列和对应的表的行数存储到新的字典中
        col_row_counts[col] = row_count
    else:
        print(f"Table '{table}' not found in table_row_counts.")

def get_relative_col(plan):   #提取当前节点的涉及到的col
    relative_col=[]
    result_string = ''
    for k,v in plan.items():
        if k not in ['Plans','Plan']:
            result_string+=str(v)
    for col in Column_list:
        if col in result_string:
            relative_col.append(col)
    return relative_col

def get_max_NullFraction_DistinctValues(cols_list):
    if len(cols_list)==0:#该算子没有涉及到列
        return [0,1]
    else:
        NullFraction_value=[Column_to_NullFraction_dict[col] for col in cols_list]
        DistinctValues_value=[Column_to_DistinctValues_dict[col]/col_row_counts[col] for col in cols_list]
        table
        
        return max(NullFraction_value),max(DistinctValues_value)



def encode_operator_heap_type(operator_type):
    # Initialize one-hot encoding vector
    encoding = [0] * 7  # There are 7 categories in total

    # Check the operator type and set the corresponding one-hot encoding
    if operator_type in limit_operators:
        encoding[0] = 1
    elif operator_type in aggregate_operators:
        encoding[1] = 1
    elif operator_type in merge_and_join_operators:
        encoding[2] = 1
    elif operator_type in sort_and_scan_operators:
        encoding[3] = 1
    elif operator_type in gather_and_materialize_operators:
        encoding[4] = 1
    elif operator_type in subquery_and_cte_operators:
        encoding[5] = 1
    elif operator_type in set_operation_and_others:
        encoding[6] = 1
    else:
        raise TreeBuilderError("Cannot extract this node type "+str(operator_type))
    return encoding

class ValueExtractor:
    def __init__(self,offset=config.offset,max_value = 20):
        self.offset = offset
        self.max_value = max_value
    # def encode(self,v):
    #     return np.log(self.offset+v)/np.log(2)/self.max_value
    # def decode(self,v):
    #     # v=-(v*v<0)
    #     return np.exp(v*self.max_value*np.log(2))#-self.offset
    def encode(self,v):
        return int(np.log(2+v)/np.log(config.max_time_out)*200)/200.
        return int(np.log(self.offset+v)/np.log(config.max_time_out)*200)/200.
    def decode(self,v):
        # v=-(v*v<0)
        # return np.exp(v/2*np.log(config.max_time_out))#-self.offset
        return np.exp(v*np.log(config.max_time_out))#-self.offset
    def cost_encode(self,v,min_cost,max_cost):
        return (v-min_cost)/(max_cost-min_cost)
    def cost_decode(self,v,min_cost,max_cost):
        return (max_cost-min_cost)*v+min_cost
    def latency_encode(self,v,min_latency,max_latency):
        return (v-min_latency)/(max_latency-min_latency)
    def latency_decode(self,v,min_latency,max_latency):
        return (max_latency-min_latency)*v+min_latency
    def rows_encode(self,v,min_cost,max_cost):
        return (v-min_cost)/(max_cost-min_cost)
    def rows_decode(self,v,min_cost,max_cost):
        return (max_cost-min_cost)*v+min_cost
value_extractor = ValueExtractor()
def get_plan_stats(data):
    return [value_extractor.encode(data["Total Cost"]),data["Plan Rows"]]


class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg

def is_join(node):
    return node["Node Type"] not in LEAF_TYPES

def is_scan(node):
    return node["Node Type"] in LEAF_TYPES

# fasttext
class PredicateEncode:
    def __init__(self,):
        pass
    def stringEncoder(self,string_predicate):
        return torch.tensor([0,1]+[0]*config.hidden_size,device = config.device).float()
        pass
    def floatEncoder(self,float1,float2):
        return torch.tensor([float1,float2]+[0]*config.hidden_size,device = config.device).float()
        pass
class TreeBuilder:
    def __init__(self):
        self.__stats = get_plan_stats
        self.id2aliasname = config.id2aliasname
        self.aliasname2id = config.aliasname2id
        self.operater_embeddings = nn.Embedding(24, 10)             # modify input num and output num
        
        
    # def get_operater_type_to_embedding(self, operator_type):
    #     return self.operater_embeddings(operator_type)
        
    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.__relations:
                if rel in node[name_key]:
                    return rel

            raise TreeBuilderError("Could not find relation name for bitmap index scan")

        raise TreeBuilderError("Cannot extract relation type from node")
    def __alias_name(self, node):
        if "Alias" in node:
            return np.asarray([self.aliasname2id[node["Alias"]]])

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Cond" #if "Index Cond" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.aliasname2id:
                if rel+'.' in node[name_key]:
                    return np.asarray([-1])
                    return np.asarray([self.aliasname2id[rel]])

        #     raise TreeBuilderError("Could not find relation name for bitmap index scan")
        print(node)
        raise TreeBuilderError("Cannot extract Alias type from node")
                
    def __featurize_join(self, node,children_inputrows,current_height):
        NullFraction_DistinctValues = get_max_NullFraction_DistinctValues(get_relative_col(node))
        cost_est_rows=self.__stats(node)
        if node["Node Type"]!='Sort':
            cost_reduction=(1-children_inputrows/cost_est_rows[1])*cost_est_rows[0]
        else:
            cost_reduction=cost_est_rows[0]
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr,encode_operator_heap_type(node['Node Type']), cost_est_rows,[children_inputrows/cost_est_rows[1],current_height,cost_reduction],NullFraction_DistinctValues))
        feature = torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(-1,config.input_size)
        return feature

    def __featurize_scan(self, node,current_height):
        NullFraction_DistinctValues = get_max_NullFraction_DistinctValues(get_relative_col(node))
        cost_est_rows=self.__stats(node)
        if node["Node Type"]!='Sort':
            cost_reduction=cost_est_rows[0]
        else:
            cost_reduction=0
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr,encode_operator_heap_type(node['Node Type']), cost_est_rows,[1,current_height,cost_reduction],NullFraction_DistinctValues))
        feature = torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(-1,config.input_size)
        return feature

    def plan_to_feature_tree(self, plan ,current_height):        
        
        # children = plan["Plans"] if "Plans" in plan else []
        if "Plan" in plan:
            plan = plan["Plan"]
        children = plan["Plan"] if "Plan" in plan else (plan["Plans"] if "Plans" in plan else [])
        
        if len(children) >2:
            raise TreeBuilderError(" len(children) >2 ")
        
        if len(children) == 1:
            children_inputrows = children[0]["Plan Rows"]
            my_vec =self.__featurize_join(plan,children_inputrows,current_height)
            child_value = self.plan_to_feature_tree(children[0],current_height+1)
            return (my_vec,child_value)
        # print(plan)
        if len(children) == 2:
            children_inputrows = children[0]["Plan Rows"] * children[1]["Plan Rows"]
            my_vec =self.__featurize_join(plan,children_inputrows,current_height)
            left = self.plan_to_feature_tree(children[0],current_height+1)
            right = self.plan_to_feature_tree(children[1],current_height+1)
            # print('is_join',my_vec)
            return (my_vec, left, right)

        if not children:
            print(plan)
            s = self.__featurize_scan(plan,current_height)
            # print('is_scan',s)
            return s

        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))


        
                
                
