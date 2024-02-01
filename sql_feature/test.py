from PGUtils import PGGRunner
from workload_embedder import *
import pandas as pd


def find_operators_with_cond(plan_node):
    operators_with_cond = set()

    if 'Node Type' in plan_node:
        result_string = ''.join(map(str, plan_node.keys()))
        if 'Cond' in result_string or 'Filter' in result_string:
            operators_with_cond.add(plan_node['Node Type'])
            if plan_node['Node Type'] == 'Bitmap Heap Scan':
                print(plan_node)

    if 'Plans' in plan_node:
        # 递归遍历子节点
        for sub_plan in plan_node['Plans']:
            operators_with_cond.update(find_operators_with_cond(sub_plan))

    return operators_with_cond


if __name__ == "__main__":
    df = pd.read_csv("/home/ubuntu/project/LSTM+Attention/information/queries_filter.csv")
    df.head()

    plan_jsons = []
    pgrunner = PGGRunner(dbname="tpcds", user="postgres")
    query_texts = df["query"][:100].values
    for query in query_texts:
        plan_jsons.append(pgrunner.getCostPlanJson(query))

    workload_embedder = PredicateEmbedderDoc2Vec(query_texts, 20, pgrunner)
    print(workload_embedder.get_embedding(df["query"][100:120].values))
