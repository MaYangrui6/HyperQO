import ast
import os
import pandas as pd
from PGUtils import pgrunner
from sql_feature.utils import embed_queries_and_plans, build_similarity_index
from sql_feature.workload_embedder import PredicateEmbedderDoc2Vec, BagOfPredicates

current_dir = os.path.dirname(__file__)



if __name__ == "__main__":
    train = pd.read_csv(os.path.join(current_dir, 'information/train.csv'))
    queries = train['query'].values
    plans_json = train["plan_json"].values

    sql_embedder_path = os.path.join("./information/", "embedder.pth")
    sql_embedder = PredicateEmbedderDoc2Vec(queries, plans_json, representation_size=20, database_runner=pgrunner,
                                            file_name=sql_embedder_path)
    workload_embeddings, workload_predicates, workload_dictionary = embed_queries_and_plans(sql_embedder, queries[:100],
                                                                                            plans_json[:100])

    sim_index = build_similarity_index(sql_embedder.model, workload_embeddings, workload_predicates, workload_dictionary, num_best=5)

    sim = sim_index[workload_dictionary.doc2bow(
        BagOfPredicates().extract_predicates_from_plan(ast.literal_eval(plans_json[100])["Plan"]))]
    print(sim)
