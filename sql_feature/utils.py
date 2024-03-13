import ast

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix, SoftCosineSimilarity
from gensim.similarities.annoy import AnnoyIndexer

from sql_feature.bag_of_predicates import BagOfPredicates


def embed_queries_and_plans(sql_embedder, workload, workload_plans):
    embeddings = sql_embedder.get_embedding(workload)
    bag = BagOfPredicates()
    predicates = []
    for plan in [ast.literal_eval(json)["Plan"] for json in workload_plans]:
        predicate = bag.extract_predicates_from_plan(plan)
        predicates.append(predicate)
    dictionary = Dictionary(predicates)
    return embeddings, predicates, dictionary


def build_similarity_index(model, embeddings, predicates, dictionary, num_trees=40, num_best=10):
    indexer = AnnoyIndexer(model, num_trees=num_trees)
    tfidf = TfidfModel(dictionary=dictionary)
    termsim_index = WordEmbeddingSimilarityIndex(embeddings, kwargs={'indexer': indexer})
    similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
    tfidf_corpus = tfidf[[dictionary.doc2bow(predicate) for predicate in predicates]]
    docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix, num_best=num_best)
    return docsim_index