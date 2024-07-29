import logging
import os
from abc import ABC

import gensim
from .bag_of_predicates import BagOfPredicates
from sklearn.decomposition import PCA
from gensim.models.fasttext import FastText
import ast

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class WorkloadEmbedder(object):
    def __init__(self, query_texts, query_plans, representation_size):
        self.query_texts = query_texts
        self.plans = [ast.literal_eval(json)["Plan"] for json in query_plans]
        self.representation_size = representation_size

    def get_embedding(self, workload):
        raise NotImplementedError


class PredicateEmbedder(WorkloadEmbedder, ABC):
    def __init__(self, query_texts, query_plans, representation_size, database_runner, file_name):
        WorkloadEmbedder.__init__(self, query_texts, query_plans, representation_size)
        self.plan_embedding_cache = {}
        self.relevant_predicates = []
        self.bop_creator = BagOfPredicates()
        self.model = None
        self.file_name = file_name
        self.database_runner = database_runner

        if self.file_name is not None and os.path.exists(self.file_name):
            # 如果文件名已经存在并且文件存在，则加载模型
            self.dictionary = gensim.corpora.Dictionary.load_from_text(self.file_name + "_dict")
            self.model = self.load_model(self.file_name)
        else:
            for plan in self.plans:
                predicate = self.bop_creator.extract_predicates_from_plan(plan)
                self.relevant_predicates.append(predicate)

            self.dictionary = gensim.corpora.Dictionary(self.relevant_predicates)
            logging.debug(f"Dictionary has {len(self.dictionary)} entries.")

            self.bow_corpus = [self.dictionary.doc2bow(predicate) for predicate in self.relevant_predicates]

            self._create_model()

            # 保存模型
            if self.file_name is not None:
                self.save_model(self.file_name)

            self.bow_corpus = None

    def _create_model(self):
        raise NotImplementedError

    def save_model(self, filename):
        self.dictionary.save_as_text(filename + "_dict")
        self.model.save(filename)

    @classmethod
    def load_model(cls, filename):
        # 创建一个新的实例，然后调用模型的load方法
        instance = cls.__new__(cls)
        instance.model = cls._load_model_from_file(filename)

        return instance.model

    @staticmethod
    def _load_model_from_file(filename):
        return gensim.models.Word2Vec.load(filename)

    def _infer(self, bow, bop):
        raise NotImplementedError

    def get_embedding(self, query_texts):
        embeddings = []
        plans = []
        for query_text in query_texts:
            plan_jsons = self.database_runner.getCostPlanJson(query_text)
            plans.append(plan_jsons['Plan'])

        for plan in plans:
            cache_key = str(plan)
            if cache_key not in self.plan_embedding_cache:
                bop = self.bop_creator.extract_predicates_from_plan(plan)
                bow = self.dictionary.doc2bow(bop)

                vector = self._infer(bow, bop)

                self.plan_embedding_cache[cache_key] = vector
            else:
                vector = self.plan_embedding_cache[cache_key]

            embeddings.append(vector)

        return embeddings


class PredicateEmbedderPCA(PredicateEmbedder, ABC):
    def __init__(self, query_texts, query_plans, representation_size, database_runner, file_name):
        PredicateEmbedder.__init__(self, query_texts, query_plans, representation_size, database_runner, file_name)

    def _to_full_corpus(self, corpus):
        new_corpus = []
        for bow in corpus:
            new_bow = [0 for _ in range(len(self.dictionary))]
            for elem in bow:
                index, value = elem
                new_bow[index] = value
            new_corpus.append(new_bow)

        return new_corpus

    def _create_model(self):
        new_corpus = self._to_full_corpus(self.bow_corpus)
        self.model = PCA(n_components=self.representation_size)
        self.model.fit(new_corpus)

        assert (
                sum(self.model.explained_variance_ratio_) > 0.8
        ), f"Explained variance must be larger than 80% (is {sum(self.model.explained_variance_ratio_)})"

    def _infer(self, bow, bop):
        new_bow = self._to_full_corpus([bow])
        return self.model.transform(new_bow)


class PredicateEmbedderDoc2Vec(PredicateEmbedder, ABC):
    def __init__(self, query_texts, query_plans, representation_size, database_runner, file_name):
        PredicateEmbedder.__init__(self, query_texts, query_plans, representation_size, database_runner, file_name)

    def _create_model(self):
        # tagged_predicates = []
        # for idx, predicate in enumerate(self.relevant_predicates):
        #     tagged_predicates.append(gensim.models.doc2vec.TaggedDocument(predicate, [idx]))
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.representation_size, min_count=3, epochs=2000)
        self.model.build_vocab(self.bow_corpus)
        self.model.train(self.bow_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def _infer(self, bow, bop):
        vector = self.model.infer_vector(bop)
        return vector


class PredicateEmbedderLSIBOW(PredicateEmbedder):
    def __init__(self, query_texts, query_plans, representation_size, database_runner, file_name):
        PredicateEmbedder.__init__(self, query_texts, query_plans, representation_size, database_runner, file_name)

    def _create_model(self):
        self.model = gensim.models.LsiModel(
            self.bow_corpus, id2word=self.dictionary, num_topics=self.representation_size
        )

        assert (
                len(self.model.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.model.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, bop):
        result = self.model[bow]
        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value

        assert len(vector) == self.representation_size
        return vector


class PredicateEmbedderLSITFIDF(PredicateEmbedder, ABC):
    def __init__(self, query_texts, query_plans, representation_size, database_runner, file_name):
        PredicateEmbedder.__init__(self, query_texts, query_plans, representation_size, database_runner, file_name)

    def _create_model(self):
        self.tfidf = gensim.models.TfidfModel(self.bow_corpus, normalize=True)
        self.corpus_tfidf = self.tfidf[self.bow_corpus]
        self.model = gensim.models.LsiModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.representation_size
        )

        assert (
                len(self.model.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.model.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, bop):
        result = self.model[self.tfidf[bow]]
        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector


class PredicateEmbedderLDA(PredicateEmbedder, ABC):
    def __init__(self, query_texts, query_plans, representation_size, database_runner, file_name):
        PredicateEmbedder.__init__(self, query_texts, query_plans, representation_size, database_runner, file_name)

    def _create_model(self):
        temp = self.dictionary[0]  # This is only to "load" the dictionary.
        id2word = self.dictionary.id2token
        print(id2word)
        self.model = gensim.models.LdaModel(corpus=self.bow_corpus, id2word=self.dictionary.id2token, chunksize=1000,
                                            alpha='auto',
                                            eta='auto',
                                            iterations=400,
                                            num_topics=self.representation_size,
                                            passes=20,
                                            eval_every=False)

    def _infer(self, bow, bop):
        result = self.model[bow]
        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector
