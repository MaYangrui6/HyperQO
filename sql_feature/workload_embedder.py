import logging
from abc import ABC

import gensim
from PGUtils import PGGRunner
from bag_of_predicates import BagOfPredicates
from sklearn.decomposition import PCA
from gensim.models.fasttext import FastText

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class WorkloadEmbedder(object):
    def __init__(self, query_texts, representation_size, database_runner: PGGRunner, retrieve_plans=False):
        self.query_texts = query_texts
        self.plans = []
        self.database_runner = database_runner
        self.representation_size = representation_size

        if retrieve_plans:
            for query_text in query_texts:
                plan_jsons = database_runner.getCostPlanJson(query_text)
                self.plans.append(plan_jsons['Plan'])

    def get_embedding(self, workload):
        raise NotImplementedError


class PredicateEmbedder(WorkloadEmbedder, ABC):
    def __init__(self, query_texts, representation_size, database_runner):
        WorkloadEmbedder.__init__(self, query_texts, representation_size, database_runner, retrieve_plans=True)

        self.plan_embedding_cache = {}
        self.relevant_predicates = []
        self.bop_creator = BagOfPredicates()

        for plan in self.plans:
            predicate = self.bop_creator.extract_predicates_from_plan(plan)
            self.relevant_predicates.append(predicate)

        # The plans are to generate operators
        self.plans = None

        self.dictionary = gensim.corpora.Dictionary(self.relevant_predicates)
        logging.debug(f"Dictionary has {len(self.dictionary)} entries.")

        self.bow_corpus = [self.dictionary.doc2bow(predicate) for predicate in self.relevant_predicates]

        self._create_model()

        # The bow_corpus are to create model
        self.bow_corpus = None

    def _create_model(self):
        raise NotImplementedError

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
    def __init__(self, query_texts, representation_size, database_runner):
        PredicateEmbedder.__init__(self, query_texts, representation_size, database_runner)

    def _to_full_corpus(self, corpus):
        new_corpus = []
        for bow in corpus:
            new_bow = [0 for i in range(len(self.dictionary))]
            for elem in bow:
                index, value = elem
                new_bow[index] = value
            new_corpus.append(new_bow)

        return new_corpus

    def _create_model(self):
        new_corpus = self._to_full_corpus(self.bow_corpus)
        self.pca = PCA(n_components=self.representation_size)
        self.pca.fit(new_corpus)

        assert (
                sum(self.pca.explained_variance_ratio_) > 0.8
        ), f"Explained variance must be larger than 80% (is {sum(self.pca.explained_variance_ratio_)})"

    def _infer(self, bow, bop):
        new_bow = self._to_full_corpus([bow])
        return self.pca.transform(new_bow)


class PredicateEmbedderDoc2Vec(PredicateEmbedder, ABC):
    def __init__(self, query_texts, representation_size, database_runner):
        PredicateEmbedder.__init__(self, query_texts, representation_size, database_runner)

    def _create_model(self):
        tagged_predicates = []
        for idx, predicates in enumerate(self.relevant_predicates):
            tagged_predicates.append(gensim.models.doc2vec.TaggedDocument(predicates, [idx]))

        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.representation_size, min_count=3, epochs=40)
        self.model.build_vocab(tagged_predicates)
        self.model.train(tagged_predicates, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def _infer(self, bow, bop):
        vector = self.model.infer_vector(bop)
        return vector


class PredicateEmbedderLSIBOW(PredicateEmbedder):
    def __init__(self, query_texts, representation_size, database_runner):
        PredicateEmbedder.__init__(self, query_texts, representation_size, database_runner)

    def _create_model(self):
        self.lsi_bow = gensim.models.LsiModel(
            self.bow_corpus, id2word=self.dictionary, num_topics=self.representation_size
        )

        assert (
                len(self.lsi_bow.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.lsi_bow.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, bop):
        result = self.lsi_bow[bow]
        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value

        assert len(vector) == self.representation_size
        return vector


class PredicateEmbedderLSITFIDF(PredicateEmbedder, ABC):
    def __init__(self, query_texts, representation_size, database_runner):
        PredicateEmbedder.__init__(self, query_texts, representation_size, database_runner)

    def _create_model(self):
        self.tfidf = gensim.models.TfidfModel(self.bow_corpus, normalize=True)
        self.corpus_tfidf = self.tfidf[self.bow_corpus]
        self.lsi_tfidf = gensim.models.LsiModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.representation_size
        )

        assert (
                len(self.lsi_tfidf.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.lsi_tfidf.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, bop):
        result = self.lsi_tfidf[self.tfidf[bow]]
        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector


class PredicateEmbedderLDA(PredicateEmbedder, ABC):
    def __init__(self, query_texts, representation_size, database_runner):
        PredicateEmbedder.__init__(self, query_texts, representation_size, database_runner)

    def _create_model(self):
        temp = self.dictionary[0]  # This is only to "load" the dictionary.
        id2word = self.dictionary.id2token
        print(id2word)
        self.lda = gensim.models.LdaModel(corpus=self.bow_corpus, id2word=self.dictionary.id2token, chunksize=1000,
                                          alpha='auto',
                                          eta='auto',
                                          iterations=400,
                                          num_topics=self.representation_size,
                                          passes=20,
                                          eval_every=False)

    def _infer(self, bow, bop):
        result = self.lda[bow]
        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector



