import concurrent.futures
import os

import numpy as np
from Endpoints import *
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances


class Dedup:
    def __init__(self):
        self.embedding_model = Embed()

    def get_embedding_vector(self, text):
        return self.embedding_model.invoke(text)

    def list2vec(self, text_list, num_workers=100):
        def process_text(text):
            return text, self.get_embedding_vector(text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_text, text_list))

        texts, embeddings = zip(*results)
        embeddings = np.array(embeddings)

        return list(texts), embeddings

    def clustering(self, embeddings, threshold=0.075):
        cosine_dist_matrix = cosine_distances(embeddings)

        agg_clustering = AgglomerativeClustering(
            n_clusters=None,
            linkage="complete",
            distance_threshold=threshold,
        )
        labels = agg_clustering.fit_predict(cosine_dist_matrix)

        return labels

    def execute(self, text_list):
        texts, embeddings = self.list2vec(text_list)
        labels = self.clustering(embeddings)

        unique_text = {}
        unique_text.update(
            {
                label: text
                for text, label in zip(texts, labels)
                if label not in unique_text
            }
        )

        return list(unique_text.values())
