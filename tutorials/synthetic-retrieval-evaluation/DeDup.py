import concurrent.futures

import numpy as np
from Endpoints import Embed
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances


class Dedup:
    def __init__(self):
        self.embedding_model = Embed()

    def get_embedding_vector(self, text: str) -> list[float]:
        return self.embedding_model.invoke(text)

    def list2vec(self, text_list: list[str], num_workers: int = 1) -> tuple[list[str], np.ndarray]:
        def process_text(text: str) -> tuple[str, list[float]]:
            return text, self.get_embedding_vector(text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_text, text_list))

        texts, embeddings = zip(*results, strict=False)
        embeddings = np.array(embeddings)

        return list(texts), embeddings

    def clustering(self, embeddings: np.ndarray, threshold: float = 0.075) -> np.ndarray:
        cosine_dist_matrix = cosine_distances(embeddings)

        agg_clustering = AgglomerativeClustering(
            n_clusters=None,
            linkage="complete",
            distance_threshold=threshold,
        )
        return agg_clustering.fit_predict(cosine_dist_matrix)

    def execute(self, text_list: list[str]) -> list[str]:
        texts, embeddings = self.list2vec(text_list)
        labels = self.clustering(embeddings)

        unique_text = {}
        unique_text.update(
            {label: text for text, label in zip(texts, labels, strict=False) if label not in unique_text},
        )

        return list(unique_text.values())
