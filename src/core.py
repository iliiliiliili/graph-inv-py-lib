import numpy as np

def create_embeddings(reducer, umap_data) -> np.ndarray:
    embeddings = reducer.transform(umap_data)
    return embeddings


def create_cluster_dict(clusters: np.ndarray):
    cluster_dict = {}
    
    for node, cluster_id in enumerate(clusters):

        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []

        cluster_dict[cluster_id].append(node)

    return cluster_dict

