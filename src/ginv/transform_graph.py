import numpy as np

from osigma.ograph import OGraph


def transform_graph_for_umap_node_level(graph: OGraph, dtype=np.float32):

    result = np.empty([graph.node_count, len(graph.nodes.features)], dtype=dtype)

    for i in range(len(graph.nodes.features)):
        result[:, i] = graph.nodes.features[i]

    return result