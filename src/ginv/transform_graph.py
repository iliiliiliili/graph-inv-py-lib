import numpy as np

from osigma.ograph import OGraph


def transform_graph_for_umap_node_level(graph: OGraph, dtype=np.float32):

    result = np.empty([graph.node_count, len(graph.nodes.features)], dtype=dtype)

    for i in range(len(graph.nodes.features)):
        result[:, i] = graph.nodes.features[i]

    return result


def transform_graph_for_umap_connection_level(graph: OGraph, dtype=np.float32):

    result = np.empty([graph.connection_count, 1 + 2 * len(graph.nodes.features)], dtype=dtype)

    for i in range(len(graph.nodes.features)):
        result[:, 1 + i] = graph.nodes.features[i][graph.connections.froms]
        result[:, 1 + len(graph.nodes.features) + i] = graph.nodes.features[i][graph.connections.tos]

    return result


def transform_connection_level_features_to_node_level_features(graph: OGraph, features):

    result = np.zeros([graph.node_count, features.shape[1]], dtype=features.dtype)

    for i in range(graph.connection_count):
        result[graph.connections.froms[i]] += features[i]
        result[graph.connections.tos[i]] += features[i]

    return result