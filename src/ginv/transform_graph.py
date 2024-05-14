import numpy as np

from osigma.ograph import OGraph


def transform_graph_for_umap_node_level(graph: OGraph, dtype=np.float32):

    result = np.empty([graph.node_count, len(graph.nodes.features)], dtype=dtype)

    for i in range(len(graph.nodes.features)):
        result[:, i] = graph.nodes.features[i]

    return result


def transform_graph_for_umap_connection_level(graph: OGraph, dtype=np.float32):

    result = np.empty(
        [graph.connection_count, 1 + 2 * len(graph.nodes.features)], dtype=dtype
    )

    for i in range(len(graph.nodes.features)):
        result[:, 1 + i] = graph.nodes.features[i][graph.connections.froms]
        result[:, 1 + len(graph.nodes.features) + i] = graph.nodes.features[i][
            graph.connections.tos
        ]

    return result


def transform_connection_level_features_to_node_level_features(graph: OGraph, features):

    result = np.zeros([graph.node_count, features.shape[1]], dtype=features.dtype)

    for i in range(graph.connection_count):
        result[graph.connections.froms[i]] += features[i]
        result[graph.connections.tos[i]] += features[i]

    return result


def transform_graph_for_umap_node_knn_simple(
    graph: OGraph,
    neighbours_count: int,
    dtype=np.int32,
    epsilon = 0.0001
):

    result = np.zeros([graph.node_count, neighbours_count], dtype=dtype)
    weights = np.ones([graph.node_count, neighbours_count], dtype=dtype) * epsilon
    sizes = np.zeros([graph.node_count], dtype=np.uint16)

    for i in range(graph.connection_count):

        if (i % 1000) == 0:
            print(f"Creating simple knn {i}/{graph.connection_count}", end="\r")

        weight = graph.connections.values[i]
        for idx, other_idx in [
            (graph.connections.froms[i], graph.connections.tos[i]),
            (graph.connections.tos[i], graph.connections.froms[i]),
        ]:
            if sizes[idx] < neighbours_count:

                result[idx, sizes[idx]] = other_idx
                weights[idx, sizes[idx]] = weight
                sizes[idx] += 1
            else:
                to_replace = None
                for q in range(sizes[idx]):
                    if weights[idx, q] < weight:
                        if (to_replace is None) or (
                            weights[idx, to_replace] > weights[idx, q]
                        ):
                            to_replace = q
                if to_replace is not None:
                    result[idx, to_replace] = other_idx
                    weights[idx, to_replace] = weight

    print()

    distances = 1.0 / weights
    return result, distances, sizes
