from dataclasses import asdict
import numpy as np
from scipy.sparse import csr_matrix

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

    print(f"Creating simple knn {graph.connection_count}")

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

    sort_ids = np.argsort(distances, axis=-1)
    result = np.take_along_axis(result, sort_ids, -1)
    distances = np.take_along_axis(distances, sort_ids, -1)

    return result, distances


def transform_graph_for_umap_node_knn_simple_full(
    graph: OGraph,
    neighbours_count: int,
    weight_to_distance_scale = 1.0,
    dtype=np.int32,
    epsilon = 0.0001
):

    result = np.zeros([graph.node_count, neighbours_count], dtype=dtype)
    distances = np.ones([graph.node_count, neighbours_count], dtype=np.float32) / epsilon
    sizes = np.zeros([graph.node_count], dtype=np.uint16)

    print(f"Creating simple full knn {graph.connection_count}")

    for i in range(graph.connection_count):

        if (i % 1000) == 0:
            print(f"Creating simple full knn {i}/{graph.connection_count}", end="\r")

        dist = weight_to_distance_scale / (graph.connections.values[i] + epsilon)

        for idx, other_idx in [
            (graph.connections.froms[i], graph.connections.tos[i]),
            (graph.connections.tos[i], graph.connections.froms[i]),
        ]:
            if sizes[idx] < neighbours_count:

                result[idx, sizes[idx]] = other_idx
                distances[idx, sizes[idx]] = dist
                sizes[idx] += 1
            else:
                to_replace = None
                for q in range(sizes[idx]):
                    if distances[idx, q] > dist:
                        if (to_replace is None) or (
                            distances[idx, to_replace] > distances[idx, q]
                        ):
                            to_replace = q
                if to_replace is not None:
                    result[idx, to_replace] = other_idx
                    distances[idx, to_replace] = dist

    print()

    sort_ids = np.argsort(distances, axis=-1)
    result = np.take_along_axis(result, sort_ids, -1)
    distances = np.take_along_axis(distances, sort_ids, -1)

    for i in range(graph.node_count):
        while sizes[i] < neighbours_count:

            second_level_neighbours = []
            second_level_distances = []

            for q in range(sizes[i]): # result[i] = [a b c ... | 0 0 0]
                for p in range(sizes[result[i, q]]): # result[i, q] = a; result[a, p] = second_level_neighbour
                    if (result[result[i, q], p] not in second_level_neighbours) and (result[result[i, q], p] != i) and (result[result[i, q], p] not in result[i]):
                        second_level_neighbours.append(result[result[i, q], p])
                        second_level_distances.append(distances[i, q] + distances[result[i, q], p])
            
            if len(second_level_neighbours) <= 0:
                break

            q = 0

            sort_ids = [i for (_, i) in sorted((v, i) for (i, v) in enumerate(second_level_distances))]

            while ((sizes[i] < neighbours_count) and (q < len(second_level_distances))):
                neighbour = second_level_neighbours[sort_ids[q]]
                dist = second_level_distances[sort_ids[q]]

                result[i, sizes[i]] = neighbour
                distances[i, sizes[i]] = dist

                sizes[i] += 1

    sort_ids = np.argsort(distances, axis=-1)
    result = np.take_along_axis(result, sort_ids, -1)
    distances = np.take_along_axis(distances, sort_ids, -1)

    return result, distances


def transform_graph_into_a_csr_matrix(graph: OGraph):

    row = graph.connections.froms
    col = graph.connections.tos
    val = graph.connections.values

    result = csr_matrix((val, (row, col)), shape=(graph.node_count, graph.node_count))

    result = np.empty([graph.node_count, len(graph.nodes.features)], dtype=dtype)

    return result

def reduce_knn(knn: np.ndarray, new_neighbours_count: int):

    assert knn.shape[1] > new_neighbours_count

    result = knn[:, :new_neighbours_count]
    
    return result

    