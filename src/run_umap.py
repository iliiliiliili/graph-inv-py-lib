from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import umap
from src.umap import plot
from fire import Fire
import os

from ginv.istanbul_ein_dataset import IstanbulEinDataset
from ginv.insider_network_snapshot_dataset import InsiderNetworkSnapshotDataset
from ginv.transform_graph import (
    transform_graph_for_umap_node_level,
    transform_graph_for_umap_connection_level,
    transform_connection_level_features_to_node_level_features,
    transform_graph_for_umap_node_knn_simple,
    transform_graph_for_umap_node_knn_simple_full,
    transform_graph_for_umap_node_knn_multilevel,
    transform_graph_into_a_csr_matrix,
    reduce_knn,
    save_knn,
    load_knn,
)


def save_embeddings(embeddings: np.ndarray, name):
    np.save(f"{name}.npy", embeddings)

    with open(f"{name}.bin", "bw") as f:
        embeddings.T.tofile(f)


def create_umap_data_and_labels_istanbul(dataset: IstanbulEinDataset, level="node"):
    umap_data = {
        "node": transform_graph_for_umap_node_level,
        "connection": transform_graph_for_umap_connection_level,
    }[level](dataset)

    labels = np.copy(dataset.node_profits)
    logs = np.log10(np.abs(labels))
    logs[labels == 0] = 0
    positive_labels = labels > 0
    labels[positive_labels] = np.floor(logs[positive_labels])
    labels[~positive_labels] = -np.floor(logs[~positive_labels])

    return umap_data, labels


def create_umap_data_and_labels_insider_snapshot(
    dataset: InsiderNetworkSnapshotDataset, level="node"
):
    umap_data = {
        "node": transform_graph_for_umap_node_level,
        "connection": transform_graph_for_umap_connection_level,
    }[level](dataset)

    labels = np.copy(dataset.node_gender)

    return umap_data, labels


def node_graph_sparse_umap(
    node_count=0,
    all_n_neighbours=[64, 32, 16, 8, 4, 2],
    all_min_dist=[0.4, 0.2],
    plots_dir="./plots/umap",
    embeddings_dir="./plots/embeddings",
    weight_to_distance_scale=255.0,
    name_suffix="",
    init="random",
):

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    dataset = IstanbulEinDataset("./data/istanbul")

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    print("Creating umap_data")
    umap_data, labels = create_umap_data_and_labels_istanbul(dataset)

    print("Creating csr matrix")
    csr_graph = transform_graph_into_a_csr_matrix(dataset)

    all_n_neighbours.sort(reverse=True)

    name_suffix = ("_" + name_suffix) if len(name_suffix) > 0 else ""

    for n_neighbours in all_n_neighbours:
        for min_dist in all_min_dist:

            name = f"umap_istanbul_node_and_graph_level_csr_{dataset.node_count}n_{n_neighbours}nb_{min_dist}d_{weight_to_distance_scale}ws{name_suffix}"

            print(f"Fitting umap for {name}")
            reducer = umap.UMAP(
                n_neighbors=n_neighbours,
                init=init,
            )
            mapper = reducer.fit(csr_graph)

            print(f"Plotting for {name}")
            umap.plot.points(mapper, labels=labels)

            plt.savefig(f"{plots_dir}/{name}.png")

            # print(f"Saving embeddings for {name}")
            # embeddings = reducer.transform(umap_data)
            # save_embeddings(embeddings, f"{embeddings_dir}/embeddings_{name}", )

            print()


def node_graph_umap(
    node_count=0,
    knn_method="multilevel",
    parametric_umap=False,
    all_n_neighbours=[128, 64, 32, 16, 8, 4],
    plots_dir="./plots/umap",
    embeddings_dir="./data/embeddings",
    name_suffix="",
    save_and_load_knn=True,
    knns_dir="./data/knn",
    dataset_name="insider_snapshot",
    insider_snapshot_day_id=0,
):

    plots_dir = Path(plots_dir) / "node_graph"
    embeddings_dir = Path(embeddings_dir) / "node_graph"

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    if save_and_load_knn:
        os.makedirs(knns_dir, exist_ok=True)

    if dataset_name == "istanbul_ein":
        dataset = IstanbulEinDataset("./data/istanbul")
    elif dataset_name == "insider_snapshot":
        dataset = InsiderNetworkSnapshotDataset(
            "./data/insider-network", insider_snapshot_day_id
        )
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    print("Creating umap_data")
    
    if dataset_name == "istanbul_ein":
        umap_data, labels = create_umap_data_and_labels_istanbul(dataset)
        knn_use_weights = True
    elif dataset_name == "insider_snapshot":
        umap_data, labels = create_umap_data_and_labels_insider_snapshot(dataset)
        knn_use_weights = False
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    knn_function = {
        "simple": transform_graph_for_umap_node_knn_simple,
        "simple_full": transform_graph_for_umap_node_knn_simple_full,
        "multilevel": transform_graph_for_umap_node_knn_multilevel,
    }[knn_method]

    knn = None
    knn_distances = None

    all_n_neighbours.sort(reverse=True)

    name_suffix = ("_" + name_suffix) if len(name_suffix) > 0 else ""

    for n_neighbours in all_n_neighbours:

        name = f"umap_{knn_method}_{dataset.node_count}n_{n_neighbours}nb{name_suffix}"
        knn_path = (
            f"{knns_dir}/knn_{knn_method}_{dataset.node_count}n_{n_neighbours}nb.npy"
        )
        knn_dist_path = f"{knns_dir}/knn_dist_{knn_method}_{dataset.node_count}n_{n_neighbours}nb.npy"

        if knn is None:

            if save_and_load_knn and os.path.exists(knn_path):
                knn, knn_distances = load_knn(knn_path, knn_dist_path)
                print("Loaded knn from file")
            else:
                knn, knn_distances = knn_function(dataset, n_neighbours, use_weights=knn_use_weights)

                if save_and_load_knn:
                    save_knn(knn, knn_distances, knn_path, knn_dist_path)
        else:
            knn = reduce_knn(knn, n_neighbours)
            knn_distances = reduce_knn(knn_distances, n_neighbours)

            if save_and_load_knn:
                save_knn(knn, knn_distances, knn_path, knn_dist_path)

        print(f"Fitting umap for {name}")
        reducer = umap.UMAP(
            n_neighbors=n_neighbours, precomputed_knn=(knn, knn_distances)
        )
        mapper = reducer.fit(umap_data)

        print(f"Plotting for {name}")
        umap.plot.points(mapper, labels=labels)

        plt.savefig(f"{plots_dir}/{name}.png")

        # print(f"Saving embeddings for {name}")
        # embeddings = reducer.transform(umap_data)
        # save_embeddings(embeddings, f"{embeddings_dir}/embeddings_{name}", )

        print()


def node_umap(
    node_count=0,
    plots_dir="./plots/umap",
    embeddings_dir="./data/embeddings",
    all_n_neighbours=[128, 64, 32, 16, 8, 4],
    all_min_dist=[0.4, 0.2],
    dataset_name="insider_snapshot",
    insider_snapshot_day_id=0,
):
    
    plots_dir = Path(plots_dir) / "node"
    os.makedirs(plots_dir, exist_ok=True)

    if dataset_name == "istanbul_ein":
        dataset = IstanbulEinDataset("./data/istanbul")
    elif dataset_name == "insider_snapshot":
        dataset = InsiderNetworkSnapshotDataset(
            "./data/insider-network", insider_snapshot_day_id
        )
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    print("Creating umap_data")
    
    if dataset_name == "istanbul_ein":
        umap_data, labels = create_umap_data_and_labels_istanbul(dataset)
    elif dataset_name == "insider_snapshot":
        umap_data, labels = create_umap_data_and_labels_insider_snapshot(dataset)
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    for n_neighbours in all_n_neighbours:
        for min_dist in all_min_dist:

            name = f"umap_{dataset_name}_{dataset.node_count}n_{n_neighbours}nb_{str(min_dist).replace('.', '')}d"

            print(f"Fitting umap for {name}")
            reducer = umap.UMAP(
                n_neighbors=n_neighbours,
                min_dist=min_dist,
            )
            mapper = reducer.fit(umap_data)

            print(f"Plotting for {name}")
            umap.plot.points(mapper, labels=labels)

            plt.savefig(f"{plots_dir}/{name}.png")

            # print(f"Saving embeddings for {name}")
            # embeddings = reducer.transform(umap_data)
            # save_embeddings(embeddings, f"{embeddings_dir}/embeddings_{name}", )

            print()


def connection_umap(
    node_count=0,
    plots_dir="./plots/umap",
    all_n_neighbours=[16, 8, 4, 2],
    all_min_dist=[0.4, 0.2],
):

    os.makedirs(plots_dir, exist_ok=True)

    dataset = IstanbulEinDataset("./data/istanbul")

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    print("Creating umap_data")
    umap_data, labels = create_umap_data_and_labels_istanbul(dataset, "connection")

    for n_neighbours in all_n_neighbours:
        for min_dist in all_min_dist:

            name = f"umap_istanbul_connection_level_{dataset.node_count}n_{n_neighbours}nb_{str(min_dist).replace('.', '')}d"

            print(f"Fitting umap for {name}")
            reducer = umap.UMAP(
                n_neighbors=n_neighbours,
                min_dist=min_dist,
            )
            mapper = reducer.fit(umap_data)
            connection_features = mapper.transform(umap_data)
            node_features = transform_connection_level_features_to_node_level_features(
                dataset, connection_features
            )

            print(f"Plotting for {name}")
            plt.scatter(node_features[:, 0], node_features[:, 1], c=labels)

            plt.savefig(f"{plots_dir}/{name}.png")

            # print(f"Saving embeddings for {name}")
            # embeddings = reducer.transform(umap_data)
            # np.save(f"{plots_dir}/embeddings_connection_level_{name}", embeddings)

            print()


if __name__ == "__main__":
    Fire()
