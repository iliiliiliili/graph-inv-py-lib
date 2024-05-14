import numpy as np
import matplotlib.pyplot as plt
from src import umap
from src.umap import plot
from sklearn.datasets import load_digits
from fire import Fire
import os

from ginv.istanbul_ein_dataset import IstanbulEinDataset
from ginv.transform_graph import (
    transform_graph_for_umap_node_level,
    transform_graph_for_umap_connection_level,
    transform_connection_level_features_to_node_level_features,
    transform_graph_for_umap_node_knn_simple,
)


def node_graph_umap(node_count=0, plots_dir="./plots/umap", all_n_neighbours = [64, 32, 16, 8, 4, 2]):

    os.makedirs(plots_dir, exist_ok=True)

    dataset = IstanbulEinDataset("./data/istanbul")

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    print("Creating umap_data")
    umap_data = transform_graph_for_umap_node_level(dataset)
    labels = np.copy(dataset.node_profits)
    logs = np.log10(np.abs(labels))
    positive_labels = labels > 0
    labels[positive_labels] = np.floor(logs[positive_labels])
    labels[~positive_labels] = -np.floor(logs[~positive_labels])

    for n_neighbours in all_n_neighbours:

            name = f"umap_istanbul_node_and_graph_level_simple_{dataset.node_count}n_{n_neighbours}nb"

            knn, knn_distances, knn_size = transform_graph_for_umap_node_knn_simple(dataset, n_neighbours)

            print(f"Fitting umap for {name}")
            reducer = umap.UMAP(
                n_neighbors=n_neighbours,
                precomputed_knn=(knn, knn_distances)
            )
            mapper = reducer.fit(umap_data)

            print(f"Plotting for {name}")
            umap.plot.points(mapper, labels=labels)

            plt.savefig(f"{plots_dir}/{name}.png")

            print(f"Saving embeddings for {name}")
            embeddings = reducer.transform(umap_data)

            np.save(f"{plots_dir}/embeddings_{name}", embeddings)

            print()

def node_umap(node_count=0, plots_dir="./plots/umap", all_n_neighbours = [16, 8, 4, 2], all_min_dist = [0.4, 0.2]):

    os.makedirs(plots_dir, exist_ok=True)

    dataset = IstanbulEinDataset("./data/istanbul")

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    print("Creating umap_data")
    umap_data = transform_graph_for_umap_node_level(dataset)
    labels = np.copy(dataset.node_profits)
    logs = np.log10(np.abs(labels))
    positive_labels = labels > 0
    labels[positive_labels] = np.floor(logs[positive_labels])
    labels[~positive_labels] = -np.floor(logs[~positive_labels])

    for n_neighbours in all_n_neighbours:
        for min_dist in all_min_dist:

            name = f"umap_istanbul_{dataset.node_count}n_{n_neighbours}nb_{str(min_dist).replace('.', '')}d"

            print(f"Fitting umap for {name}")
            reducer = umap.UMAP(
                n_neighbors=n_neighbours,
                min_dist=min_dist,
            )
            mapper = reducer.fit(umap_data)

            print(f"Plotting for {name}")
            umap.plot.points(mapper, labels=labels)

            plt.savefig(f"{plots_dir}/{name}.png")

            print(f"Saving embeddings for {name}")
            embeddings = reducer.transform(umap_data)

            np.save(f"{plots_dir}/embeddings_{name}", embeddings)

            print()


def connection_umap(node_count=0, plots_dir="./plots/umap", all_n_neighbours = [16, 8, 4, 2], all_min_dist = [0.4, 0.2]):

    os.makedirs(plots_dir, exist_ok=True)

    dataset = IstanbulEinDataset("./data/istanbul")

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    print("Creating umap_data")
    umap_data = transform_graph_for_umap_connection_level(dataset)
    labels = np.copy(dataset.node_profits)
    logs = np.log10(np.abs(labels))
    positive_labels = labels > 0
    labels[positive_labels] = np.floor(logs[positive_labels])
    labels[~positive_labels] = -np.floor(logs[~positive_labels])

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
            node_features = transform_connection_level_features_to_node_level_features(dataset, connection_features)

            print(f"Plotting for {name}")
            plt.scatter(node_features[:, 0], node_features[:, 1], c=labels)

            plt.savefig(f"{plots_dir}/{name}.png")

            print(f"Saving embeddings for {name}")
            embeddings = reducer.transform(umap_data)

            np.save(f"{plots_dir}/embeddings_connection_level_{name}", embeddings)

            print()


if __name__ == "__main__":
    Fire()
