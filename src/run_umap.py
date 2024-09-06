import math
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from src import umap
from src.umap import plot
from fire import Fire
import os

from ginv.istanbul_ein_dataset import IstanbulEinDataset
from ginv.insider_network_snapshot_dataset import (
    InsiderNetworkSnapshotDataset,
    SnapshotAggregation,
)
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


def parse_insider_days(days):

    if isinstance(days, list):
        if len(days) == 2:
            return list(range(days[0], days[1] + 1))

    return days


def save_embeddings(embeddings: np.ndarray, name):
    np.save(f"{name}.npy", embeddings)

    with open(f"{name}.bin", "bw") as f:
        embeddings.T.tofile(f)


def create_umap_data_istanbul(dataset: IstanbulEinDataset, level="node"):
    umap_data = {
        "node": transform_graph_for_umap_node_level,
        "connection": transform_graph_for_umap_connection_level,
    }[level](dataset)

    return umap_data


def create_umap_labels_istanbul(dataset: IstanbulEinDataset):

    labels = np.copy(dataset.node_profits)
    logs = np.log10(np.abs(labels))
    logs[labels == 0] = 0
    positive_labels = labels > 0
    labels[positive_labels] = np.floor(logs[positive_labels])
    labels[~positive_labels] = -np.floor(logs[~positive_labels])

    return labels


def create_umap_data_insider_snapshot(
    dataset: InsiderNetworkSnapshotDataset,
    level="node",
    normalize=True,
):
    umap_data = {
        "node": transform_graph_for_umap_node_level,
        "connection": transform_graph_for_umap_connection_level,
    }[level](dataset, normalize_single_dim=normalize)

    return umap_data


def create_umap_labels_insider_snapshot(
    dataset: InsiderNetworkSnapshotDataset,
    label_feature=0,
):
    if isinstance(label_feature, int):
        label_feature = dataset.feature_files[label_feature]

    if "year_of_birth" in label_feature:
        labels = np.copy(dataset.node_year_of_birth)
        labels -= labels % 10
    elif "gender" in label_feature:
        labels = np.copy(dataset.node_gender)
    elif "sector_code" in label_feature:
        labels = np.copy(dataset.node_sector_code)
    elif "juridical_form" in label_feature:
        labels = np.copy(dataset.node_juridical_form)
    elif "postal_code" in label_feature:
        labels = np.copy(dataset.node_postal_code)
    elif "birthday" in label_feature:
        labels = np.copy(dataset.node_birthday)
        labels -= labels % 10
    elif "country" in label_feature:
        labels = np.copy(dataset.node_country)
    elif "domicile" in label_feature:
        labels = np.copy(dataset.node_domicile)
    elif "language" in label_feature:
        labels = np.copy(dataset.node_language)
    else:
        raise ValueError()

    return labels


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
    # all_n_neighbours=[128, 64, 32, 16, 8, 4],
    all_n_neighbours=[32, 16, 8, 4, 2],
    plots_dir="./plots/umap",
    embeddings_dir="./data/embeddings",
    name_suffix="",
    save_and_load_knn=True,
    knns_dir="./data/knn",
    dataset_name="insider_snapshot",
    label_features="all",
    insider_snapshot_day_ids=[0, 10],
    insider_snapshot_aggregation=SnapshotAggregation.CONNECTED_ONCE,
    normalize_data=True,
    plot_connections=True,
    plot_connections_hammer=True,
    subplot_size=25,
    dataset=None
):

    insider_snapshot_aggregation = SnapshotAggregation(insider_snapshot_aggregation)

    plots_dir = (
        Path(plots_dir)
        / f"node_graph{'_n' if normalize_data else ''}"
        / dataset_name
        / knn_method
    )
    embeddings_dir = (
        Path(embeddings_dir)
        / f"node_graph{'_n' if normalize_data else ''}"
        / dataset_name
        / knn_method
    )

    if dataset_name == "istanbul_ein":
        if dataset is None:
            dataset = IstanbulEinDataset("./data/istanbul")
        if label_features == "all":
            label_features = [a.replace(".bin", "") for a in dataset.feature_files]
    elif dataset_name == "insider_snapshot":
        insider_snapshot_day_ids = parse_insider_days(insider_snapshot_day_ids)
        if dataset is None:
            dataset = InsiderNetworkSnapshotDataset(
                "./data/insider-network",
                insider_snapshot_day_ids,
                aggregation=insider_snapshot_aggregation,
            )
        if label_features == "all":
            label_features = [a.replace(".bin", "") for a in dataset.feature_files]
        extra_path = (
            f"{len(insider_snapshot_day_ids)}_days"
            + (
                ""
                if insider_snapshot_day_ids[0] == 0
                else f"_from_{insider_snapshot_day_ids[0]}"
            )
            + f"_agg_{insider_snapshot_aggregation.value}"
        )

        plots_dir /= extra_path
        embeddings_dir /= extra_path

    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    if save_and_load_knn:
        os.makedirs(knns_dir, exist_ok=True)

    if not isinstance(label_features, list):
        label_features = [label_features]

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    print("Creating umap_data")

    if dataset_name == "istanbul_ein":
        umap_data = create_umap_data_istanbul(dataset, normalize=normalize_data)
        knn_use_weights = True
    elif dataset_name == "insider_snapshot":
        umap_data = create_umap_data_insider_snapshot(dataset, normalize=normalize_data)
        knn_use_weights = True
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

        name = f"umap_{dataset.node_count}n_{n_neighbours}nb{name_suffix}"
        knn_path = f"{knns_dir}/knn_{dataset_name}_{knn_method}_{dataset.node_count}n_{n_neighbours}nb.npy"
        knn_dist_path = f"{knns_dir}/knn_dist_{dataset_name}_{knn_method}_{dataset.node_count}n_{n_neighbours}nb.npy"

        if knn is None:

            if save_and_load_knn and os.path.exists(knn_path):
                knn, knn_distances = load_knn(knn_path, knn_dist_path)
                print("Loaded knn from file")
            else:
                knn, knn_distances = knn_function(
                    dataset, n_neighbours, use_weights=knn_use_weights
                )

                if save_and_load_knn:
                    save_knn(knn, knn_distances, knn_path, knn_dist_path)
        else:
            knn = reduce_knn(knn, n_neighbours)
            knn_distances = reduce_knn(knn_distances, n_neighbours)

            if save_and_load_knn:
                save_knn(knn, knn_distances, knn_path, knn_dist_path)

        print(f"Fitting umap for {name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = umap.UMAP(
                n_neighbors=n_neighbours, precomputed_knn=(knn, knn_distances)
            )

            mapper = reducer.fit(umap_data)

        print(f"Plotting for {plots_dir}/{name}")

        box_size = math.ceil(math.sqrt(len(label_features)))

        figure, axarr = plt.subplots(
            box_size,
            box_size,
            figsize=(subplot_size * box_size, subplot_size * box_size),
        )

        for i, label_feature in enumerate(label_features):

            label_feature_name = (
                label_feature
                if isinstance(label_feature, str)
                else f"label_feature_{label_feature}"
            )

            if dataset_name == "istanbul_ein":
                labels = create_umap_labels_istanbul(
                    dataset, label_feature=label_feature
                )
            elif dataset_name == "insider_snapshot":
                labels = create_umap_labels_insider_snapshot(
                    dataset, label_feature=label_feature
                )
            else:
                raise ValueError(f"Unknown dataset name {dataset_name}")
            ax = umap.plot.points(
                mapper,
                labels=labels,
                ax=axarr[i % box_size, i // box_size],
                theme="blue",
            )
            ax.title.set_text(
                f"Label: {label_feature_name}, n_neighbours: {n_neighbours}"
            )
            # plt.savefig(f"{plots_dir}/{name}_{label_feature_name}.png")

        figure.savefig(f"{plots_dir}/{name}.png")

        if plot_connections:
            umap.plot.connectivity(
                mapper,
                show_points=True,
                labels=labels,
                width=subplot_size * 100,
                height=subplot_size * 100,
            )
            plt.savefig(f"{plots_dir}/{name}_connectivity.png")

        if plot_connections_hammer:
            umap.plot.connectivity(
                mapper,
                show_points=True,
                labels=labels,
                width=subplot_size * 100,
                height=subplot_size * 100,
                edge_bundling="hammer",
            )
            plt.savefig(f"{plots_dir}/{name}_connectivity_hammer.png")

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
    insider_snapshot_day_ids=[0, 0],
    insider_snapshot_aggregation=SnapshotAggregation.CONNECTED_ONCE,
):

    plots_dir = Path(plots_dir) / "node"
    os.makedirs(plots_dir, exist_ok=True)

    if dataset_name == "istanbul_ein":
        dataset = IstanbulEinDataset("./data/istanbul")
    elif dataset_name == "insider_snapshot":
        dataset = InsiderNetworkSnapshotDataset(
            "./data/insider-network",
            insider_snapshot_day_ids,
            aggregation=insider_snapshot_aggregation,
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
    embeddings_dir="./data/embeddings",
    all_n_neighbours=[16, 8, 4, 2],
    all_min_dist=[0.4, 0.2],
    dataset_name="insider_snapshot",
    insider_snapshot_day_id=0,
    label_features="all",
):

    plots_dir = Path(plots_dir) / "connection"
    # embeddings_dir = Path(embeddings_dir) / "connection"

    os.makedirs(plots_dir, exist_ok=True)
    # os.makedirs(embeddings_dir, exist_ok=True)

    if dataset_name == "istanbul_ein":
        dataset = IstanbulEinDataset("./data/istanbul")

        if label_features == "all":
            label_features = [a.replace(".bin", "") for a in dataset.feature_files]
    elif dataset_name == "insider_snapshot":
        dataset = InsiderNetworkSnapshotDataset(
            "./data/insider-network", insider_snapshot_day_id
        )

        if label_features == "all":
            label_features = [a.replace(".bin", "") for a in dataset.feature_files]
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    if not isinstance(label_features, list):
        label_features = [label_features]

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    for label_feature in label_features:

        label_feature_name = (
            label_feature
            if isinstance(label_feature, str)
            else f"label_feature_{label_feature}"
        )

        print("Creating umap_data")
        if dataset_name == "istanbul_ein":
            umap_data, labels = create_umap_data_and_labels_istanbul(
                dataset, label_feature=label_feature, level="connection"
            )
        elif dataset_name == "insider_snapshot":
            umap_data, labels = create_umap_data_and_labels_insider_snapshot(
                dataset, label_feature=label_feature, level="connection"
            )
        else:
            raise ValueError(f"Unknown dataset name {dataset_name}")

        for n_neighbours in all_n_neighbours:
            for min_dist in all_min_dist:

                name = f"umap_{dataset.node_count}n_{n_neighbours}nb_{str(min_dist).replace('.', '')}d_{label_feature_name}"

                print(f"Fitting umap for {name}")
                reducer = umap.UMAP(
                    n_neighbors=n_neighbours,
                    min_dist=min_dist,
                )
                mapper = reducer.fit(umap_data)
                connection_features = mapper.transform(umap_data)
                node_features = (
                    transform_connection_level_features_to_node_level_features(
                        dataset, connection_features
                    )
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
