import os
import numpy as np
from typing import List
from pathlib import Path
from fire import Fire

from core import create_cluster_dict
from ginv.insider_network_dynamic_dataset import InsiderNetworkDynamicDataset
from ginv.insider_network_snapshot_dataset import InsiderNetworkSnapshotDataset
from run_umap import parse_insider_days


def describe_cluster_by_feature(
    dataset: InsiderNetworkSnapshotDataset,
    nodes: List[int],
    label_feature,
    max_entries_to_show=7,
    max_normalized_std=0.6,
):
    if isinstance(label_feature, int):
        label_feature = dataset.feature_files[label_feature]

    def describe_and_score_by_mean_std(name: str, features: np.ndarray):
        mean = np.mean(features)
        std = np.std(features)
        normalized_std = std / np.abs(mean)

        score = 1 - min(np.abs(normalized_std) / max_normalized_std, 1)
        result = f"{name: <15} similarity={score:.2f}, mean={mean}, std={std}"

        return result, score

    def describe_and_score_by_range(name: str, features: np.ndarray, max_difference):
        mx = np.max(features)
        mn = np.min(features)

        score = 1 - min((mx - mn) / max_difference, 1)
        result = f"{name: <15} similarity={score:.2f}, range=[{mn}, {mx}], mean={np.mean(features.take(np.where(features != 0))):.2f}"

        return result, score

    def describe_and_score_by_entries(name: str, features: np.ndarray, max_difference, max_entries_to_show=max_entries_to_show):
        entries, counts = np.unique(features, return_counts=True)
        counts = counts.astype(np.float32) / len(features)

        if len(entries) <= max_entries_to_show:
            score = (max_entries_to_show - len(entries) + 1) / max_entries_to_show
            result = f"{name: <15} similarity={score:.2f}, entries={[int(a) for a in entries]} counts={[float(a) for a in counts]}"
            return result, score

        return describe_and_score_by_range(name, features, max_difference)

    if "year_of_birth" in label_feature:
        features = np.copy(dataset.node_year_of_birth).take(nodes)
        features -= features % 5
        return describe_and_score_by_entries("Year of birth:", features, 10, max_entries_to_show=3)
    elif "gender" in label_feature:
        features = np.copy(dataset.node_gender).take(nodes)
        return describe_and_score_by_entries("Gender:", features, 1, max_entries_to_show=2)
    elif "sector_code" in label_feature:
        features = np.copy(dataset.node_sector_code).take(nodes)
        return describe_and_score_by_entries("Sector code:", features, 222000 - 100000)
    elif "juridical_form" in label_feature:
        features = np.copy(dataset.node_juridical_form).take(nodes)
        return describe_and_score_by_entries("Juridical form:", features, 5)
    elif "postal_code" in label_feature:
        features = np.copy(dataset.node_postal_code).take(nodes)
        features -= features % 1000
        return describe_and_score_by_entries("Postal code:", features, 10000)
    elif "birthday" in label_feature:
        features = np.copy(dataset.node_birthday).take(nodes)
        features -= features % 5
        return describe_and_score_by_entries("Birthday:", features, 10)
    elif "country" in label_feature:
        features = np.copy(dataset.node_country).take(nodes)
        return describe_and_score_by_entries("Country:", features, 2)
    elif "domicile" in label_feature:
        features = np.copy(dataset.node_domicile).take(nodes)
        return describe_and_score_by_entries("Domicile:", features, 5)
    elif "language" in label_feature:
        features = np.copy(dataset.node_language).take(nodes)
        return describe_and_score_by_entries("Language:", features, 2)
    else:
        raise ValueError()

def analyze_cluster_labels_insiders(
    clusters_path: str,
    days: List[int] = [0, 501],
    dataset_path="./data/insider-network",
    ignore_features=["birthday"],
    max_entries_to_show=7,
    max_normalized_std=0.6,
):
    save_path = clusters_path + ".description"
    clusters_path = Path(clusters_path)

    days = parse_insider_days(days)

    dataset = InsiderNetworkDynamicDataset(
        dataset_path,
        days,
    )
    dataset.reduce_nodes()

    snapshot = dataset.snapshots[0]
    label_features = [a.replace(".bin", "") for a in snapshot.feature_files]

    clusters = np.fromfile(clusters_path, np.int32)
    cluster_dict = create_cluster_dict(clusters)

    result = []

    for cluster_id in sorted([*cluster_dict.keys()]):

        nodes = cluster_dict[cluster_id]

        descriptions = []
        total_score = 0

        for label in label_features:

            ignore = False

            for f in ignore_features:
                if f in label:
                    ignore = True

            if ignore:
                continue

            descriptions.append(
                describe_cluster_by_feature(
                    snapshot, nodes, label, max_entries_to_show, max_normalized_std
                )
            )

        descriptions = sorted(descriptions, key=lambda x: -x[1])
        total_score = sum(a[1] for a in descriptions)

        cluster_description = "\n".join(["    " + a[0] for a in descriptions])
        cluster_description = (
            f"Cluster {cluster_id} with {len(nodes)} nodes:\n"
            + cluster_description
            + "\n"
        )

        result.append((cluster_description, total_score))

    result = sorted(result, key=lambda x: -x[1])

    result = "\n".join([a[0] for a in result])

    print(result)

    with open(save_path, "w") as f:
        print(result, file=f)

    print("Done")


if __name__ == "__main__":
    Fire()
