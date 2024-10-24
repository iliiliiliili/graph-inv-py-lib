import os
from pathlib import Path
from typing import List
from fire import Fire

from ginv.insider_network_dynamic_dataset import ExtraFeaturesAggregation, InsiderNetworkDynamicDataset, SnapshotAggregation
from run_umap import parse_insider_days, node_graph_umap


def node_graph_umap_with_extra_embeddings(
    days: List[int] = [0, 501],
    # snapshot_days: List[int] = [0, 501],
    snapshot_aggregation: SnapshotAggregation = SnapshotAggregation.CONNECTED_ONCE,
    extra_feature_aggregation: ExtraFeaturesAggregation = ExtraFeaturesAggregation.MEAN,
    embeddings_path="../GloDyNE/output/insider_DynWalks.pkl",
    dataset_path="./data/insider-network",
    embeddings_count=64,
):
    snapshot_aggregation = SnapshotAggregation(snapshot_aggregation)
    extra_feature_aggregation = ExtraFeaturesAggregation(extra_feature_aggregation)

    days = parse_insider_days(days)

    dataset = InsiderNetworkDynamicDataset(
        dataset_path,
        days,
    )

    dataset.load_pickled_embeddings(embeddings_path)
    active_nodes = dataset.reduce_nodes()

    snapshot = dataset.aggregate_snapshots(days, snapshot_aggregation, extra_feature_aggregation)

    node_graph_umap(
        name_suffix=f"extra_embeddings_agg_{extra_feature_aggregation.value}",
        dataset_name="insider_snapshot",
        insider_snapshot_day_ids=days,
        insider_snapshot_aggregation=snapshot_aggregation,
        dataset=snapshot,
        embeddings_count=embeddings_count,
        active_nodes=active_nodes,
    )

    print("Done")


if __name__ == "__main__":
    Fire()
