import pickle
from typing import Dict, List
import numpy as np
from enum import Enum

from ginv.insider_network_snapshot_dataset import (
    InsiderNetworkSnapshotDataset,
    SnapshotAggregation,
)


class ExtraFeaturesAggregation(Enum):
    MEAN = "mean"
    MAX = "max"


class InsiderNetworkDynamicDataset:

    def __init__(
        self,
        root: str,
        day_ids: int | List[int],
        dtypes: dict | None = None,
        feature_files: List[str] | None = None,
    ) -> None:

        self.day_ids = day_ids

        self.snapshots = [
            InsiderNetworkSnapshotDataset(
                root, day, dtypes, feature_files, load_nodes=i == 0
            )
            for i, day in enumerate(day_ids)
        ]

        self.active_nodes = list(range(0, self.snapshots[0].node_count))
        self.extra_features = None

    @property
    def node_year_of_birth(self):
        return self.snapshots[0].nodes.features[0]

    @property
    def node_gender(self):
        return self.snapshots[0].nodes.features[1]

    @property
    def node_sector_code(self):
        return self.snapshots[0].nodes.features[2]

    @property
    def node_juridical_form(self):
        return self.snapshots[0].nodes.features[3]

    @property
    def node_postal_code(self):
        return self.snapshots[0].nodes.features[4]

    @property
    def node_birthday(self):
        return self.snapshots[0].nodes.features[5]

    @property
    def node_country(self):
        return self.snapshots[0].nodes.features[6]

    @property
    def node_domicile(self):
        return self.snapshots[0].nodes.features[7]

    @property
    def node_language(self):
        return self.nodes.features[8]

    def reduce_nodes(self):

        active_nodes = set()

        for snapshot in self.snapshots:
            for f in snapshot.connections.froms:
                active_nodes.add(f)
            for t in snapshot.connections.tos:
                active_nodes.add(t)

        self.active_nodes = sorted(list(active_nodes))

        self.snapshots[0].take_nodes(self.active_nodes)

        if self.extra_features is not None:
            for i in range(len(self.snapshots)):
                self.extra_features[i] = self.extra_features[i].take(self.active_nodes, axis=0)
        
        node_id_map = {}

        for i, node in enumerate(self.active_nodes):
            node_id_map[node] = i
        
        for snapshot in self.snapshots:
            for i in range(snapshot.connection_count):
                snapshot.connections.froms[i] = node_id_map[snapshot.connections.froms[i]]
                snapshot.connections.tos[i] = node_id_map[snapshot.connections.tos[i]]
        
        return self.active_nodes

    def aggregate_snapshots(
        self,
        day_ids: int | List[int],
        snapshot_aggregation: SnapshotAggregation,
        extra_feature_aggregation: ExtraFeaturesAggregation,
    ):

        snapshots: List[InsiderNetworkSnapshotDataset] = []
        extra_features = []

        for i, day_id in enumerate(self.day_ids):
            if day_id in day_ids:
                snapshots.append(self.snapshots[i])

                if self.extra_features is not None:
                    extra_features.append(self.extra_features[i])

        result = InsiderNetworkSnapshotDataset(
            None,
            None,
            None,
            None,
            None,
            False,
        )

        result.nodes = self.snapshots[0].nodes.copy()

        if len(extra_features) > 0:

            combined_extra_features = None

            if extra_feature_aggregation == ExtraFeaturesAggregation.MEAN:
                combined_extra_features = np.mean(extra_features, axis=0)
            elif extra_feature_aggregation == ExtraFeaturesAggregation.MAX:
                combined_extra_features = np.max(extra_features, axis=0)
            else:
                raise ValueError()

            result.nodes.features.append(combined_extra_features)

        all_froms = []
        all_tos = []
        all_values = []

        for snapshot in snapshots:

            all_froms.append(snapshot.connections.froms)
            all_tos.append(snapshot.connections.tos)
            all_values.append(snapshot.connections.values)

        combined_froms, combined_tos, combined_values = (
            InsiderNetworkSnapshotDataset.aggregate_connections(
                all_froms,
                all_tos,
                all_values,
                snapshot_aggregation,
                self.snapshots[0].dtypes["connections"]["froms"],
                self.snapshots[0].dtypes["connections"]["tos"],
                self.snapshots[0].dtypes["connections"]["values"],
            )
        )

        result.connections.froms = combined_froms
        result.connections.tos = combined_tos
        result.connections.values = combined_values

        return result

    def to_networkx_list(self, prune=True):

        nodes = [*range(0, self.snapshots[0].node_count)]
        result = [a.connections_to_networkx(nodes, prune) for a in self.snapshots]

        return result

    def to_networkx_pickle(self, path, prune=True):

        networkx_list = self.to_networkx_list(prune)

        with open(path, "wb") as f:
            pickle.dump(
                networkx_list, f, protocol=4
            )  # the higher protocol, the smaller file

    def load_pickled_embeddings(self, embeddings_path):
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)

        self.extra_features = []

        for i in range(len(self.snapshots)):
            features = self.__pickled_snapshot_embeddings_to_features(embeddings[i])
            self.extra_features.append(features)

        print()

    def __pickled_snapshot_embeddings_to_features(
        self, snapshot_embeddings: Dict[str, np.ndarray]
    ):

        first_embedding = next(iter(snapshot_embeddings.values()))

        result = np.zeros(
            [self.snapshots[0].node_count, first_embedding.shape[0]],
            dtype=first_embedding.dtype,
        )

        for node, embedding in snapshot_embeddings.items():
            node_id = int(node)
            result[node_id, :] = embedding

        return result

    def __repr__(self) -> str:
        return (
            f"InsiderNetworkSnapshotDataset(with {len(self.day_ids)} days of "
            + self.snapshots[0].nodes.__repr__()
            + " and ~"
            + self.snapshots[0].connections.__repr__()
            + ")"
        )
