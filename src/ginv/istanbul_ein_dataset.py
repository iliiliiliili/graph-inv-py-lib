from typing import List
import numpy as np
import json

from osigma.oconnections import OSpatialConnections
from osigma.ograph import OGraph
from osigma.onodes import OSpatialNodes


def read(file_name: str, file_count: int, storage: np.ndarray):
    offset = 0
    to_replace = "${FILE_ID}"

    for i in range(file_count):
        current_file_name = file_name.replace(to_replace, str(i))
        current_data = np.fromfile(current_file_name, dtype=storage.dtype)
        storage[offset : offset + len(current_data)] = current_data

        print(offset, storage[offset : offset + 4])
        print(f"loaded {current_file_name}")

        offset += len(current_data)


class IstanbulEinDataset(OGraph):

    def __init__(
        self,
        root: str,
        global_params_file: str = "global_params.json",
        dtypes: dict | None = None,
        feature_files: List[str] | None = None,
        feature_file_count=1,
    ) -> None:
        super().__init__(
            OSpatialNodes(None, None, None, [None, None, None, None, None, None]),
            OSpatialConnections(None, None, None, None, None, None, []),
        )

        self.dtypes = (
            {
                "nodes": {
                    "x_coordinates": np.float32,
                    "y_coordinates": np.float32,
                    "z_index": np.uint8,
                    "features": [
                        np.int32,
                        np.float32,
                        np.int32,
                        np.float32,
                        np.float32,
                        np.float32,
                    ],
                },
                "connections": {
                    "froms": np.int32,
                    "tos": np.int32,
                    "values": np.uint8,
                    "x_coordinates": np.float32,
                    "y_coordinates": np.float32,
                    "z_index": np.uint8,
                    "features": [],
                },
            }
            if dtypes is None
            else dtypes
        )

        self.feature_files = (
            [
                "feature_degree_${FILE_ID}.bin",
                "feature_centrality_${FILE_ID}.bin",
                "feature_number_of_trades_${FILE_ID}.bin",
                "feature_profits_${FILE_ID}.bin",
                "feature_profits_excess_${FILE_ID}.bin",
                "feature_volume_${FILE_ID}.bin",
            ]
            if feature_files is None
            else feature_files
        )

        self.feature_file_count = feature_file_count

        self.root = root
        self.__load_dataset(root, global_params_file)

    def reduce(self, node_count):

        self.nodes.x_coordinates = self.nodes.x_coordinates[:node_count]
        self.nodes.y_coordinates = self.nodes.y_coordinates[:node_count]
        self.nodes.z_index = self.nodes.z_index[:node_count]

        for i in range(len(self.nodes.features)):
            self.nodes.features[i] = self.nodes.features[i][:node_count]

        i = 0
        subset_graph_connections_start = self.connection_count
        subset_graph_connections_end = 0

        while self.connections.froms[i] < node_count:

            if (
                (self.connections.tos[i] >= 0)
                and (self.connections.tos[i] < node_count)
                and (self.connections.froms[i] != self.connections.tos[i])
            ):

                subset_graph_connections_start = min(subset_graph_connections_start, i)
                subset_graph_connections_end = max(subset_graph_connections_end, i)

            i += 1
        
        self.connections.froms = self.connections.froms[subset_graph_connections_start:subset_graph_connections_end]
        self.connections.tos = self.connections.tos[subset_graph_connections_start:subset_graph_connections_end]
        self.connections.values = self.connections.values[subset_graph_connections_start:subset_graph_connections_end]
        self.connections.x_coordinates = self.connections.x_coordinates[subset_graph_connections_start:subset_graph_connections_end]
        self.connections.y_coordinates = self.connections.y_coordinates[subset_graph_connections_start:subset_graph_connections_end]
        self.connections.z_index = self.connections.z_index[subset_graph_connections_start:subset_graph_connections_end]

        for i in range(len(self.connections.features)):
            self.connections.features[i] = self.connections.features[i][subset_graph_connections_start:subset_graph_connections_end]


    @property
    def node_degree(self):
        return self.nodes.features[0]

    @property
    def node_centrality(self):
        return self.nodes.features[1]

    @property
    def node_number_of_trades(self):
        return self.nodes.features[2]

    @property
    def node_profits(self):
        return self.nodes.features[3]

    @property
    def node_profits_excess(self):
        return self.nodes.features[4]

    @property
    def node_volume(self):
        return self.nodes.features[5]

    def __load_dataset(self, root: str, global_params_file: str):

        with open(root + "/" + global_params_file, "r") as f:
            params = json.load(f)

        nodes = params["nodes"]
        connections = params["links"]

        self.nodes.x_coordinates = np.ndarray(
            [nodes], dtype=self.dtypes["nodes"]["x_coordinates"]
        )
        self.nodes.y_coordinates = np.ndarray(
            [nodes], dtype=self.dtypes["nodes"]["y_coordinates"]
        )
        self.nodes.z_index = np.ndarray([nodes], dtype=self.dtypes["nodes"]["z_index"])

        for i in range(len(self.nodes.features)):
            self.nodes.features[i] = np.ndarray(
                [nodes], dtype=self.dtypes["nodes"]["features"][i]
            )

        self.connections.froms = np.ndarray(
            [connections], dtype=self.dtypes["connections"]["froms"]
        )
        self.connections.tos = np.ndarray(
            [connections], dtype=self.dtypes["connections"]["tos"]
        )
        self.connections.values = np.ndarray(
            [connections], dtype=self.dtypes["connections"]["values"]
        )
        self.connections.x_coordinates = np.ndarray(
            [connections], dtype=self.dtypes["connections"]["x_coordinates"]
        )
        self.connections.y_coordinates = np.ndarray(
            [connections], dtype=self.dtypes["connections"]["y_coordinates"]
        )
        self.connections.z_index = np.ndarray(
            [connections], dtype=self.dtypes["connections"]["z_index"]
        )

        for i in range(len(self.connections.features)):
            self.connections.features[i] = np.ndarray(
                [connections], dtype=self.dtypes["connections"]["features"][i]
            )

        self.__load_ein_bins(root)
        self.__load_features(root)

    def __load_ein_bins(
        self,
        ein_folder: str,
        from_name: str = "ein_from_${FILE_ID}.bin",
        to_name: str = "ein_to_${FILE_ID}.bin",
        value_name: str = "ein_value_${FILE_ID}.bin",
        file_counts=(2, 2, 1),
    ):
        read(ein_folder + "/" + from_name, file_counts[0], self.connections.froms)
        read(ein_folder + "/" + to_name, file_counts[1], self.connections.tos)
        read(ein_folder + "/" + value_name, file_counts[2], self.connections.values)

    def __load_features(self, root: str):

        for i in range(len(self.nodes.features)):
            read(
                root + "/" + self.feature_files[i],
                self.feature_file_count,
                self.nodes.features[i],
            )

    def __repr__(self) -> str:
        return (
            "IstanbulEinDatasetBin(with "
            + self.nodes.__repr__()
            + " and "
            + self.connections.__repr__()
            + ")"
        )
