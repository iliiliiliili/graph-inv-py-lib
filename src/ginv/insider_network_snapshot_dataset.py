from typing import List
import numpy as np
import json
from enum import Enum

from osigma.oconnections import OSpatialConnections
from osigma.ograph import OGraph
from osigma.onodes import OSpatialNodes


class SnapshotAggregation(Enum):
    CONNECTED_ONCE = "once"
    CONNECTED_HALF_DAYS = "half_days"
    CONNECTED_MORE_THAN_AVERAGE = "more_than_average"


def read(file_name: str, dtype):
    current_data = np.fromfile(file_name, dtype=dtype)
    print(f"loaded {file_name}")

    return current_data


class InsiderNetworkSnapshotDataset(OGraph):

    def __init__(
        self,
        root: str,
        day_ids: int | List[int],
        dtypes: dict | None = None,
        feature_files: List[str] | None = None,
        aggregation: SnapshotAggregation = SnapshotAggregation.CONNECTED_ONCE,
    ) -> None:
        super().__init__(
            OSpatialNodes(
                None, None, None, [None, None, None, None, None, None, None, None, None]
            ),
            OSpatialConnections(None, None, None, None, None, None, []),
        )

        self.dtypes = (
            {
                "nodes": {
                    "x_coordinates": np.float32,
                    "y_coordinates": np.float32,
                    "z_index": np.uint8,
                    "features": [
                        np.uint16,
                        np.uint16,
                        np.int32,
                        np.uint8,
                        np.uint32,
                        np.uint16,
                        np.uint8,
                        np.uint8,
                        np.uint8,
                    ],
                },
                "connections": {
                    "froms": np.uint16,
                    "tos": np.uint16,
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
                "year_of_birth.bin",
                "gender.bin",
                "sector_code.bin",
                "juridical_form.bin",
                "postal_code.bin",
                "birthday.bin",
                "country.bin",
                "domicile.bin",
                "language.bin",
            ]
            if feature_files is None
            else feature_files
        )

        self.root = root
        self.day_ids = (day_ids,)
        self.aggregation = aggregation
        self.__load_dataset(root, day_ids, aggregation)

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

        self.connections.froms = self.connections.froms[
            subset_graph_connections_start:subset_graph_connections_end
        ]
        self.connections.tos = self.connections.tos[
            subset_graph_connections_start:subset_graph_connections_end
        ]
        self.connections.values = self.connections.values[
            subset_graph_connections_start:subset_graph_connections_end
        ]
        self.connections.x_coordinates = self.connections.x_coordinates[
            subset_graph_connections_start:subset_graph_connections_end
        ]
        self.connections.y_coordinates = self.connections.y_coordinates[
            subset_graph_connections_start:subset_graph_connections_end
        ]
        self.connections.z_index = self.connections.z_index[
            subset_graph_connections_start:subset_graph_connections_end
        ]

        for i in range(len(self.connections.features)):
            self.connections.features[i] = self.connections.features[i][
                subset_graph_connections_start:subset_graph_connections_end
            ]

    @property
    def node_year_of_birth(self):
        return self.nodes.features[0]

    @property
    def node_gender(self):
        return self.nodes.features[1]

    @property
    def node_sector_code(self):
        return self.nodes.features[2]

    @property
    def node_juridical_form(self):
        return self.nodes.features[3]

    @property
    def node_postal_code(self):
        return self.nodes.features[4]

    @property
    def node_birthday(self):
        return self.nodes.features[5]

    @property
    def node_country(self):
        return self.nodes.features[6]

    @property
    def node_domicile(self):
        return self.nodes.features[7]

    @property
    def node_language(self):
        return self.nodes.features[8]

    def __load_dataset(self, root: str, day_ids: int | List[int], aggregation: SnapshotAggregation):

        self.__load_connections(root, day_ids, aggregation)
        self.__load_features(root)

        node_count = len(self.nodes.features[0])
        connection_count = self.connection_count

        self.nodes.x_coordinates = np.ndarray(
            [node_count], dtype=self.dtypes["nodes"]["x_coordinates"]
        )
        self.nodes.y_coordinates = np.ndarray(
            [node_count], dtype=self.dtypes["nodes"]["y_coordinates"]
        )
        self.nodes.z_index = np.ndarray(
            [node_count], dtype=self.dtypes["nodes"]["z_index"]
        )

        self.connections.x_coordinates = np.ndarray(
            [connection_count], dtype=self.dtypes["connections"]["x_coordinates"]
        )
        self.connections.y_coordinates = np.ndarray(
            [connection_count], dtype=self.dtypes["connections"]["y_coordinates"]
        )
        self.connections.z_index = np.ndarray(
            [connection_count], dtype=self.dtypes["connections"]["z_index"]
        )

    def __load_connections(
        self,
        root,
        day_ids: int | List[int],
        aggregation: SnapshotAggregation,
        from_name: str = "graph/froms_${DAY_ID}.bin",
        to_name: str = "graph/tos_${DAY_ID}.bin",
        value_name: str = None,
    ):

        all_froms = []
        all_tos = []
        all_values = []

        for day_id in day_ids:

            froms = read(
                root + "/" + from_name.replace("${DAY_ID}", str(day_id)),
                self.dtypes["connections"]["froms"],
            )
            tos = read(
                root + "/" + to_name.replace("${DAY_ID}", str(day_id)),
                self.dtypes["connections"]["tos"],
            )

            if value_name is not None:
                values = read(
                    root + "/" + value_name.replace("${DAY_ID}", str(day_id)),
                    self.dtypes["connections"]["values"],
                )
            else:
                values = np.ones_like(froms, dtype=self.dtypes["connections"]["values"])

            all_froms.append(froms)
            all_tos.append(tos)
            all_values.append(values)

        if len(all_froms) == 1:
            combined_froms = all_froms[0]
            combined_tos = all_tos[0]
            combined_values = all_values[0]
        else:

            found_connections = {}

            for day in range(len(day_ids)):
                day_connections = set()

                for i in range(len(all_froms[day])):


                    f = all_froms[day][i]
                    t = all_tos[day][i]

                    if f > t:
                        f, t = t, f

                    if (f, t) in day_connections:
                        continue

                    v = all_values[day][i]

                    if (f, t) in found_connections:
                        found_connections[(f, t)][0] += v
                        found_connections[(f, t)][1] += 1
                    else:
                        found_connections[(f, t)] = [v, 1]
                        day_connections.add((f, t))

            if aggregation == SnapshotAggregation.CONNECTED_ONCE:

                total_connections = len(found_connections)

                combined_froms = np.empty(
                    [total_connections], dtype=self.dtypes["connections"]["froms"]
                )
                combined_tos = np.empty(
                    [total_connections], dtype=self.dtypes["connections"]["tos"]
                )
                combined_values = np.empty(
                    [total_connections], dtype=self.dtypes["connections"]["values"]
                )

                for i, (k, v) in enumerate(found_connections.items()):
                    weight = v[0]
                    f, t = k

                    combined_froms[i] = f
                    combined_tos[i] = t
                    combined_values[i] = weight
                
            else:
                raise NotImplementedError()

        self.connections.froms = combined_froms
        self.connections.tos = combined_tos
        self.connections.values = combined_values

    def __load_features(self, root: str):

        for i in range(len(self.nodes.features)):
            self.nodes.features[i] = read(
                root + "/" + self.feature_files[i],
                self.dtypes["nodes"]["features"][i],
            )

    def __repr__(self) -> str:
        return (
            "InsiderNetworkSnapshotDataset(with "
            + self.nodes.__repr__()
            + " and "
            + self.connections.__repr__()
            + ")"
        )
