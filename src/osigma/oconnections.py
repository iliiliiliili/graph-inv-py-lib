from typing import List
import numpy as np


class OConnections:

    def __init__(
        self,
        froms: np.ndarray,
        tos: np.ndarray,
        values: np.ndarray,
        features: List[np.ndarray],
    ) -> None:
        self.froms = froms
        self.tos = tos
        self.values = values
        self.features = features

    @property
    def features_dtypes(self):
        return [a.dtype for a in self.features]

    def __repr__(self) -> str:
        return (
            "OConnections(froms, tos, values + "
            + str(len(self.features))
            + " features of "
            + str(len(self.froms))
            + " nodes)"
        )


class OSpatialConnections(OConnections):

    def __init__(
        self,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        z_index: np.ndarray,
        froms: np.ndarray,
        tos: np.ndarray,
        values: np.ndarray,
        features: List[np.ndarray],
    ) -> None:
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_index = z_index
        super().__init__(froms, tos, values, features)

    def __repr__(self) -> str:
        return (
            "OSpatialConnections(x, y, froms, tos, values + "
            + str(len(self.features))
            + " features of "
            + str(len(self.froms))
            + " nodes)"
        )
