from typing import List
import numpy as np

class ONodes:

    def __init__(self, features: List[np.ndarray]) -> None:
        self.features = features

    @property
    def features_dtypes(self):
        return [a.dtype for a in self.features]

    def __repr__(self) -> str:
        return (
            "ONodes("
            + str(len(self.features))
            + " features of "
            + str(len(self.features[0] if len(self.features) > 0 else 0))
            + " nodes)"
        )


class OSpatialNodes(ONodes):

    def __init__(
        self,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        z_index: np.ndarray,
        features: List[np.ndarray],
    ) -> None:
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_index = z_index
        super().__init__(features)

    def __repr__(self) -> str:
        return (
            "OSpatialNodes(x, y + "
            + str(len(self.features))
            + " features of "
            + str(len(self.x_coordinates))
            + " nodes)"
        )
