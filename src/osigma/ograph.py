import numpy as np
from .onodes import OSpatialNodes
from .oconnections import OSpatialConnections

class OGraph:
    def __init__(self, nodes: OSpatialNodes, connections: OSpatialConnections) -> None:
        self.nodes = nodes
        self.connections = connections

    @property
    def node_count(self):
        return len(self.nodes.x_coordinates)
    
    @property
    def connection_count(self):
        return len(self.connections.froms)

    def __repr__(self) -> str:
        return "OGraph(with " + self.nodes.__repr__() + " and " + self.connections.__repr__() + ")"