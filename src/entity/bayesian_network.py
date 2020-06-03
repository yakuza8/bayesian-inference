import logging
from collections import defaultdict
from typing import List

import networkx as nx

from .network_node import NetworkNode

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)-8s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class BayesianNetwork(object):
    """
    Bayesian Network class where it keeps Directed Acyclic Graph in it

    .. note::
        In constructor, it can be constructed with initial network node list
        where each node is added by calling single add function of class
    """

    def __init__(self, initial_network: List[NetworkNode]):
        # Directed graph
        self.G = nx.DiGraph()
        # Nodes
        self.nodes = {}
        # Container to keep edges which are not added since the predecessor does not exist
        self.edges_to_add = defaultdict(list)

        for node in initial_network:
            self.add_node(node)

    def add_node(self, node: NetworkNode) -> None:
        """
        Procedure of adding network node into the graph
        *   Node is added to nodes dictionary in network and node name itself is added to directed acyclic graph
        *   For each existing predecessor, the edge between them is linked and for the others they are kept in another
            dict to be added later when that predecessor is available
        *   If the node is expected to be linked to previously added edges, link them at the end
        :param node: Network node instance to be added into graph
        :return: None
        """
        # Add node
        node_key = node.node_name
        self.G.add_node(node_key)

        # Check acyclic condition of graph
        has_cycle = self._guarantee_graph_has_no_cycle(node_key)
        if has_cycle:
            logging.warning(f'{node_key} cannot be added since acyclic condition does not hold.')
            return
        else:
            logging.debug(f'{node_key} is successfully added without violation of acyclic state of graph.')

        self.nodes[node_key] = node

        # Link edges
        for predecessor in node.predecessors:
            # If predecessor not exist yet, then mark that predecessor to be added later
            if predecessor not in self.G:
                self.edges_to_add[predecessor].append(node_key)
            else:
                self.G.add_edge(predecessor, node_key)

        # Link edges of the node which could not be added before
        for successor in self.edges_to_add[node_key]:
            self.G.add_edge(node_key, successor)
        del self.edges_to_add[node_key]

    def remove_node(self, node_name: str) -> None:
        """
        Procedure of removal node from the network where it is removed from node dict and graph, and removed expected
        link lists
        :param node_name: Node name to refer node itself
        :return: None
        """
        if node_name in self.nodes:
            del self.nodes[node_name]

        for edge_to_add_later_list in self.edges_to_add.keys():
            if node_name in edge_to_add_later_list:
                edge_to_add_later_list.remove(node_name)

        if node_name in self.G:
            self.G.remove_node(node_name)

    def _guarantee_graph_has_no_cycle(self, node_key: str) -> bool:
        """ Guaranteeing having no cycle in the graph where if so remove the node otherwise continue """
        try:
            # Check if any cycle exists
            nx.find_cycle(self.G)
            raise nx.HasACycle
        except nx.NetworkXNoCycle:
            return False
        except nx.HasACycle:
            self.G.remove_node(node_key)
            return True
