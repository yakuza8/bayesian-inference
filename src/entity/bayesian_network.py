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

    .. note:: In constructor, it can be constructed with initial network node list where each node
              is added by calling single add function of class
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

    def add_node(self, node: NetworkNode) -> bool:
        """
        Procedure of adding network node into the graph

        *   Node is added to nodes dictionary in network and node name itself is added to directed
            acyclic graph
        *   For each existing predecessor, the edge between them is linked and for the others they
            are kept in another dict to be added later when that predecessor is available
        *   If the node is expected to be linked to previously added edges, link them at the end

        :param node: Network node instance to be added into graph
        :return: Boolean flag whether node is successfully added or not
        """
        node_key = node.node_name

        # Already in network
        if node_key in self.nodes:
            return False

        # Check acyclic condition of graph
        has_cycle = self._guarantee_graph_has_no_cycle(node_key, node=node)
        if has_cycle:
            logging.warning(f'{node_key} cannot be added since acyclic condition does not hold.')
            return False
        else:
            logging.debug(
                f'{node_key} is successfully added without violation of acyclic state of graph.')

        # Add node
        self.nodes[node_key] = node
        self.G.add_node(node_key)
        self._add_predecessor_edges(node_key=node_key, node=node, target_graph=self.G)
        self._add_expected_edges_if_exist(node_key=node_key, target_graph=self.G)

        return True

    def _guarantee_graph_has_no_cycle(self, node_key: str, node: NetworkNode) -> bool:
        """
        Guaranteeing having no cycle in the graph where if so remove the node otherwise continue

        .. note:: Changes are done on temporary graph copied from the actual graph. In order to keep
                  internal changes, we pass `update_internal_variables=False` while adding edges

        :param node_key: Node name to refer the node
        :param node: Node instance to pass edge adding methods
        """
        try:
            # Check if any cycle exists
            copied_graph = self.G.copy()
            self._add_predecessor_edges(node_key=node_key, node=node, target_graph=copied_graph,
                                        update_internal_variables=False)
            self._add_expected_edges_if_exist(node_key=node_key, target_graph=copied_graph,
                                              update_internal_variables=False)
            nx.find_cycle(copied_graph)
        except nx.NetworkXNoCycle:
            return False
        return True

    def _add_predecessor_edges(self, node_key: str, node: NetworkNode, target_graph: nx.DiGraph,
                               update_internal_variables: bool = True):
        """
        Linking predecessors' related edges with the given node.

        .. note:: In case of not having any predecessor in the graph, then it is added to be added
                  later on when encountered that parent

        .. note:: Adding an edge to self is prevented

        :param node_key: Node name to refer the node
        :param node: Node instance to get predecessors
        :param target_graph: Target graph on which updates will be done
        :param update_internal_variables: Boolean flag whether to do update on internal variables
            such as `edges_to_add` mapping and logging
        """
        for predecessor in node.predecessors:
            # If predecessor not exist yet, then mark that predecessor to be added later
            if predecessor not in target_graph:
                # Just update internal variables if it is enabled
                if update_internal_variables:
                    self.edges_to_add[predecessor].append(node_key)
            else:
                # Predecessor is itself which results in cycle no need to add
                if predecessor == node_key:
                    if update_internal_variables:
                        logging.warning(f'{node_key} has itself as predecessor.')
                    continue
                target_graph.add_edge(predecessor, node_key)

    def _add_expected_edges_if_exist(self, node_key: str, target_graph: nx.DiGraph,
                                     update_internal_variables: bool = True):
        """
        Link edges of the node which could not be added before
        :param node_key: Node name to refer the node
        :param target_graph: Target graph on which updates will be done
        :param update_internal_variables: Boolean flag whether to do update on internal variables
            such as `edges_to_add` mapping
        """
        if node_key in self.edges_to_add:
            for successor in self.edges_to_add[node_key]:
                target_graph.add_edge(node_key, successor)
            if update_internal_variables:
                del self.edges_to_add[node_key]

    def remove_node(self, node_name: str) -> bool:
        """
        Procedure of removal node from the network where it is removed from node dict and graph,
        and removed expected link lists
        :param node_name: Node name to refer node itself
        :return: Boolean flag whether the node name found and removed successfully from network
        """
        if node_name in self.nodes:
            del self.nodes[node_name]
        else:
            logging.debug(f'{node_name} does not exist in the network.')
            return False

        for edge_to_add_later_list in self.edges_to_add.values():
            if node_name in edge_to_add_later_list:
                edge_to_add_later_list.remove(node_name)

        if node_name in self.G:
            self.G.remove_node(node_name)

        logging.debug(f'{node_name} is successfully removed from the network.')
        return True
