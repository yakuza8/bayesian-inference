import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from typing import List, Callable, Dict, Generator, Set, Iterable, Tuple, Union

import networkx as nx

from .network_node import NetworkNode
from ..exceptions.exceptions import InvalidQuery, InvalidProbabilityFactor
from ..probability.probability import query_parser, QueryVariable

__all__ = ['ProbabilityFactor', 'BayesianNetwork', 'P']

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)-8s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


@dataclass
class ProbabilityFactor:
    """
    Internally maintained class where it will be used for calculation of probability value in
    the network

    Use cases listed below:
        * value is None and sum_out = False     : Query variable feed by outer scope value
        * value is not None and sum_out = False : Known variable with value
        * value is None and sum_out = True      : Hidden variable which will be summed-out
        * value is not None and sum_out = True  : No meaning and exception
    """

    name: str
    value: str = None
    sum_out: bool = False


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

    def P(self, query: str) -> Union[float, Dict[str, float]]:
        """
        Exact probabilistic inference function that will be used for calculation of posterior
        probability on the given bayesian network context.

        .. note:: Nominator is returned as with multiple query variable combinations if there
            exist any query variable

        :param query: Query that will be evaluated with the network context
        :return: Exact inference probability of the query in the network
        :raises InvalidQuery: If query is not valid
        """
        is_parsed, queries, evidences = query_parser(query=query,
                                                     expected_symbol_and_values=self.symbol_context)
        # If not parsed, then raise error immediately
        if not is_parsed:
            raise InvalidQuery("Query does not hold for full match!")
        # Get nominator for different query variables and denominator for each evidence variable
        nominator_context = self._calculate_joint_probability(queries + evidences)
        denominator = self._calculate_joint_probability(evidences)
        # Calculate exact inferred probability of each query variable combination
        if type(nominator_context) == float:
            return nominator_context / denominator
        else:
            return {context: value / denominator for context, value in nominator_context.items()}

    def _calculate_joint_probability(self, variables: List[QueryVariable]) \
            -> Union[float, Dict[str, float]]:
        """
        Calculation of joint probability of the given variable set where it is made up of query and
        evidence variables

        Procedural steps:
            * Find needed variables, hidden variables
            * Decide order of calculation for the probability factors
            * For each query variable combination, calculate probability if exist

        :param variables: Variables composed from query and evidence variables
        :return: Single float if no query variable exist, otherwise dictionary with keys query
            variable contexts
        """
        # Set of variable names of query + evidence
        needed_variable_names = {v.name: v for v in variables}
        # Set of all the variables where query + evidence + hidden variables included
        purified_variables = self._eliminate_unnecessary_variables(
            variables=needed_variable_names.keys())
        # Hidden variables
        hidden_variables = {v for v in purified_variables if v not in needed_variable_names}

        order = self._decide_calculation_order(needed_variable_names=needed_variable_names,
                                               purified_variables=purified_variables,
                                               hidden_variables=hidden_variables)

        # Find all query variables
        query_variable_names = [variable.name for variable in variables if variable.value is None]
        query_variable_values = [self.nodes[variable_name].random_variables for variable_name in
                                 query_variable_names]

        # If any exists, calculate probability for each combination of query variables
        if query_variable_names:
            return_context_probability = {}
            for combination in product(*query_variable_values):
                context = dict(zip(query_variable_names, combination))
                p = self._probability_inference(tuple(order), **context)
                return_context_probability[str(context)] = p
            return return_context_probability
        else:
            return self._probability_inference(tuple(order))

    def _probability_inference(self, calculation_order: Tuple[ProbabilityFactor], tab_stop: int = 0,
                               **context) -> float:
        """
        Probability calculation of defined calculation order with the given initial context for the
        exact bayesian inference with the below formulation:

        .. math::
             S_{PS}(x_{1}...x_{n}) = \prod_{i=1}^{n}P(X_{i} | parents(X_{i})

        Recursive function where

        * Base Case -> returning 1.0 if no factor exist
        * Otherwise -> fetches first factor from the list
            * If value is query variable -> Find its probability and multiplies with rest of factors
            * If value is hidden variable -> Sum out all possible values multiply with other factors
            * If value is known variable -> Find its probability and multiplies with rest of factors
            * Otherwise, raises exception since factor does not represent anything

        :param calculation_order: Predefined order of factors to be calculated of full-joint
            probability
        :return: Calculated probability of the given factors
        :raises InvalidProbabilityFactor: When the current factor has value and needs sum-out
        """
        if len(calculation_order) == 0:
            # If no factor is found, then return 1.0
            base_probability = 1.0
            logging.debug(
                '\t' * tab_stop + f'Hit base case: {base_probability} with context: {context}')
            return base_probability
        else:
            first_factor = calculation_order[0]
            node_name = first_factor.name
            node = self.nodes[node_name]

            if first_factor.value is None and not first_factor.sum_out:
                # Query variable - Value should be fetched from context
                probability = node.probability(**context)
                logging.debug(
                    '\t' * tab_stop + f'Query variable {node_name} : {probability} with context: '
                                      f'{context}')
                return probability * self._probability_inference(calculation_order[1:],
                                                                 tab_stop=tab_stop + 1, **context)
            elif first_factor.value is None and first_factor.sum_out:
                # Hidden variable
                logging.debug(
                    '\t' * tab_stop + f'Hidden variable {node_name} with context: {context}')
                return sum(self._probability_inference(
                    (ProbabilityFactor(name=node_name, value=v),) + calculation_order[1:],
                    tab_stop=tab_stop + 1, **context) for v in node.random_variables)
            elif first_factor.value is not None and not first_factor.sum_out:
                # Known variable
                new_context = context.copy()
                new_context[node_name] = first_factor.value
                probability = node.probability(**new_context)
                logging.debug(
                    '\t' * tab_stop + f'Known variable {node_name} : {probability} with context: '
                                      f'{new_context}')
                return probability * self._probability_inference(calculation_order[1:],
                                                                 tab_stop=tab_stop + 1,
                                                                 **new_context)
            else:
                error_message = f'Unexpected probability factor for {node_name}!'
                logging.error(error_message)
                raise InvalidProbabilityFactor(error_message)

    @property
    def symbol_context(self) -> Dict[str, List[str]]:
        """
        Utility to get nodes and their random variables in dictionary format
        """
        return {node_name: node.random_variables for node_name, node in self.nodes.items()}

    @property
    def network_topology(self) -> Generator[str, str, str]:
        return nx.topological_sort(self.G)

    def _decide_calculation_order(self, needed_variable_names: Dict[str, QueryVariable],
                                  purified_variables: Set[str],
                                  hidden_variables: Set[str]) -> List[ProbabilityFactor]:
        """
        Functionality for deciding calculation order of probability where variables

        :param needed_variable_names: Definitely needed variables composed of query and evidence
            variables
        :param purified_variables: Set of all the variables constructed from the needed_variables
            by fetching their parents recursively
        :param hidden_variables: Hidden variables filtered from purified variables
        :return: Ordered state of variables
        """
        topology_order = {node: position for position, node in enumerate(self.network_topology)}
        calculation_order = [ProbabilityFactor(name=variable, value=needed_variable_names[
            variable].value if variable in needed_variable_names else None,
                                               sum_out=variable in hidden_variables, ) for variable
                             in sorted(list(purified_variables),
                                       key=lambda node: topology_order[node])]
        return calculation_order

    def _eliminate_unnecessary_variables(self, variables: Iterable[str]) -> Set[str]:
        """
        Breadth-first traversal of variables with recursive predecessors for elimination of
        unnecessary variables for the sake of optimization

        :param variables: Set of variables which are necessarily needed
        :return: Set of variables where parameters and their recursive parents are added
        """
        logging.debug(f"Variable elimination with the given variable set: {variables}")
        set_of_needed_variables = set()
        for node in reversed(list(self.network_topology)):
            if node in variables:
                # Breadth-first strategy for iterating recursive predecessors
                traverse_queue = deque([node])
                while traverse_queue:
                    variable = traverse_queue.pop()
                    if variable not in set_of_needed_variables:
                        set_of_needed_variables.add(variable)
                        traverse_queue.extend(self.nodes[variable].predecessors)
        logging.debug(f"Extracted necessary variables: {set_of_needed_variables}")
        return set_of_needed_variables


P: Callable[[BayesianNetwork, str], float] = lambda network, query: network.P(query=query)
