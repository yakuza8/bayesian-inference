import itertools
from typing import List, Dict, Tuple
from tabulate import tabulate


class NetworkNode(object):
    """
    Bayesian network node which have random variable list, predecessor list and probability table in it
    """

    def __init__(self, node_name: str, random_variables: List[str], predecessors: List[str],
                 probabilities: Dict[str, float],
                 all_random_variables: List[List[str]]):
        self._node_name = node_name
        self._random_variables = random_variables
        self._predecessors = predecessors
        self._probabilities = probabilities
        self._all_random_variables = all_random_variables

    def __repr__(self):
        return 'NetworkNode({!r}, {!r}, {!r}, {!r}, {!r})'.format(self.node_name, self.random_variables,
                                                                  self.predecessors, self.probabilities,
                                                                  self.all_random_variables)

    def __str__(self):
        """ Table representation of the probabilities with predecessors' and self random variables"""

        def probability_key(dict_key: Tuple[str]):
            return '(' + ','.join(str(v) for v in dict_key) + ')'

        # Get all combinations of keys for probabilities
        all_combinations = list(itertools.product(*self.all_random_variables))
        count_of_rv = len(self.random_variables)
        grouped_combinations = [all_combinations[i * count_of_rv:(i + 1) * count_of_rv] for i in
                                range(int(len(all_combinations) / count_of_rv))]
        # Create header for each predecessor and random variables
        headers = self.predecessors + [f'P({self.node_name}={variable})' for variable in self.random_variables]
        # Row data where predecessor random variables are changing and each probability is inserted
        rows = [list(group[0][:-1]) + [self.probabilities[probability_key(key)] for key in group] for group in
                grouped_combinations]
        return tabulate(tabular_data=rows, headers=headers, tablefmt='github')

    def __hash__(self):
        """ Hash method to be used in network graph"""
        return hash(self.node_name)

    @property
    def node_name(self):
        return self._node_name

    @property
    def random_variables(self):
        return self._random_variables

    @property
    def predecessors(self):
        return self._predecessors

    @property
    def probabilities(self):
        return self._probabilities

    @property
    def all_random_variables(self):
        return self._all_random_variables
