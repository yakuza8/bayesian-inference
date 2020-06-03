import itertools
import json
from typing import TextIO, List

from ..entity.bayesian_network import BayesianNetwork
from ..entity.network_node import NetworkNode
from ..exceptions.exceptions import IncompleteNodeDataException, PredecessorNotExistInNetwork, \
    NotAllExpectedProbabilityExist, HaveAtLeastOneRandomVariable


class InputParser(object):
    """
    Input parsing class where network is expected to be read from a file in JSON format

    .. note::
        In addition to parsing, validation of node and the network is also applied
        where the applied validations are as follows:
            - All essential fields exist
            - All predecessors exist
            - All probabilities have the complete set of keys to be expected with combined
             with their predecessor

    .. warning::
        After validations, corresponding exceptions are thrown
    """
    PREDECESSORS_TOKEN = 'predecessors'
    RANDOM_VARIABLES_TOKEN = 'random_variables'
    PROBABILITIES_TOKEN = 'probabilities'

    ESSENTIAL_FIELDS = [PREDECESSORS_TOKEN, RANDOM_VARIABLES_TOKEN, PROBABILITIES_TOKEN]

    @staticmethod
    def parse(file: TextIO) -> List[NetworkNode]:
        # Read data
        network = json.load(file)

        parsed_nodes = []
        for node_name in network:
            parsed_node = InputParser.validate_and_parse_node(node_name, network)
            parsed_nodes.append(parsed_node)

        return parsed_nodes

    @staticmethod
    def validate_and_parse_node(node_name: str, network: dict) -> NetworkNode:
        # Get node related data
        node_data: dict = network[node_name]
        random_variables: list = node_data[InputParser.RANDOM_VARIABLES_TOKEN]
        probabilities: dict = node_data[InputParser.PROBABILITIES_TOKEN]
        predecessors: list = node_data[InputParser.PREDECESSORS_TOKEN]
        all_random_variables: list = [network[predecessor][InputParser.RANDOM_VARIABLES_TOKEN] for predecessor in
                                      predecessors] + [random_variables]
        # Make probability keys to have proper form
        for key, value in probabilities.items():
            expected_form = key.replace(' ', '')
            if key != expected_form:
                probabilities[expected_form] = value
                del probabilities[key]

        # Make assertions
        InputParser._assert_essential_fields_exist(node_name=node_name, node_data=node_data)
        InputParser._assert_all_predecessors_exist(node_data=node_data, network=network)
        InputParser._assert_all_probabilities_exist(node_name=node_name, probabilities=probabilities,
                                                    all_random_variables=all_random_variables)

        return NetworkNode(node_name=node_name, random_variables=random_variables, predecessors=predecessors,
                           probabilities=probabilities, all_random_variables=all_random_variables)

    @staticmethod
    def _assert_essential_fields_exist(node_name: str, node_data: dict) -> None:
        """
        Checking essential fields where they are predecessors, probabilities and random variables
        :param node_name: Node name to refer in exception
        :param node_data: Node data to check essential fields in it
        :return: None
        :raises IncompleteNodeDataException: In case of having one of the essential fields
        :raises HaveAtLeastOneRandomVariable: In case of having no random variable
        """
        # Check essential fields exist
        for field in InputParser.ESSENTIAL_FIELDS:
            if field not in node_data:
                raise IncompleteNodeDataException(
                    f'Check node {node_name}, it lacks of {field} field.')

        if len(node_data[InputParser.RANDOM_VARIABLES_TOKEN]) == 0:
            raise HaveAtLeastOneRandomVariable(f'Node {node_name} should have at least one random variable.')

    @staticmethod
    def _assert_all_predecessors_exist(node_data: dict, network: dict) -> None:
        """
        Checking all the predecessors exist in the given network
        :param node_data: Node data to fetch predecessor list
        :param network: Whole network which is candidate to be parsed
        :return: None
        :raises PredecessorNotExistInNetwork: In case of not having one of the predecessors in the network
        """
        # Check all predecessors exist
        predecessors = node_data[InputParser.PREDECESSORS_TOKEN]
        for predecessor in predecessors:
            if predecessor not in network:
                raise PredecessorNotExistInNetwork(f'No predecessor {predecessor} exist in network.')

    @staticmethod
    def _assert_all_probabilities_exist(node_name: str, probabilities: dict, all_random_variables: list) -> None:
        """
        Checking whether all the expected probabilities exist in the given input
        :param node_name: Node name to refer in exception
        :param probabilities: Probabilities of the current node
        :param all_random_variables: All random variables as a list where predecessor and current node's variables
        are combined
        :return: None
        :raises NotAllExpectedProbabilityExist: In case of not having one of the expected probabilities in the input
        """
        # Iterate over each combination so that verify all of them exist
        for combination in itertools.product(*all_random_variables):
            key = '(' + ','.join(str(v) for v in combination) + ')'
            if key not in probabilities:
                raise NotAllExpectedProbabilityExist(
                    f'Expected probability {key} not exist among {node_name} probabilities.')


if __name__ == '__main__':
    with open('../../sample_data/network_1.json', 'r') as inp:
        nodes = InputParser.parse(inp)
        for node in nodes:
            print(node)
            print()
        net = BayesianNetwork(nodes)
