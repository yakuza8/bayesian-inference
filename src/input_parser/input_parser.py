import itertools
import json
from typing import TextIO, List

from ..entity.network_node import NetworkNode
from ..exceptions.exceptions import IncompleteNodeDataException, PredecessorNotExistInNetwork, \
    NotAllExpectedProbabilityExist, HaveAtLeastOneRandomVariable


class InputParser(object):
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
            node = InputParser.validate_and_parse_node(node_name, network)
            parsed_nodes.append(node)

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
        InputParser._assert_all_probabilities_exist(random_variables=random_variables, probabilities=probabilities,
                                                    predecessors=predecessors, network=network,
                                                    all_random_variables=all_random_variables)

        return NetworkNode(node_name=node_name, random_variables=random_variables, predecessors=predecessors,
                           probabilities=probabilities, all_random_variables=all_random_variables)

    @staticmethod
    def _assert_essential_fields_exist(node_name: str, node_data: dict):
        # Check essential fields exist
        for field in InputParser.ESSENTIAL_FIELDS:
            if field not in node_data:
                raise IncompleteNodeDataException(
                    f'Check node {node_name}, it lacks of {field} field.')

        if len(node_data[InputParser.RANDOM_VARIABLES_TOKEN]) == 0:
            raise HaveAtLeastOneRandomVariable(f'Node {node_name} should have at least one random variable.')

    @staticmethod
    def _assert_all_predecessors_exist(node_data: dict, network: dict):
        # Check all predecessors exist
        predecessors = node_data[InputParser.PREDECESSORS_TOKEN]
        for predecessor in predecessors:
            if predecessor not in network:
                raise PredecessorNotExistInNetwork(f'No predecessor {predecessor} exist in network.')

    @staticmethod
    def _assert_all_probabilities_exist(random_variables: list, probabilities: dict, predecessors: list, network: dict,
                                        all_random_variables: list):
        # Iterate over each combination so that verify all of them exist
        for combination in itertools.product(*all_random_variables):
            key = '(' + ','.join(str(v) for v in combination) + ')'
            if key not in probabilities:
                raise NotAllExpectedProbabilityExist(f'Expected probability {key} not exist among probabilities.')


if __name__ == '__main__':
    with open('../../sample_data/network_1.json', 'r') as inp:
        nodes = InputParser.parse(inp)
        for node in nodes:
            print(node)
            print()
