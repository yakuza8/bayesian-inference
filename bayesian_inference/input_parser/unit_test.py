from typing import List
from unittest import TestCase
from unittest.mock import patch

from .input_parser import InputParser
from ..entity.network_node import NetworkNode
from ..exceptions.exceptions import (
    IncompleteNodeDataException, HaveAtLeastOneRandomVariable, NotAllExpectedProbabilityExist,
    PredecessorNotExistInNetwork,
)

__all__ = []


class TestInputParser(TestCase):
    _node_name = "marvelous_node"

    def test_essential_field_assertion_fields(self):
        for field in InputParser.ESSENTIAL_FIELDS:
            sample_node_data = {_: "" for _ in InputParser.ESSENTIAL_FIELDS}

            # Remove current item
            del sample_node_data[field]

            with self.assertRaises(IncompleteNodeDataException) as e:
                InputParser._assert_essential_fields_exist(self._node_name, sample_node_data)
            self.assertTrue(field in str(e.exception))

    def test_essential_field_assertion_random_variable_count(self):
        sample_node_data = {_: "" for _ in InputParser.ESSENTIAL_FIELDS}

        with self.assertRaises(HaveAtLeastOneRandomVariable):
            InputParser._assert_essential_fields_exist(self._node_name, sample_node_data)

    def test_essential_field_assertion_valid_data(self):
        sample_node_data = {_: (1, 2, 3) for _ in InputParser.ESSENTIAL_FIELDS}
        try:
            InputParser._assert_essential_fields_exist(self._node_name, sample_node_data)
        except (IncompleteNodeDataException, HaveAtLeastOneRandomVariable):
            self.fail('Unexcepted exception occurred.')

    def test_all_predecessors_exist_lack_predecessor(self):
        predecessor_a = 'A'
        predecessor_b = 'B'

        sample_network = {
            self._node_name: {
                InputParser.PREDECESSORS_TOKEN: [predecessor_a, predecessor_b]
            }, predecessor_a: {},
        }

        with self.assertRaises(PredecessorNotExistInNetwork) as e:
            InputParser._assert_all_predecessors_exist(sample_network[self._node_name],
                                                       sample_network)
        self.assertTrue(predecessor_b in str(e.exception))

    def test_all_predecessors_exist_valid_data(self):
        predecessor_a = 'A'
        predecessor_b = 'B'

        sample_network = {
            self._node_name: {
                InputParser.PREDECESSORS_TOKEN: [predecessor_a, predecessor_b]
            }, predecessor_a: {}, predecessor_b: {},
        }
        try:
            InputParser._assert_all_predecessors_exist(sample_network[self._node_name],
                                                       sample_network)
        except PredecessorNotExistInNetwork:
            self.fail('Unexcepted exception occurred.')

    def test_all_probabilities_exist_lack_probability(self):
        all_random_variables = [['0', '1'], ['1', '2', '3']]
        sample_probability = {
            '(0,1)': 0.5, '(0,2)': 0.5, '(0,3)': 0.5, '(1,2)': 0.5, '(1,3)': 0.5,
        }

        with self.assertRaises(NotAllExpectedProbabilityExist) as e:
            InputParser._assert_all_probabilities_exist(self._node_name, sample_probability,
                                                        all_random_variables)
        self.assertTrue('(1,1)' in str(e.exception))

    def test_all_probabilities_exist_with_wrong_key(self):
        all_random_variables = [['0', '1'], ['1', '2', '3']]
        sample_probability = {
            '(0,1)': 0.5, '(0,2)': 0.5, '(0,3)': 0.5, '(1, 1)': 0.5,  # Here the key is wrong
            '(1,2)': 0.5, '(1,3)': 0.5,
        }

        with self.assertRaises(NotAllExpectedProbabilityExist) as e:
            InputParser._assert_all_probabilities_exist(self._node_name, sample_probability,
                                                        all_random_variables)
        self.assertTrue('(1,1)' in str(e.exception))

    def test_all_probabilities_exist_valid_data(self):
        all_random_variables = [['0', '1'], ['1', '2', '3']]
        sample_probability = {
            '(0,1)': 0.5, '(0,2)': 0.5, '(0,3)': 0.5, '(1,1)': 0.5, '(1,2)': 0.5, '(1,3)': 0.5,
        }
        try:
            InputParser._assert_all_probabilities_exist(self._node_name, sample_probability,
                                                        all_random_variables)
        except NotAllExpectedProbabilityExist:
            self.fail('Un-excepted exception occurred.')

    @patch('json.load')
    def test_parse_network(self, mock_load):
        sample_network = {
            'D': {
                'predecessors': [], 'random_variables': ['0', '1'], 'probabilities': {
                    '(0)': 0.6, '(1)': 0.4
                }
            }, 'I': {
                'predecessors': [], 'random_variables': ['0', '1'], 'probabilities': {
                    '(0)': 0.7, '(1)': 0.3
                }
            }, 'G': {
                'predecessors': ['D', 'I'], 'random_variables': ['1', '2', '3'], 'probabilities': {
                    '(0, 0, 1)': 0.3, '(0, 0, 2)': 0.7, '(0, 0, 3)': 0.3, '(0, 1, 1)': 0.05,
                    '(0, 1, 2)': 0.25, '(0, 1, 3)': 0.7, '(1, 0, 1)': 0.9, '(1, 0, 2)': 0.08,
                    '(1, 0, 3)': 0.02, '(1, 1, 1)': 0.5, '(1, 1, 2)': 0.3, '(1, 1, 3)': 0.2
                }
            }, 'K': {
                'predecessors': [], 'random_variables': ['A', 'B', 'C'], 'probabilities': {
                    '(A)': 0.586, '(B)': 0.413, '(C)': 0.001
                }
            }
        }
        mock_load.return_value = sample_network

        # Test begins
        network_nodes: List[NetworkNode] = InputParser.parse(None)
        self.assertEqual(4, len(network_nodes))

        for index, node_name in enumerate(sample_network.keys()):
            parsed_node = network_nodes[index]
            actual_value = sample_network[node_name]

            self.assertEqual(node_name, parsed_node.node_name)
            self.assertListEqual(actual_value[InputParser.PREDECESSORS_TOKEN],
                                 parsed_node.predecessors)
            self.assertListEqual(actual_value[InputParser.RANDOM_VARIABLES_TOKEN],
                                 parsed_node.random_variables)
            self.assertListEqual(actual_value[InputParser.PREDECESSORS_TOKEN],
                                 parsed_node.predecessors)
            self.assertEqual(len(actual_value[InputParser.PROBABILITIES_TOKEN]), len(parsed_node.probabilities))
            self.assertEqual(len(actual_value[InputParser.PREDECESSORS_TOKEN]) + 1,
                             len(parsed_node.all_random_variables))
