import unittest

from .input_parser import InputParser

from ..exceptions.exceptions import (
    IncompleteNodeDataException,
    HaveAtLeastOneRandomVariable,
    NotAllExpectedProbabilityExist,
    PredecessorNotExistInNetwork
)


class TestInputParser(unittest.TestCase):
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
            },
            predecessor_a: {},
        }

        with self.assertRaises(PredecessorNotExistInNetwork) as e:
            InputParser._assert_all_predecessors_exist(sample_network[self._node_name], sample_network)
        self.assertTrue(predecessor_b in str(e.exception))

    def test_all_predecessors_exist_valid_data(self):
        predecessor_a = 'A'
        predecessor_b = 'B'

        sample_network = {
            self._node_name: {
                InputParser.PREDECESSORS_TOKEN: [predecessor_a, predecessor_b]
            },
            predecessor_a: {},
            predecessor_b: {},
        }
        try:
            InputParser._assert_all_predecessors_exist(sample_network[self._node_name], sample_network)
        except PredecessorNotExistInNetwork:
            self.fail('Unexcepted exception occurred.')

    def test_all_probabilities_exist_lack_probability(self):
        all_random_variables = [['0', '1'], ['1', '2', '3']]
        sample_probability = {
            '(0,1)': 0.5,
            '(0,2)': 0.5,
            '(0,3)': 0.5,
            '(1,2)': 0.5,
            '(1,3)': 0.5,
        }

        with self.assertRaises(NotAllExpectedProbabilityExist) as e:
            InputParser._assert_all_probabilities_exist(self._node_name, sample_probability, all_random_variables)
        self.assertTrue('(1,1)' in str(e.exception))

    def test_all_probabilities_exist_with_wrong_key(self):
        all_random_variables = [['0', '1'], ['1', '2', '3']]
        sample_probability = {
            '(0,1)': 0.5,
            '(0,2)': 0.5,
            '(0,3)': 0.5,
            '(1, 1)': 0.5,  # Here the key is wrong
            '(1,2)': 0.5,
            '(1,3)': 0.5,
        }

        with self.assertRaises(NotAllExpectedProbabilityExist) as e:
            InputParser._assert_all_probabilities_exist(self._node_name, sample_probability, all_random_variables)
        self.assertTrue('(1,1)' in str(e.exception))

    def test_all_probabilities_exist_valid_data(self):
        all_random_variables = [['0', '1'], ['1', '2', '3']]
        sample_probability = {
            '(0,1)': 0.5,
            '(0,2)': 0.5,
            '(0,3)': 0.5,
            '(1,1)': 0.5,
            '(1,2)': 0.5,
            '(1,3)': 0.5,
        }
        try:
            InputParser._assert_all_probabilities_exist(self._node_name, sample_probability, all_random_variables)
        except NotAllExpectedProbabilityExist:
            self.fail('Unexcepted exception occurred.')


if __name__ == '__main__':
    unittest.main()
