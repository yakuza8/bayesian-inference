import itertools
from unittest import TestCase, mock

from .bayesian_network import BayesianNetwork, ProbabilityFactor
from .network_node import NetworkNode
from ..exceptions.exceptions import InvalidProbabilityFactor
from ..probability.probability import QueryVariable


class TestNetworkNode(TestCase):

    def test_network_node_properties(self):
        node_name = 'marvelous_node'
        random_variables = ['0', '1']
        predecessors = []
        probabilities = {
            '(0)': 0.7, '(1)': 0.3
        }
        all_random_variables = [['0', '1']]

        # Prepare node
        node = NetworkNode(node_name=node_name, random_variables=random_variables,
                           predecessors=predecessors, probabilities=probabilities,
                           all_random_variables=all_random_variables)

        # Make assertions
        self.assertEqual(node_name, node.node_name)
        self.assertListEqual(random_variables, node.random_variables)
        self.assertListEqual(predecessors, node.predecessors)
        self.assertEqual(probabilities, node.probabilities)
        self.assertEqual(all_random_variables, node.all_random_variables)

        # Probability
        self.assertEqual(0.7, node.probability(marvelous_node='0'))
        self.assertEqual(0.3, node.probability(marvelous_node='1'))

        # Representation
        repr_node = eval(repr(node))
        self.assertEqual(node_name, repr_node.node_name)
        self.assertListEqual(random_variables, repr_node.random_variables)
        self.assertListEqual(predecessors, repr_node.predecessors)
        self.assertEqual(probabilities, repr_node.probabilities)
        self.assertEqual(all_random_variables, repr_node.all_random_variables)

        # Hash
        self.assertEqual(hash(node_name), hash(node))

        # String representation as table
        table = str(node)
        self.assertTrue(node_name in table)
        self.assertTrue(all(random_variable in table for random_variable in random_variables))
        self.assertTrue(all(str(probability) in table for probability in probabilities.values()))


class TestBayesianNetwork(TestCase):
    sample_network = [
        NetworkNode(node_name='D', predecessors=[], random_variables=[], probabilities={},
                    all_random_variables=[]),
        NetworkNode(node_name='I', predecessors=[], random_variables=[], probabilities={},
                    all_random_variables=[]),
        NetworkNode(node_name='G', predecessors=['D', 'I'], random_variables=[], probabilities={},
                    all_random_variables=[]),
        NetworkNode(node_name='K', predecessors=[], random_variables=[], probabilities={},
                    all_random_variables=[])]

    def _make_assertion(self, network: BayesianNetwork, network_node_count: int,
                        graph_node_count: int, graph_edge_count: int, edges_to_add_count: int):
        self.assertEqual(network_node_count, len(network.nodes))
        self.assertEqual(graph_node_count, len(network.G.nodes))
        self.assertEqual(graph_edge_count, len(network.G.edges))
        self.assertEqual(edges_to_add_count, len(network.edges_to_add))

    def test_empty_network(self):
        empty_net = BayesianNetwork([])
        self._make_assertion(network=empty_net, network_node_count=0, graph_node_count=0,
                             graph_edge_count=0, edges_to_add_count=0)

    def test_add_duplicate_node(self):
        empty_net = BayesianNetwork([])

        node = NetworkNode(node_name='G', predecessors=['D', 'I'], random_variables=[],
                           probabilities={}, all_random_variables=[])
        # Add firstly
        self.assertTrue(empty_net.add_node(node))
        self._make_assertion(network=empty_net, network_node_count=1, graph_node_count=1,
                             graph_edge_count=0, edges_to_add_count=2)
        # Try again
        self.assertFalse(empty_net.add_node(node))
        self._make_assertion(network=empty_net, network_node_count=1, graph_node_count=1,
                             graph_edge_count=0, edges_to_add_count=2)

    @mock.patch('logging.warning')
    def test_add_node_with_predecessor_itself(self, m_logger_warning):
        empty_net = BayesianNetwork([])

        node = NetworkNode(node_name='G', predecessors=['D', 'I', 'G'], random_variables=[],
                           probabilities={}, all_random_variables=[])
        empty_net.add_node(node)
        self._make_assertion(network=empty_net, network_node_count=1, graph_node_count=1,
                             graph_edge_count=0, edges_to_add_count=2)
        for predecessor, expected_count in zip(node.predecessors, [1, 1, 0]):
            self.assertEqual(expected_count, len(empty_net.edges_to_add[predecessor]))

        m_logger_warning.assert_called_once()

    def test_non_added_parent(self):
        empty_net = BayesianNetwork([])

        # Add node with non-existent predecessors
        node1 = NetworkNode(node_name='G', predecessors=['D', 'I'], random_variables=[],
                            probabilities={}, all_random_variables=[])
        empty_net.add_node(node1)

        self._make_assertion(network=empty_net, network_node_count=1, graph_node_count=1,
                             graph_edge_count=0, edges_to_add_count=2)
        for predecessor, expected_count in zip(node1.predecessors, [1, 1]):
            self.assertEqual(expected_count, len(empty_net.edges_to_add[predecessor]))

        # Add node with non-existent predecessors
        node2 = NetworkNode(node_name='I', predecessors=['D', 'L', 'K'], random_variables=[],
                            probabilities={}, all_random_variables=[])
        empty_net.add_node(node2)

        self._make_assertion(network=empty_net, network_node_count=2, graph_node_count=2,
                             graph_edge_count=1, edges_to_add_count=3)
        for predecessor, expected_count in zip(node2.predecessors, [2, 1, 1]):
            self.assertEqual(expected_count, len(empty_net.edges_to_add[predecessor]))

    @mock.patch('logging.warning')
    def test_add_cyclic_node(self, m_logger_warning):
        network = BayesianNetwork(self.sample_network)
        self._make_assertion(network=network, network_node_count=4, graph_node_count=4,
                             graph_edge_count=2, edges_to_add_count=0)

        m_logger_warning.assert_not_called()
        node1 = NetworkNode(node_name='B', predecessors=['A'], random_variables=[],
                            probabilities={}, all_random_variables=[])
        self.assertTrue(network.add_node(node1))
        m_logger_warning.assert_not_called()
        self._make_assertion(network=network, network_node_count=5, graph_node_count=5,
                             graph_edge_count=2, edges_to_add_count=1)

        node2 = NetworkNode(node_name='C', predecessors=['B'], random_variables=[],
                            probabilities={}, all_random_variables=[])
        self.assertTrue(network.add_node(node2))
        m_logger_warning.assert_not_called()
        self._make_assertion(network=network, network_node_count=6, graph_node_count=6,
                             graph_edge_count=3, edges_to_add_count=1)

        node3 = NetworkNode(node_name='A', predecessors=['C'], random_variables=[],
                            probabilities={}, all_random_variables=[])
        self.assertFalse(network.add_node(node3))
        m_logger_warning.assert_called_once()
        self._make_assertion(network=network, network_node_count=6, graph_node_count=6,
                             graph_edge_count=3, edges_to_add_count=1)

        node4 = NetworkNode(node_name='A', predecessors=[], random_variables=[], probabilities={},
                            all_random_variables=[])
        self.assertTrue(network.add_node(node4))
        self._make_assertion(network=network, network_node_count=7, graph_node_count=7,
                             graph_edge_count=4, edges_to_add_count=0)

    def test_initial_network(self):
        network = BayesianNetwork(self.sample_network)
        self._make_assertion(network=network, network_node_count=4, graph_node_count=4,
                             graph_edge_count=2, edges_to_add_count=0)

    def test_remove_node_from_network(self):
        network = BayesianNetwork(self.sample_network)
        self._make_assertion(network=network, network_node_count=4, graph_node_count=4,
                             graph_edge_count=2, edges_to_add_count=0)

        # Non-existent node
        self.assertFalse(network.remove_node('P'))
        self._make_assertion(network=network, network_node_count=4, graph_node_count=4,
                             graph_edge_count=2, edges_to_add_count=0)

        self.assertTrue(network.remove_node('D'))
        self._make_assertion(network=network, network_node_count=3, graph_node_count=3,
                             graph_edge_count=1, edges_to_add_count=0)

        self.assertTrue(network.remove_node('G'))
        self._make_assertion(network=network, network_node_count=2, graph_node_count=2,
                             graph_edge_count=0, edges_to_add_count=0)

        self.assertTrue(network.remove_node('K'))
        self._make_assertion(network=network, network_node_count=1, graph_node_count=1,
                             graph_edge_count=0, edges_to_add_count=0)

        # Non-existent node
        self.assertFalse(network.remove_node('P'))
        self._make_assertion(network=network, network_node_count=1, graph_node_count=1,
                             graph_edge_count=0, edges_to_add_count=0)

    def test_remove_node_check_edges_to_add_dict(self):
        network = BayesianNetwork(self.sample_network)
        self._make_assertion(network=network, network_node_count=4, graph_node_count=4,
                             graph_edge_count=2, edges_to_add_count=0)

        node1 = NetworkNode(node_name='A', predecessors=['X'], random_variables=[],
                            probabilities={}, all_random_variables=[])
        self.assertTrue(network.add_node(node1))
        node2 = NetworkNode(node_name='B', predecessors=['X', 'Y'], random_variables=[],
                            probabilities={}, all_random_variables=[])
        self.assertTrue(network.add_node(node2))
        node3 = NetworkNode(node_name='C', predecessors=['X', 'Y', 'Z', 'Q'], random_variables=[],
                            probabilities={}, all_random_variables=[])
        self.assertTrue(network.add_node(node3))

        self.assertEqual(3, len(network.edges_to_add['X']))
        self.assertEqual(2, len(network.edges_to_add['Y']))
        self.assertEqual(1, len(network.edges_to_add['Z']))
        self.assertEqual(1, len(network.edges_to_add['Q']))

        self.assertTrue(network.remove_node('B'))
        self.assertEqual(2, len(network.edges_to_add['X']))
        self.assertEqual(1, len(network.edges_to_add['Y']))
        self.assertEqual(1, len(network.edges_to_add['Z']))
        self.assertEqual(1, len(network.edges_to_add['Q']))

        self.assertTrue(network.remove_node('C'))
        self.assertEqual(1, len(network.edges_to_add['X']))
        self.assertEqual(0, len(network.edges_to_add['Y']))
        self.assertEqual(0, len(network.edges_to_add['Z']))
        self.assertEqual(0, len(network.edges_to_add['Q']))

        self.assertTrue(network.remove_node('A'))
        self.assertEqual(0, len(network.edges_to_add['X']))
        self.assertEqual(0, len(network.edges_to_add['Y']))
        self.assertEqual(0, len(network.edges_to_add['Z']))
        self.assertEqual(0, len(network.edges_to_add['Q']))


class BayesianNetworkProbabilityTest(TestCase):
    BURGLARY = "Burglary"
    EARTHQUAKE = "Earthquake"
    ALARM = "Alarm"
    JOHN_CALLS = "JohnCalls"
    MARRY_CALLS = "MaryCalls"

    LARGE_ERROR_DELTA = 0.0005
    SMALL_ERROR_DELTA = 0.000005

    sample_network = {
        BURGLARY: {
            "predecessors": [], "random_variables": ["t", "f"], "probabilities": {
                "(t)": 0.001, "(f)": 0.999
            }
        }, EARTHQUAKE: {
            "predecessors": [], "random_variables": ["t", "f"], "probabilities": {
                "(t)": 0.002, "(f)": 0.998
            }
        }, ALARM: {
            "predecessors": [BURGLARY, EARTHQUAKE], "random_variables": ["t", "f"],
            "probabilities": {
                "(f,f,f)": 0.999, "(f,f,t)": 0.001, "(f,t,f)": 0.71, "(f,t,t)": 0.29,
                "(t,f,f)": 0.06, "(t,f,t)": 0.94, "(t,t,f)": 0.05, "(t,t,t)": 0.95
            }
        }, JOHN_CALLS: {
            "predecessors": [ALARM], "random_variables": ["t", "f"], "probabilities": {
                "(f,f)": 0.95, "(f,t)": 0.05, "(t,f)": 0.10, "(t,t)": 0.90
            }
        }, MARRY_CALLS: {
            "predecessors": [ALARM], "random_variables": ["t", "f"], "probabilities": {
                "(f,f)": 0.99, "(f,t)": 0.01, "(t,f)": 0.30, "(t,t)": 0.70
            }
        }
    }

    def setUp(self) -> None:
        from ..input_parser.input_parser import InputParser
        self.network = BayesianNetwork(initial_network=InputParser.from_dict(self.sample_network))

    @mock.patch('src.entity.bayesian_network.BayesianNetwork._calculate_joint_probability')
    def test_single_float_returned_probability(self, mock_joint_probability):
        mock_joint_probability.side_effect = [0.3, 0.75]
        value = self.network.P(f'{self.BURGLARY}')
        self.assertAlmostEqual(0.3 / 0.75, value)

    @mock.patch('src.entity.bayesian_network.BayesianNetwork._calculate_joint_probability')
    def test_dictionary_returned_probability(self, mock_joint_probability):
        mock_joint_probability.side_effect = [{'a': 0.3, 'b': 0.2, 'c': 0.75}, 0.75]
        value = self.network.P(f'{self.BURGLARY}')
        self.assertAlmostEqual(0.3 / 0.75, value['a'])
        self.assertAlmostEqual(0.2 / 0.75, value['b'])
        self.assertAlmostEqual(0.75 / 0.75, value['c'])

    def test_invalid_query_probability(self):
        from src.exceptions.exceptions import InvalidQuery
        with self.assertRaises(InvalidQuery):
            self.network.P(f'{self.BURGLARY} | , ')

    @mock.patch('src.entity.bayesian_network.BayesianNetwork._probability_inference')
    def test_calculate_joint_probability(self, mock_probability_inference):
        # Note: Anyways we will have +1 for denominator part. The test will test nominator section
        mock_probability_inference.return_value = 0.8

        # No query variable without value
        query = f'{self.BURGLARY} = t, {self.ALARM} = f | {self.EARTHQUAKE} = f'
        self.network.P(query=query)
        self.assertEqual(2, mock_probability_inference.call_count)
        mock_probability_inference.reset_mock()

        # One query variable without value
        query = f'{self.BURGLARY} = t, {self.ALARM} | {self.EARTHQUAKE} = f'
        self.network.P(query=query)
        self.assertEqual(3, mock_probability_inference.call_count)
        mock_probability_inference.reset_mock()

        # One query variable without value
        query = f'{self.BURGLARY}, {self.ALARM} =f | {self.EARTHQUAKE} = f'
        self.network.P(query=query)
        self.assertEqual(3, mock_probability_inference.call_count)
        mock_probability_inference.reset_mock()

        # Two query variable without value
        query = f'{self.BURGLARY}, {self.ALARM} | {self.EARTHQUAKE} = f'
        self.network.P(query=query)
        self.assertEqual(5, mock_probability_inference.call_count)
        mock_probability_inference.reset_mock()

        # Two query variable without value
        query = f'{self.BURGLARY}, {self.ALARM}'
        self.network.P(query=query)
        self.assertEqual(5, mock_probability_inference.call_count)
        mock_probability_inference.reset_mock()

        # Three query variable without value
        query = f'{self.BURGLARY}, {self.ALARM}, {self.EARTHQUAKE}'
        self.network.P(query=query)
        self.assertEqual(9, mock_probability_inference.call_count)
        mock_probability_inference.reset_mock()

        # Three query variable without value
        query = f'{self.BURGLARY}, {self.ALARM}, {self.EARTHQUAKE} | {self.JOHN_CALLS} = t'
        self.network.P(query=query)
        self.assertEqual(9, mock_probability_inference.call_count)
        mock_probability_inference.reset_mock()

        # Three query variable without value
        query = f'{self.BURGLARY}, {self.ALARM}, {self.EARTHQUAKE} | {self.JOHN_CALLS} = f, ' \
                f'{self.MARRY_CALLS} = t'
        self.network.P(query=query)
        self.assertEqual(9, mock_probability_inference.call_count)
        mock_probability_inference.reset_mock()

    def test_probability_inference_with_no_factor(self):
        self.assertAlmostEqual(1.0, self.network._probability_inference(tuple()))

    def test_probability_inference_with_one_factor(self):
        self.assertAlmostEqual(0.001, self.network._probability_inference(
            (ProbabilityFactor(name=f'{self.ALARM}'),), Alarm='t', Earthquake='f', Burglary='f'))

        self.assertAlmostEqual(1.0, self.network._probability_inference(
            (ProbabilityFactor(name=f'{self.ALARM}', sum_out=True),), Earthquake='f', Burglary='f'))

        self.assertAlmostEqual(0.001, self.network._probability_inference(
            (ProbabilityFactor(name=f'{self.ALARM}', value='t'),), Earthquake='f', Burglary='f'))

        with self.assertRaises(InvalidProbabilityFactor):
            self.assertAlmostEqual(0.001, self.network._probability_inference(
                (ProbabilityFactor(name=f'{self.ALARM}', value='t', sum_out=True),)))

    def test_decide_calculation_order(self):
        def _get_ordering_for_parameters(_query_variables):
            needed_variables = {v.name: v for v in _query_variables}
            purified_variables = self.network._eliminate_unnecessary_variables(
                variables=needed_variables.keys())
            hidden_variables = {v for v in purified_variables if v not in needed_variables.keys()}

            return self.network._decide_calculation_order(needed_variable_names=needed_variables,
                                                          purified_variables=purified_variables,
                                                          hidden_variables=hidden_variables)

        ordering = _get_ordering_for_parameters([QueryVariable(name=f'{self.JOHN_CALLS}')])
        self.assertListEqual([ProbabilityFactor(name=f'{self.EARTHQUAKE}', sum_out=True),
                              ProbabilityFactor(name=f'{self.BURGLARY}', sum_out=True),
                              ProbabilityFactor(name=f'{self.ALARM}', sum_out=True),
                              ProbabilityFactor(name=f'{self.JOHN_CALLS}'), ], ordering)

        ordering = _get_ordering_for_parameters(
            [QueryVariable(name=f'{self.JOHN_CALLS}', value='t')])
        self.assertListEqual([ProbabilityFactor(name=f'{self.EARTHQUAKE}', sum_out=True),
                              ProbabilityFactor(name=f'{self.BURGLARY}', sum_out=True),
                              ProbabilityFactor(name=f'{self.ALARM}', sum_out=True),
                              ProbabilityFactor(name=f'{self.JOHN_CALLS}', value='t')], ordering)

        ordering = _get_ordering_for_parameters(
            [QueryVariable(name=f'{self.EARTHQUAKE}', value='f'),
             QueryVariable(name=f'{self.JOHN_CALLS}')])
        self.assertListEqual([ProbabilityFactor(name=f'{self.EARTHQUAKE}', value='f'),
                              ProbabilityFactor(name=f'{self.BURGLARY}', sum_out=True),
                              ProbabilityFactor(name=f'{self.ALARM}', sum_out=True),
                              ProbabilityFactor(name=f'{self.JOHN_CALLS}'), ], ordering)

    def test_variable_elimination(self):
        # Try each variable one by one
        variables = {self.JOHN_CALLS}
        needed_variable_set = self.network._eliminate_unnecessary_variables(variables=variables)
        self.assertSetEqual({self.JOHN_CALLS, self.ALARM, self.EARTHQUAKE, self.BURGLARY},
                            needed_variable_set)

        variables = {self.MARRY_CALLS}
        needed_variable_set = self.network._eliminate_unnecessary_variables(variables=variables)
        self.assertSetEqual({self.MARRY_CALLS, self.ALARM, self.EARTHQUAKE, self.BURGLARY},
                            needed_variable_set)

        variables = {self.ALARM}
        needed_variable_set = self.network._eliminate_unnecessary_variables(variables=variables)
        self.assertSetEqual({self.ALARM, self.EARTHQUAKE, self.BURGLARY}, needed_variable_set)

        variables = {self.EARTHQUAKE}
        needed_variable_set = self.network._eliminate_unnecessary_variables(variables=variables)
        self.assertSetEqual({self.EARTHQUAKE}, needed_variable_set)

        variables = {self.BURGLARY}
        needed_variable_set = self.network._eliminate_unnecessary_variables(variables=variables)
        self.assertSetEqual({self.BURGLARY}, needed_variable_set)

        # Try multiple variables at a time
        variables = {self.JOHN_CALLS, self.MARRY_CALLS}
        needed_variable_set = self.network._eliminate_unnecessary_variables(variables=variables)
        self.assertSetEqual(
            {self.JOHN_CALLS, self.MARRY_CALLS, self.ALARM, self.EARTHQUAKE, self.BURGLARY},
            needed_variable_set)

        variables = {self.JOHN_CALLS, self.ALARM}
        needed_variable_set = self.network._eliminate_unnecessary_variables(variables=variables)
        self.assertSetEqual({self.JOHN_CALLS, self.ALARM, self.EARTHQUAKE, self.BURGLARY},
                            needed_variable_set)

        variables = {self.EARTHQUAKE, self.BURGLARY}
        needed_variable_set = self.network._eliminate_unnecessary_variables(variables=variables)
        self.assertSetEqual({self.EARTHQUAKE, self.BURGLARY}, needed_variable_set)

        variables = {self.MARRY_CALLS, self.EARTHQUAKE}
        needed_variable_set = self.network._eliminate_unnecessary_variables(variables=variables)
        self.assertSetEqual({self.MARRY_CALLS, self.ALARM, self.EARTHQUAKE, self.BURGLARY},
                            needed_variable_set)

    def test_bayesian_network_exact_inference_1(self):
        p = self.network.P(f'{self.BURGLARY} | {self.JOHN_CALLS} = t, {self.MARRY_CALLS} = t')

        # Case of exposure to burglary when your both neighbors call you
        self.assertAlmostEqual(0.284, p[str({f'{self.BURGLARY}': 't'})],
                               delta=self.LARGE_ERROR_DELTA)
        # Case of non-exposure to burglary when your both neighbors call you
        self.assertAlmostEqual(0.716, p[str({f'{self.BURGLARY}': 'f'})],
                               delta=self.LARGE_ERROR_DELTA)

    def test_bayesian_network_exact_inference_2(self):
        # Should be the same with respect to network structure
        mary_call_alarm_combination = itertools.product(['t', 'f'], ['t', 'f'])
        for mary_call, alarm in mary_call_alarm_combination:
            p1 = self.network.P(
                f'{self.MARRY_CALLS} = {mary_call} | {self.JOHN_CALLS} = f, {self.ALARM} = '
                f'{alarm}, {self.EARTHQUAKE}= f, {self.BURGLARY} = t')
            p2 = self.network.P(f'{self.MARRY_CALLS} = {mary_call} | {self.ALARM} = {alarm}')
            self.assertAlmostEqual(p1, p2, delta=self.LARGE_ERROR_DELTA)

    def test_bayesian_network_exact_inference_3(self):
        p = self.network.P(
            f'{self.JOHN_CALLS} = t, {self.MARRY_CALLS} = t, {self.ALARM} = t,  {self.BURGLARY} = '
            f'f, {self.EARTHQUAKE} = f')
        self.assertAlmostEqual(0.000628, p, delta=self.SMALL_ERROR_DELTA)

    def test_bayesian_network_exact_inference_4(self):
        p1 = self.network.P(f'{self.ALARM} = t | {self.BURGLARY} = t, {self.EARTHQUAKE} = t')
        self.assertAlmostEqual(0.95, p1, delta=self.LARGE_ERROR_DELTA)

        p1 = self.network.P(f'{self.ALARM} = f | {self.BURGLARY} = f, {self.EARTHQUAKE} = f')
        self.assertAlmostEqual(0.999, p1, delta=self.LARGE_ERROR_DELTA)
