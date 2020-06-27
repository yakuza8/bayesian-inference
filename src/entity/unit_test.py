from unittest import TestCase, mock

from .bayesian_network import BayesianNetwork
from .network_node import NetworkNode


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
