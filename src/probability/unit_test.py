from unittest import TestCase

from .probability import query_parser, QueryVariable
from ..exceptions.exceptions import NonUniqueRandomVariablesInQuery, RandomVariableNotInContext


class TestProbabilityParser(TestCase):

    def test_valid_expression_1(self):
        query = 'A'
        value, queries, evidences = query_parser(query=query)

        self.assertTrue(value)
        self.assertListEqual([QueryVariable('A')], queries)
        self.assertListEqual([], evidences)

    def test_valid_expression_2(self):
        query = 'A, B, C'
        value, queries, evidences = query_parser(query=query)

        self.assertTrue(value)
        self.assertListEqual([QueryVariable('A'), QueryVariable('B'), QueryVariable('C')], queries)
        self.assertListEqual([], evidences)

    def test_valid_expression_3(self):
        query = 'A, B = b, C = c, D'
        value, queries, evidences = query_parser(query=query)

        self.assertTrue(value)
        self.assertListEqual([QueryVariable('A'), QueryVariable('B', 'b'), QueryVariable('C', 'c'),
                              QueryVariable('D')], queries)
        self.assertListEqual([], evidences)

    def test_valid_expression_4(self):
        query = 'B = b, A, C, D = D'
        value, queries, evidences = query_parser(query=query)

        self.assertTrue(value)
        self.assertListEqual([QueryVariable('B', 'b'), QueryVariable('A'), QueryVariable('C'),
                              QueryVariable('D', 'D')], queries)
        self.assertListEqual([], evidences)

    def test_valid_expression_5(self):
        query = 'A,B=b,C,D,E,F=f,G'
        value, queries, evidences = query_parser(query=query)

        self.assertTrue(value)
        self.assertListEqual(
            [QueryVariable('A'), QueryVariable('B', 'b'), QueryVariable('C'), QueryVariable('D'),
             QueryVariable('E'), QueryVariable('F', 'f'), QueryVariable('G')], queries)
        self.assertListEqual([], evidences)

    def test_valid_expression_6(self):
        query = '     B  =   b ,  A   ,   C ,   D  =  D '
        value, queries, evidences = query_parser(query=query)

        self.assertTrue(value)
        self.assertListEqual([QueryVariable('B', 'b'), QueryVariable('A'), QueryVariable('C'),
                              QueryVariable('D', 'D')], queries)
        self.assertListEqual([], evidences)

    def test_valid_expression_7(self):
        query = 'A, B=b, C | D = d'
        value, queries, evidences = query_parser(query=query)

        self.assertTrue(value)
        self.assertListEqual([QueryVariable('A'), QueryVariable('B', 'b'), QueryVariable('C')],
                             queries)
        self.assertListEqual([QueryVariable('D', 'd')], evidences)

    def test_valid_expression_8(self):
        query = 'A, B=b, C | D = d, E = e'
        value, queries, evidences = query_parser(query=query)

        self.assertTrue(value)
        self.assertListEqual([QueryVariable('A'), QueryVariable('B', 'b'), QueryVariable('C')],
                             queries)
        self.assertListEqual([QueryVariable('D', 'd'), QueryVariable('E', 'e')], evidences)

    def test_valid_expression_9(self):
        query = 'A, B=b, C |D=d,E=e,             F = f         '
        value, queries, evidences = query_parser(query=query)

        self.assertTrue(value)
        self.assertListEqual([QueryVariable('A'), QueryVariable('B', 'b'), QueryVariable('C')],
                             queries)
        self.assertListEqual(
            [QueryVariable('D', 'd'), QueryVariable('E', 'e'), QueryVariable('F', 'f')], evidences)

    def test_invalid_expression_1(self):
        query = ''
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_2(self):
        query = ','
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_3(self):
        query = ' , | '
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_4(self):
        query = 'A ,'
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_5(self):
        query = ', A'
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_6(self):
        query = 'A | '
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_7(self):
        query = 'A = a |'
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_8(self):
        query = 'A = a | , '
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_9(self):
        query = 'A = , '
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_10(self):
        query = 'A = |'
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_11(self):
        query = 'A = a | B'
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_12(self):
        query = 'A = a | B, '
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_13(self):
        query = 'A = a | B = b, '
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_14(self):
        query = 'A = a, C | B = b | K '
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_invalid_expression_15(self):
        query = 'A K = a, C | B = b'
        value, queries, evidences = query_parser(query=query)
        self.assertFalse(value)

    def test_non_unique_valid_expression_1(self):
        query = 'A, B, C, A, G'
        with self.assertRaises(NonUniqueRandomVariablesInQuery):
            query_parser(query=query)

    def test_non_unique_valid_expression_2(self):
        query = 'A, B, C | A = a, G = g'
        with self.assertRaises(NonUniqueRandomVariablesInQuery):
            query_parser(query=query)

    def test_non_unique_valid_expression_3(self):
        query = 'K | A = a, B = b, C = c, A = a, G = g'
        with self.assertRaises(NonUniqueRandomVariablesInQuery):
            query_parser(query=query)

    def test_all_variables_exist_1(self):
        query = 'A, B=b, C, D=d | E=e, F=ff, G=g, H=hh'
        context = {
            'A': ['a', 'aa'], 'B': ['b', 'bb'], 'C': ['c', 'cc'], 'D': ['d', 'dd'],
            'E': ['e', 'ee'], 'F': ['f', 'ff'], 'G': ['g', 'gg'], 'H': ['h', 'hh']
        }
        value, queries, evidences = query_parser(query=query, expected_symbol_and_values=context)
        self.assertTrue(value)
        self.assertListEqual([QueryVariable('A'), QueryVariable('B', 'b'), QueryVariable('C'),
                              QueryVariable('D', 'd')], queries)
        self.assertListEqual(
            [QueryVariable('E', 'e'), QueryVariable('F', 'ff'), QueryVariable('G', 'g'),
             QueryVariable('H', 'hh')], evidences)

    def test_all_variables_exist_2(self):
        query = 'A, B=b, C, D=d | E=e, F=ff, G=g, H=hh'
        context = {
            'A': ['a', 'aa'], 'B': ['b', 'bb'], 'D': ['d', 'dd'], 'E': ['e', 'ee'],
            'F': ['f', 'ff'], 'G': ['g', 'gg'], 'H': ['h', 'hh']
        }
        with self.assertRaises(RandomVariableNotInContext) as e:
            query_parser(query=query, expected_symbol_and_values=context)
        self.assertTrue('C' in str(e.exception))

    def test_all_variables_exist_3(self):
        query = 'A, B=b, C, D=d | E=e, F=ff, G=g, H=hh'
        context = {
            'A': ['a', 'aa'], 'C': ['c', 'cc'], 'D': ['d', 'dd'], 'E': ['e', 'ee'],
            'F': ['f', 'ff'], 'G': ['g', 'gg'], 'H': ['h', 'hh']
        }
        with self.assertRaises(RandomVariableNotInContext) as e:
            query_parser(query=query, expected_symbol_and_values=context)
        self.assertTrue('B' in str(e.exception))

    def test_all_variables_exist_4(self):
        query = 'A, B=b, C, D=d | E=e, F=ff, G=g, H=hh'
        context = {
            'A': ['a', 'aa'], 'B': ['bb'], 'C': ['c', 'cc'], 'D': ['d', 'dd'], 'E': ['e', 'ee'],
            'F': ['f', 'ff'], 'G': ['g', 'gg'], 'H': ['h', 'hh']
        }
        with self.assertRaises(RandomVariableNotInContext) as e:
            query_parser(query=query, expected_symbol_and_values=context)
        self.assertTrue('B' in str(e.exception))

    def test_all_variables_exist_5(self):
        query = 'A, B=b, C, D=d | E=e, F=ff, G=g, H=hh'
        context = {
            'A': ['a', 'aa'], 'B': ['b', 'bb'], 'C': ['c', 'cc'], 'D': ['d', 'dd'],
            'E': ['e', 'ee'], 'G': ['g', 'gg'], 'H': ['h', 'hh']
        }
        with self.assertRaises(RandomVariableNotInContext) as e:
            query_parser(query=query, expected_symbol_and_values=context)
        self.assertTrue('F' in str(e.exception))

    def test_all_variables_exist_6(self):
        query = 'A, B=b, C, D=d | E=e, F=ff, G=g, H=hh'
        context = {
            'A': ['a', 'aa'], 'B': ['b', 'bb'], 'C': ['c', 'cc'], 'D': ['d', 'dd'],
            'E': ['e', 'ee'], 'F': ['f', 'ff'], 'G': ['gg'], 'H': ['h', 'hh']
        }
        with self.assertRaises(RandomVariableNotInContext) as e:
            query_parser(query=query, expected_symbol_and_values=context)
        self.assertTrue('G' in str(e.exception))
