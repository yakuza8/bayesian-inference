from unittest import TestCase

from .probability import query_parser, QueryVariable


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
        self.assertListEqual([QueryVariable('D', 'd'), QueryVariable('E', 'e'),QueryVariable('F', 'f')], evidences)

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
