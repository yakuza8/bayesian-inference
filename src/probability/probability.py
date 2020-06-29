import re

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ..exceptions.exceptions import NonUniqueRandomVariablesInQuery, RandomVariableNotInContext

WORD = r'(\s*\w+\s*)'
NON_VALUED_GROUP = rf'(?:{WORD}(?:={WORD})?)'
VALUED_GROUP = rf'(?:{WORD}={WORD})'
QUERY_VARIABLES = rf'{NON_VALUED_GROUP}(?:,{NON_VALUED_GROUP})*'
EVIDENCE_VARIABLES = rf'{VALUED_GROUP}(?:,{VALUED_GROUP})*'
QUERY = rf'{QUERY_VARIABLES}(?:\s*\|\s*{EVIDENCE_VARIABLES})?'


@dataclass
class QueryVariable:
    name: str
    value: str = None


def query_parser(query: str, expected_symbol_and_values: Dict[str, List[str]] = None) -> Tuple[
    bool, Optional[List[QueryVariable]], Optional[List[QueryVariable]]]:
    """
    Query parsing functionality where conditional probabilities also supported

    :param query: String valued query where expected format is like 'X,Y=2..|Z=3,K=k..' where query
    variables should have at least one entity and evidence part could have zero or more entity.
    Also, query variables could have no value, but all evidence variables should have known
    variables
    :param expected_symbol_and_values: In case of giving this parameter, validation of entries could
    be done whether query variables exist among them and their values are valid
    :return: Boolean flag whether the query successfully, list of query variables and list of
    evidence variables

    >>> # Valid queries
    >>> query_parser('A, B, C')[0]
    >>> True
    >>> query_parser('A, B=b, C')[0]
    >>> True
    >>> query_parser('A=1, B, C')[0]
    >>> True
    >>> query_parser('A, B, C=2')[0]
    >>> True
    >>> query_parser('A=1, B=2, C=3')[0]
    >>> True
    >>> query_parser('A, B, C | D=d')[0]
    >>> True
    >>> query_parser('A=1, B=2, C=2 | D=d')[0]
    >>> True
    >>> query_parser('A, B=2, C | D=d, E=5')[0]
    >>> True
    >>> # Invalid queries (It is expected that all evidence variables should have value)
    >>> query_parser('A, B, C | D')[0]
    >>> False
    >>> query_parser('A, B=b, C | D')[0]
    >>> False
    >>> query_parser('A=1, B, C | D')[0]
    >>> False
    """

    def map_to_query_variable(matched: re.Match):
        """
        Map regular expression match to QueryVariable
        """
        lhs, rhs = matched.group(1), matched.group(2)
        return QueryVariable(str(lhs).strip(), str(rhs).strip() if rhs is not None else None)

    def make_all_variables_unique(query_variables: List[QueryVariable],
                                  evidence_variables: List[QueryVariable]):
        """
        Combination of query variables and evidence variables has all variables as unique
        """
        unique_set = set(e.name for e in query_variables)
        unique_set.update(set(e.name for e in evidence_variables))
        if len(unique_set) != len(query_variables) + len(evidence_variables):
            raise NonUniqueRandomVariablesInQuery('Need all variables in query to be unique.')

    def check_all_variables_exist_in_context(variables: List[QueryVariable],
                                             context: Dict[str, List[str]]):
        """
        Check all variable names in the context and if it has value then it is expected in the list
        of that variable in the context
        """
        for variable in variables:
            if variable.name not in context or (
                    variable.value is not None and variable.value not in context[variable.name]):
                raise RandomVariableNotInContext(
                    f'{variable.name} is either not in context or its value is not satified to '
                    f'have.')

    # Try to full match, we need full match for query. Otherwise, it is not parsable
    match = re.fullmatch(QUERY, query)

    if match:
        # Separate queries and evidences
        split = re.split(r'\|', query)
        queries, evidences = split[0], split[1] if len(split) > 1 else ''

        queries = [map_to_query_variable(matched=m) for m in
                   re.finditer(rf'{NON_VALUED_GROUP}|{VALUED_GROUP}', queries)]
        evidences = [map_to_query_variable(matched=m) for m in re.finditer(VALUED_GROUP, evidences)]

        # Make validations
        make_all_variables_unique(query_variables=queries, evidence_variables=evidences)
        if expected_symbol_and_values is not None:
            check_all_variables_exist_in_context(variables=queries,
                                                 context=expected_symbol_and_values)
            check_all_variables_exist_in_context(variables=evidences,
                                                 context=expected_symbol_and_values)
        # Return parsed query variables and evidence variables
        return True, queries, evidences
    else:
        return False, None, None
