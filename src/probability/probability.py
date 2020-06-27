import re

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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


def query_parser(query: str, network_context: Dict[str, List[str]] = None) -> Tuple[
    bool, Optional[List[QueryVariable]], Optional[List[QueryVariable]]]:
    """
    Query parsing functionality where conditional probabilities also supported

    :param query: String valued query where expected format is like 'X,Y=2...|Z=3,K...' where query
    variables should have at least one entity and evidence part could have zero or more entity.
    Also, query variables could have no value, but all evidence variables should have known
    variables
    :param network_context: In case of giving this parameter, validation of entries could be done
    :return: Boolean flag whether the query successfully, list of query variables and list of
    evidence variables
    """

    def map_to_query_variable(matched: re.Match):
        lhs, rhs = matched.group(1), matched.group(2)
        return QueryVariable(str(lhs).strip(), str(rhs).strip() if rhs is not None else None)

    match = re.fullmatch(QUERY, query)

    if match:
        # Separate queries and evidences
        split = re.split(r'\|', query)
        queries, evidences = split[0], split[1] if len(split) > 1 else ''

        return True, [map_to_query_variable(m) for m in
                      re.finditer(rf'{NON_VALUED_GROUP}|{VALUED_GROUP}', queries)], [
                   map_to_query_variable(m) for m in re.finditer(VALUED_GROUP, evidences)]
    else:
        return False, None, None
