# Input Parser Exceptions
class IncompleteNodeDataException(Exception):
    pass


class PredecessorNotExistInNetwork(Exception):
    pass


class NotAllExpectedProbabilityExist(Exception):
    pass


class HaveAtLeastOneRandomVariable(Exception):
    pass


# Probability
class NonUniqueRandomVariablesInQuery(Exception):
    pass


class RandomVariableNotInContext(Exception):
    pass


# Bayesian Network
class InvalidQuery(Exception):
    pass
