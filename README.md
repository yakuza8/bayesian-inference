# Bayesian Inference
[![Build Status](https://travis-ci.com/yakuza8/bayesian-inference.svg?branch=master)](https://travis-ci.com/yakuza8/bayesian-inference)
[![codecov](https://codecov.io/gh/yakuza8/bayesian-inference/branch/master/graph/badge.svg)](https://codecov.io/gh/yakuza8/bayesian-inference)

## Project Description
Probabilistic reasoning module on `Bayesian Networks` where the dependencies between variables are
represented as links among nodes on the `directed acyclic graph`. Even we could infer any probability
in the knowledge world via full joint distribution, we can optimize this calculation by independence
and conditional independence. 

In current implementation, one can define properties of the network as follows:
* Each node represents a single random variable
* Links between nodes represent direct effect on each other such as if `random variable X` has link
to `random variable Y`, then there is a conditional probability relation between them. 
* There is no cycle in the network and that makes the network `Directed Acyclic Graph`

## Entities
Usable entities available in the project are listed below which are `NetworkNode` and `BayesianNetwork`.
There is a simple network configuration as dictionary format below and entities will be explained with
respect to example network.

```python
BURGLARY = "Burglary"
EARTHQUAKE = "Earthquake"
ALARM = "Alarm"
JOHN_CALLS = "JohnCalls"
MARRY_CALLS = "MaryCalls"

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
        "predecessors": [BURGLARY, EARTHQUAKE], "random_variables": ["t", "f"], "probabilities": {
            "(f,f,f)": 0.999, "(f,f,t)": 0.001, "(f,t,f)": 0.71, "(f,t,t)": 0.29, "(t,f,f)": 0.06,
            "(t,f,t)": 0.94, "(t,t,f)": 0.05, "(t,t,t)": 0.95
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
```

### Network Node
Single unit in the network representing a random variable in the uncertain world.
It has the following fields:
* node_name: Random variable name which will be the node name in the network
* random_variables: List of available values of random variable in string format
* predecessors: Parents of the random variable in the network as a list of string where each item
is the name of parent random variable
* probabilities: Probability list of the random variable described as conditional probabilities
* all_random_variables: List of lists of strings representing random variable values respectively
parents of the node and the values of current node


Single node can be represented with the following representation:

.. code:: python

    >>> node = eval(NetworkNode('Alarm', ['t', 'f'], ['Burglary', 'Earthquake'], {'(f,f,f)': 0.999, '(f,f,t)': 0.001, '(f,t,f)': 0.71, '(f,t,t)': 0.29, '(t,f,f)': 0.06, '(t,f,t)': 0.94, '(t,t,f)': 0.05, '(t,t,t)': 0.95}, [['t', 'f'], ['t', 'f'], ['t', 'f']]))
    >>> print(node)
    | Burglary   | Earthquake   |   P(Alarm=t) |   P(Alarm=f) |
    |------------|--------------|--------------|--------------|
    | t          | t            |        0.95  |        0.05  |
    | t          | f            |        0.94  |        0.06  |
    | f          | t            |        0.29  |        0.71  |
    | f          | f            |        0.001 |        0.999 |

