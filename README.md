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
>>> from bayesian_inference import BayesianNetwork, InputParser
>>>
>>> BURGLARY = "Burglary"
>>> EARTHQUAKE = "Earthquake"
>>> ALARM = "Alarm"
>>> JOHN_CALLS = "JohnCalls"
>>> MARRY_CALLS = "MaryCalls"
>>> 
>>> sample_network = {
...     BURGLARY: {
...         "predecessors": [], "random_variables": ["t", "f"], "probabilities": {
...             "(t)": 0.001, "(f)": 0.999
...         }
...     }, EARTHQUAKE: {
...         "predecessors": [], "random_variables": ["t", "f"], "probabilities": {
...             "(t)": 0.002, "(f)": 0.998
...         }
...     }, ALARM: {
...         "predecessors": [BURGLARY, EARTHQUAKE], "random_variables": ["t", "f"], "probabilities": {
...             "(f,f,f)": 0.999, "(f,f,t)": 0.001, "(f,t,f)": 0.71, "(f,t,t)": 0.29, "(t,f,f)": 0.06,
...             "(t,f,t)": 0.94, "(t,t,f)": 0.05, "(t,t,t)": 0.95
...         }
...     }, JOHN_CALLS: {
...         "predecessors": [ALARM], "random_variables": ["t", "f"], "probabilities": {
...             "(f,f)": 0.95, "(f,t)": 0.05, "(t,f)": 0.10, "(t,t)": 0.90
...         }
...     }, MARRY_CALLS: {
...         "predecessors": [ALARM], "random_variables": ["t", "f"], "probabilities": {
...             "(f,f)": 0.99, "(f,t)": 0.01, "(t,f)": 0.30, "(t,t)": 0.70
...         }
...     }
... }
>>> network = BayesianNetwork(initial_network=InputParser.from_dict(sample_network))
```

### Network Node
Single unit in the network representing a random variable in the uncertain world.
It has the following fields expected by constructor:
* node_name: Random variable name which will be the node name in the network
* random_variables: List of available values of random variable in string format
* predecessors: Parents of the random variable in the network as a list of string where each item
is the name of parent random variable
* probabilities: Probability list of the random variable described as conditional probabilities
* all_random_variables: List of lists of strings representing random variable values respectively
parents of the node and the values of current node

Single node can be represented with the following representation:

```python
>>> from bayesian_inference import NetworkNode
>>> node = eval(NetworkNode('Alarm', ['t', 'f'], ['Burglary', 'Earthquake'], {'(f,f,f)': 0.999, '(f,f,t)': 0.001, '(f,t,f)': 0.71, '(f,t,t)': 0.29, '(t,f,f)': 0.06, '(t,f,t)': 0.94, '(t,t,f)': 0.05, '(t,t,t)': 0.95}, [['t', 'f'], ['t', 'f'], ['t', 'f']]))
>>> print(node)
| Burglary   | Earthquake   |   P(Alarm=t) |   P(Alarm=f) |
|------------|--------------|--------------|--------------|
| t          | t            |        0.95  |        0.05  |
| t          | f            |        0.94  |        0.06  |
| f          | t            |        0.29  |        0.71  |
| f          | f            |        0.001 |        0.999 |
```

**Note:** It is important that you need to provide probability dictionary of `NetworkNode` as explained
in the following example. Let's have node named `X` and parents as `[A, B, C]`, then you need to have all
probability keys as `(value_a,value_b,value_c,value_x)` where no whitespace between commas and value are
listed order of parents and node itself if you want to create node from yourself.
If you parse with `InputParser`, then it goes over keys and removes whitespaces to make them as expected format. 

### Bayesian Network
Bayesian network structure that keeps `Directed Acyclic Graph` inside and encapsulates `NetworkNode` instances
The structure has an instance of [NetworkX](https://github.com/networkx/networkx) DiGraph. Network can be created
with initial node list. Also, one can add and remove node to the network at runtime. From probability perspective,
one can query exact inference of probability from Bayesian network.

```python
>>> # Network initiated above
>>> from bayesian_inference import NetworkNode
>>> node1 = NetworkNode(node_name='B', predecessors=['A'], random_variables=[], probabilities={}, all_random_variables=[])
>>> node2 = NetworkNode(node_name='C', predecessors=['B'], random_variables=[], probabilities={}, all_random_variables=[])
>>> 
>>> # Adding node to network, Method expects network node directly
>>> network.add_node(node1)
>>> network.add_node(node2)
>>> 
>>> # Removal of node from network. Method expects node name to remove
>>> network.remove_node(node2.node_name)
>>>
>>> # Query exact inference from network, details of queries will be explained in next sections
>>> network.P('Burglary | JohnCalls = t, MaryCalls = t')
{"{'Burglary': 't'}": 0.28417183536439294, "{'Burglary': 'f'}": 0.7158281646356072}
>>> network.P('JohnCalls = t, MaryCalls = t, Alarm = t,  Burglary = f, Earthquake = f')
0.0006281112599999999
```

