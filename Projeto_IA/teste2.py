# -*- coding: utf-8 -*-

from collections import defaultdict, Counter
import itertools
import math
import random


class BayesNet(object):
    "Bayesian network: a graph of variables connected by parent links."

    def __init__(self):
        self.variables = []  # List of variables, in parent-first topological sort order
        self.lookup = {}  # Mapping of {variable_name: variable} pairs

    def add(self, name, parentnames, cpt):
        "Add a new Variable to the BayesNet. Parentnames must have been added previously."
        parents = [self.lookup[name] for name in parentnames]
        var = Variable(name, cpt, parents)
        self.variables.append(var)
        self.lookup[name] = var
        return self


class Variable(object):
    "A discrete random variable; conditional on zero or more parent Variables."

    def __init__(self, name, cpt, parents=()):
        "A variable has a name, list of parent variables, and a Conditional Probability Table."
        self.__name__ = name
        self.parents = parents
        self.cpt = CPTable(cpt, parents)
        self.domain = set(itertools.chain(*self.cpt.values()))  # All the outcomes in the CPT

    def __repr__(self): return self.__name__


class Factor(dict): "An {outcome: frequency} mapping."


class ProbDist(Factor):
    """A Probability Distribution is an {outcome: probability} mapping.
    The values are normalized to sum to 1.
    ProbDist(0.75) is an abbreviation for ProbDist({T: 0.75, F: 0.25})."""

    def __init__(self, mapping=(), **kwargs):
        if isinstance(mapping, float):
            mapping = {T: mapping, F: 1 - mapping}
        self.update(mapping, **kwargs)
        normalize(self)


class Evidence(dict):
    "A {variable: value} mapping, describing what we know for sure."


class CPTable(dict):
    "A mapping of {row: ProbDist, ...} where each row is a tuple of values of the parent variables."

    def __init__(self, mapping, parents=()):
        """Provides two shortcuts for writing a Conditional Probability Table.
        With no parents, CPTable(dist) means CPTable({(): dist}).
        With one parent, CPTable({val: dist,...}) means CPTable({(val,): dist,...})."""
        if len(parents) == 0 and not (isinstance(mapping, dict) and set(mapping.keys()) == {()}):
            mapping = {(): mapping}
        for (row, dist) in mapping.items():
            if len(parents) == 1 and not isinstance(row, tuple):
                row = (row,)
            self[row] = ProbDist(dist)


class Bool(int):
    "Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'"
    __str__ = __repr__ = lambda self: 'T' if self else 'F'


T = Bool(True)
F = Bool(False)

"""### Funções associadas:"""


def P(var, evidence={}):
    "The probability distribution for P(variable | evidence), when all parent variables are known (in evidence)."
    row = tuple(evidence[parent] for parent in var.parents)
    return var.cpt[row]


def normalize(dist):
    "Normalize a {key: value} distribution so values sum to 1.0. Mutates dist and returns it."
    total = sum(dist.values())
    for key in dist:
        dist[key] = dist[key] / total
        assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
    return dist


def sample(probdist):
    "Randomly sample an outcome from a probability distribution."
    r = random.random()  # r is a random point in the probability distribution
    c = 0.0  # c is the cumulative probability of outcomes seen so far
    for outcome in probdist:
        c += probdist[outcome]
        if r <= c:
            return outcome


"""# Redes bayesianas enquanto Distribuição de Probabilidade Conjunta

#### P(*X*<sub>1</sub>=*x*<sub>1</sub>, ..., *X*<sub>*n*</sub>=*x*<sub>*n*</sub>) = <font size=large>&Pi;</font><sub>*i*</sub> P(*X*<sub>*i*</sub> = *x*<sub>*i*</sub> | parents(*X*<sub>*i*</sub>))
"""


def joint_distribution(net):
    "Given a Bayes net, create the joint distribution over all variables."
    return ProbDist({row: prod(P_xi_given_parents(var, row, net)
                               for var in net.variables)
                     for row in all_rows(net)})


def all_rows(net): return itertools.product(*[var.domain for var in net.variables])


def P_xi_given_parents(var, row, net):
    "The probability that var = xi, given the values in this row."
    dist = P(var, Evidence(zip(net.variables, row)))
    xi = row[net.variables.index(var)]
    return dist[xi]


def prod(numbers):
    "The product of numbers: prod([2, 3, 5]) == 30. Analogous to `sum([2, 3, 5]) == 10`."
    result = 1
    for x in numbers:
        result *= x
    return result


"""# Inferência por enumeração (visto em sala)

`P(variable, evidence)` serve para sabermos a probabilidade de uma variável quando todas as outras da rede são evidências. Mas e quando não são? Uma técnica (vista em sala) é a inferência por enumeração. A função a seguir `enumeration_ask` possui esse propósito. 
"""


def enumeration_ask(X, evidence, net):
    "The probability distribution for query variable X in a belief net, given evidence."
    i = net.variables.index(X)  # The index of the query variable X in the row
    dist = defaultdict(float)  # The resulting probability distribution over X
    for (row, p) in joint_distribution(net).items():
        if matches_evidence(row, evidence, net):
            dist[row[i]] += p
    return ProbDist(dist)


def matches_evidence(row, evidence, net):
    "Does the tuple of values for this row agree with the evidence?"
    return all(evidence[v] == row[net.variables.index(v)]
               for v in evidence)


"""# Exemplo do Alarme"""

# CPTable({(T, T): .95,
#             (T, F): .94,
#             (F, T): .29,
#             (F, F): .001},
#         [Burglary, Earthquake])

alarm_net = (BayesNet()
             .add('Burglary', [], 0.001)
             .add('Earthquake', [], 0.002)
             .add('Alarm', ['Burglary', 'Earthquake'], {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001})
             .add('JohnCalls', ['Alarm'], {T: 0.90, F: 0.05})
             .add('MaryCalls', ['Alarm'], {T: 0.70, F: 0.01}))

# Tornar Burglary, Earthquake, etc. variáveis globais
globals().update(alarm_net.lookup)

# """# Exemplos de uso das classes e funções"""

# Earthquake = Variable('Earthquake', 0.002)

# P(Earthquake)

# P(Earthquake)[T]

# # Amostragem aleatória:
# sample(P(Earthquake))

# # 100 mil amostragens:
# Counter(sample(P(Earthquake)) for i in range(100000))

# # Duas maneiras equivalentes de se especificar a mesma distribuição booleana:
# assert ProbDist(0.75) == ProbDist({T: 0.75, F: 0.25})
# ProbDist(0.75)

# # # Duas maneiras equivalentes de se especificar a mesma distribuição NÃO booleana:
# assert ProbDist(win=15, lose=3, tie=2) == ProbDist({'win': 15, 'lose': 3, 'tie': 2})
# ProbDist(win=15, lose=3, tie=2)

# # A diferença entre um Factor e uma ProbDist -- a ProbDist é normalizada:
# Factor(a=1, b=2, c=3, d=4)

# ProbDist(x=1, y=2, z=3, k=4)