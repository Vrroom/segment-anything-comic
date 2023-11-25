"""
Used for visualizing the optimal matching between two trees.
"""
import networkx as nx
from uuid import uuid4
from networkx.algorithms import bipartite
from more_itertools import unzip

def optimalBipartiteMatching (costTable) :
    """
    Return the minimum cost bipartite matching.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the cost of matching 
        that pair together.
    """
    A, B = list(map(set, unzip(costTable.keys())))
    # Rename the elements of sets A and B uniquely and record
    # the mapping. If the set elements have common names, then
    # they won't clash while computing the bipartite matching.
    aMap = dict(zip(A, [str(_) + "-" + str(uuid4()) for _ in A]))
    aInvMap = dict(map(reversed, aMap.items()))
    bMap = dict(zip(B, [str(_) + "-" + str(uuid4()) for _ in B]))
    bInvMap = dict(map(reversed, bMap.items()))
    # Create bipartite graph with given edge costs.
    G = nx.Graph()
    for key, cost in costTable.items() :
        i, j = key
        G.add_node(aMap[i], bipartite=0)
        G.add_node(bMap[j], bipartite=1)
        G.add_edge(aMap[i], bMap[j], weight=cost)
    # Solve the bipartite matching problem and remove duplicate 
    # edges from the matching.
    matchingWithDups = bipartite.minimum_weight_full_matching(G)
    matching = dict()
    for i in A : 
        if aMap[i] in matchingWithDups : 
            matching[i] = bInvMap[matchingWithDups[aMap[i]]]
    return matching

def bestAssignmentCost (costTable) :
    """
    Compute the minimum total cost assignment.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the cost of matching that pair
        together.
    """
    matching = optimalBipartiteMatching(costTable)
    cost = sum([costTable[e] for e in matching.items()])
    return cost

