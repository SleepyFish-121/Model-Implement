from math import log2

import graphviz
import numpy as np
import pandas as pd

from .BaseDecisionTree import BaseDecisionTree


class DecisionTreeClassifier(BaseDecisionTree) :
    def __init__(self, max_depth = None, algorithm = 'ID3', criterion = "entropy") :
        self.algorithm = algorithm
        super().__init__(criterion = criterion, max_depth = max_depth)

    def __repr__(self) :
        return f"Max depth: {self.max_depth}"

    def fit(self, X: pd.DataFrame, y: np.array) :
        # criterion paramter disabled, use entrophy instead
        self.root = self.add_Node(None, X, np.array(y), ID3, 0, self.max_depth)
        return self

    def visualize_with_graphviz(self, graph_name = 'Tree Graph') :
        global dot
        dot = graphviz.Digraph(comment = graph_name)
        global node
        node = 0

        def add_node(parent, root, attribute) :
            global dot
            global node
            node = node + 1
            if root.children == None :
                dot.node(str(node), f"{root.value[1]}", color = 'red')
                dot.edge(str(parent), str(node))
                return
            if parent == None :
                dot.node(str(node), f"root")
            else :
                dot.node(str(node), f"\"{attribute}\"={root.value[0]}")
                dot.edge(str(parent), str(node))
            current = node
            for child in root.children :
                add_node(current, child, root.value[1])
            return

        add_node(None, self.root, None)
        return dot

def ID3(X, y) :
    calculate_entrophy = lambda a : -1 * (
        np.sum([a[i] / (sum(a) + 1e-5) * log2(a[i] / (sum(a) + 1e-5) + 1e-5) for i in range(len(a))]))
    entrophies = []
    data_dummies = pd.get_dummies(
        pd.DataFrame(np.hstack([np.array(X), np.array(y).reshape(-1, 1)]), columns = list(X.columns) + ['_Response']))
    for label in X.columns :
        HS = 0
        for value in list(set(X[label])) :
            a = []
            for y_value in list(set(y)) :
                a.append(len((data_dummies[np.logical_and(data_dummies[str(label) + '_' + str(value)] == 1,
                                                          data_dummies['_Response_' + y_value] == 1)])))
            p = len(data_dummies[(data_dummies[str(label) + '_' + str(value)] == 1)]) / len(X)
            HS = p * calculate_entrophy(a) + HS
        entrophies.append(HS)
    return np.argmin(entrophies)
