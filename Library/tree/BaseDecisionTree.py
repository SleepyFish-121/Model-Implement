from collections import Counter

import numpy as np

from DataStructure import Tree, TreeNode
from ..base.BaseEstimator import BaseEstimator


class BaseDecisionTree(Tree, BaseEstimator) :
    def __init__(self, criterion, max_depth = None) :
        self.criterion = criterion
        self.max_depth = max_depth
        super().__init__()

    def add_Node(self, input, X, y, func, depth, max_depth) :
        if depth == max_depth :
            return TreeNode(value = y)
        if len(X) == 0 :
            return None
        if len(set(y)) == 1 :
            value = (input, y[0])
            return TreeNode(children = None, value = value)
        n = func(X, y)
        label = X.columns[n]
        children = []
        for value in set(X[label]) :
            children.append(
                self.add_Node(value, X[X[label] == value].copy(), y[X[label] == value].copy(), func, depth + 1,
                              max_depth))
        value = (input, label)
        return TreeNode(children = children, value = value)

    def predict(self, X) :
        def solve_unknown(node) :
            values = []
            if node.children == None :
                values.append(node.value[1])
            else :
                for child in node.children :
                    values.append(solve_unknown(child))
            return Counter(values).most_common(1)[0][0]

        def predict_single(X) :
            attribute = self.root.value[1]
            node = self.root
            while node.children != None :
                if X[attribute] in [child.value[0] for child in node.children] :
                    node = node.children[[child.value[0] for child in node.children].index(X[attribute])]
                else :
                    return solve_unknown(node)
                attribute = node.value[1]
            return node.value[1]

        results = []
        for i in X.index :
            results.append(predict_single(X.loc[i]))
        return np.array(results)
