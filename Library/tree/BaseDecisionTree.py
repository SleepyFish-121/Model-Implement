from DataStructure import Tree, TreeNode
from ..base.BaseEstimator import BaseEstimator

class BaseDecisionTree(Tree, BaseEstimator):
    def __init__(self, criterion, max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        super().__init__()
    def add_Node(self, input, X, y, func, depth, max_depth):
        if depth == max_depth:
            return TreeNode(value=y)
        if len(X) == 0:
            return None
        if len(set(y)) == 1:
            value = (input, y[0])
            return TreeNode(children=None, value=value)
        n = func(X, y)
        label = X.columns[n]
        children = []
        for value in set(X[label]):
            children.append(
                self.add_Node(value, X[X[label] == value].copy(), y[X[label] == value].copy(), func, depth + 1,
                              max_depth))
        value = (input, label)
        return TreeNode(children=children, value=value)
    def fit(self, X, y):
        pass
