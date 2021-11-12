from .TreeNode import TreeNode


class Tree(TreeNode):
    def __init__(self, root=None):
        self.root = root
        self.children = []

    def visualize_with_graphviz(self, graph_name):
        pass
