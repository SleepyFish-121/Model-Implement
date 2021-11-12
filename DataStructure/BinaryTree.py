from .Tree import Tree


class BinaryTree(Tree):
    def __init__(self, root=None, left=None, right=None):
        self.root = root
        self.left = left
        self.right = right
        Tree.super()
        del self.children
