from .Node import Node


class TreeNode(Node) :
    def __init__(self, children = [], value = None) :
        self.children = children
        super().__init__(value)

    def __repr__(self) :
        return f"{self.value}"
