from base_tree import Tree

class BinaryTree(Tree):
    """ Abstract base class representing a binary tree structure. """

    # ----- abstract methods -----
    def left(self, p):
        """
        Return a position representing p's left child.

        Return None if p does not have a left child.
        """
        raise NotImplementedError('Must be implemented by subclass.')

    def right(self, p):
        """
        Return a position representing p's right child.

        Return None if p does not have a right child.
        """
        raise NotImplementedError('Must be implemented by subclass.')

    # ----- concrete methods implemented in this class -----
    def sibling(self, p):
        """ Return a Position representing p's sibling (or None if no sibling). """
        parent = self.parent(p)
        if parent is None:
            return None
        else:
            if p == self.left(parent):
                return self.right(parent) # possibly None
            else:
                return self.left(parent) # possibly None

    def children(self, p):
        """ Generate an iteration of Positions representing p's children. """
        if self.left(p) is not None:
            yield self.left(p)
        if self.right(p) is not None:
            yield self.right(p)
