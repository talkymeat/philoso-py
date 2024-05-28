class TreeIndexError(IndexError):
    """Exception raised when a `Tree` is indexed with an invalid 
    index

    Attributes
    ----------
        msg (str): 
            explanatory error message
        tree_str (str):
            String representation of the `Tree` that raised the error
        idx (tuple of ints):
            The invalid index
        cutoff (int):
            Index of the `tuple` `idx` indicating the first value of
            `idx` that fails to index a branch of the `Tree`
    """
    def __init__(
            self, msg: str, 
            tree_str: str = None, 
            idx: tuple[int] = None, 
            cutoff: int = None
        ):
        self.msg = msg
        self.tree_str = tree_str
        self.idx = idx
        self.cutoff = cutoff

    def __str__(self):
        if self.tree_str and self.idx and self.cutoff:
            return (
                f"Invalid index. The Tree:\n{self.tree_str}\nwas "+
                f"indexed with {self.idx}, but has no subtree at "+
                f"{self.idx[:self.cutoff]}. {self.msg}"
            )
        else:
            return self.msg

    def update(self, parent_str, parent_idx = None, child_idx = None):
        """If a `TreeIndexError` is raised, it may happen a few recursive steps
        down the `Tree` from the node at which the attempt was made to retrieve
        and invalid index: it is better, however, to have the node at which the
        initial call was made throw the error: therefore, the recursive call is
        made in a `try ... except` block, which updates the error with indexing
        information relevant to the parent node

        Parameters
        ----------
            parent_str (str):
                string representation of the catching-and-re-raising node
            parent_idx (int): 
                the index of the current node, above the original raising node
            child_idx (int):
                the index of the current node, below the original raising node
        """
        self.tree_str = parent_str
        if not parent_idx is None:
            self.idx = (parent_idx,) + self.idx
            if self.cutoff >= 0:
                self.cutoff += 1
        if not child_idx is None:
            self.idx = self.idx + (child_idx,)
            if self.cutoff < 0:
                self.cutoff -= 1


class OperatorError(Exception):
    def __init__(self, msg):
        self.treestr = None
        self.msg = msg

    def __str__(self):
        if self.treestr:
            return f"Error in {self.treestr}: {self.msg}"
        
class UserIDCollision(Exception):
    def __init__(self, msg, id=None):
        self.id  = id
        self.msg = msg

    def __str__(self):
        clash = ""
        if self.id:
            clash = f"Two distinct Agents share the ID {self.id}. "
        return clash + self.msg
