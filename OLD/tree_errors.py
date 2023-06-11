

class TreeIndexError(IndexError):
    def __init__(self, msg, tree_str = None, idx = None, cutoff = None):
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
