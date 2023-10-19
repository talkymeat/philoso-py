
class SizeDepth:
    def __init__(
        self,
        size: int,
        depth: int,
        max_size: int = 0,
        max_depth: int = 0
    ):
        self._size = size
        self._depth = depth
        self.max_size = max_size
        self.max_depth = max_depth

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        if size > self.max_size:
            return False
        self._size = size
        return True

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, depth):
        if depth > self.max_depth:
            return False
        self._depth = depth
        return True
    
    def __call__(self, size, depth):
        if (self.max_size and size > self.max_size) or (self.max_depth and depth > self.max_depth):
            return False
        self.size, self._depth = size, depth
        return True
    
    def __str__(self):
        return f"SizeDepth(size={self.size}, depth={self.depth}, max_size={self.max_size}, max_depth={self.max_depth})"
            

