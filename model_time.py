

class ModelTime:
    def __init__(self) -> None:
        self._t = 0

    def __str__(self):
        return f'{self._t}'
    
    def __call__(self):
        return self._t
    
    def tick(self):
        self._t += 1

    def __eq__(self, other):
        return self._t == other

    def __gt__(self, other):
        return self._t > other

    def __lt__(self, other):
        return self._t < other
    
    def reset(self):
        self._t = 0