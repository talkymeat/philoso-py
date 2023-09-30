# import json 
# from typing import Protocol, runtime_checkable

# """
# IOs take Treebanks as args when writing
# IOs take Treebanks as optional args when reading, if none provided, create new
# Treebank: if one is provided, add new trees to old, no need for `clear` arg
# """

# @runtime_checkable
# class TreebankIO(Protocol):
#     # def __init__(self, tree_type: type):
#     #     ...

#     def load(cls, filename: str, clear: bool):
#         ...

#     def save(cls, filename: str):
#         ...


# @runtime_checkable
# class JSONiser(Protocol):
#     # def __init__(self, tree_type: type):
#     #     ...

#     def load(cls, filename: str, clear: bool):
#         ...

#     def save(cls, filename: str):
#         ...

#     def to_json(self) -> json:
#         ...

#     def from_json(self, json_treebank: str, clear: bool):
#         ...


# class JSONIO:
#     """Translates between a Treebank and JSON: saves and loads a Treebank
#         to/from a JSON file.

#     Attributes:
#         tree_type (type): Specify the tree-system you wish to use (e.g. Tree,
#         DOP, GP, etc) by passing in the class name here: this ensures that the
#         IO writes from the correct Label-class, and when reading generates trees
#         using the correct Terminal and NonTerminal classes. Use the class
#         itself, not an instance of the class. DEPRECATED

#     Raises:
#         TypeError: If the value passes is not of type `type` (that is, not a
#             class), or if the type is not Tree or one of its subclasses.
#     """
#     # def __init__(self, tree_type: type):
#     #     if isinstance(tree_type, 'type') and issubclass(tree_type, Tree):
#     #         self._tree_type = tree_type
#     #     elif isinstance(tree_type, 'Tree'):
#     #         er = "tree_type should be the class Tree or one of its subclasses: "
#     #         raise TypeError(er + "not an instance of the class.")
#     #     elif isinstance(tree_type, 'type'):
#     #         raise TypeError(er + f"not a {tree_type}")
#     #     else:
#     #         raise TypeError(er + f"not a {type(tree_type)}")

#     # NOE
#     # @property
#     # def _all_roots(treebank):
#     #     """An alias for `get_all_root_nodes` in `Treebank`"""
#     #     return self._tree_type.L().get_all_root_nodes()

#     def to_json(self, treebank) -> json:
#         """Gets a JSON representation of the entire treebank: that is, for each
#         Label, JSON representations of all root nodes with that Label.

#         Returns
#         -------
#             (str): string containing the json expression for the entire treebank
#         """
#         # Label.get_all_root_nodes returns a dict with lists of Trees as values,
#         # but that's not quite what we want; the dict and list comprehensions
#         # here convert the trees to strings.
#         return json.dumps(
#             {k: [str(t) for t in v] for k, v in treebank._all_roots if v},
#             indent = 4
#         )

#     def save(self, filename: str):
#         """Saves the JSON representation of the treebank to a file.

#         Parameters
#         ----------
#             filename (str): name of file to be saved
#         """
#         with open(filename, 'w') as outfile:
#             json.dump(self.to_json(), outfile)

#     def from_json(self, json_treebank: str, clear: bool):
#         """Populates an entire treebank from JSON

#         Parameters
#         ----------
#             json (str):
#             clear (bool): Ensures there are no other trees already loaded before
#                 proceeding, if True.
#         """
#         if clear:
#             self._L.clear()
#         trees_dict = json.loads(json_treebank)
#         for val in trees_dict.values():
#             for tree in val:
#                 self._tree_type.tree(tree)

#     def load(self, filename: str, clear: bool):
#         """Reads in a JSON file for an entire treebank, populates the treebank.
#         """
#         with open(filename) as json_treebank:
#             self.from_json(json_treebank.read(), clear, tree_type)
