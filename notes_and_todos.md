##NOTES and TODOS

#TODO

* Allow to_LaTeX to make qtrees for tree-fragments.
* ~~Can I use a tuple as an index? a list? if not, mod the code so I can.~~ YES
* make treebank a \@property of Terminal and NonTerminal, only have it as an attribute for Label
* Add metadata attributes to Terminal, NonTerminal, and Label
* In NonTerminal.@children.setter, check the children belong to the same treebank
* Does Label have an __eq__ method?
* better error handling for __getitem__, so it's clear where the problem is
* adding a node to a Label and the Label to the node without tangled-up dependencies? can this be done?
* JSONIO needs redone!
* tree creation should by default use the Treebank default
* write test for `get_all_root_nodes`

* __in__
