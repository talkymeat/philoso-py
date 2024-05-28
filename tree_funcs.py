from collections import deque
from m import MDict
from functools import reduce
from trees import Tree

from icecream import ic


# XXX make maxlen a hyperparam
def sign_tree(tree: Tree, agent, *args, **kwargs):
    """This takes a tree-node and an agent (the agent that created the tree) and appends the id
    of the agent to the list of 'credits' in the tree-nodes metadata dict. The credits are 
    recorded in a fixed-length (len=4) dict, so when a 5th id is added, the oldest id in the
    credits is dropped. 
    
    This function is designed to be used with Tree.apply

    >>> from test_materials import T0, AG, to_list, get_credit_pos
    >>> T0.apply(sign_tree, AG[5])
    >>> T0.apply(sign_tree, AG[7])
    >>> get_credit_pos(T0, 0)
    'a7'
    >>> get_credit_pos(T0, 1)
    'a5'
    >>> get_credit_pos(T0[0,0,0,0], 0)
    'a7'
    >>> get_credit_pos(T0[0,0,0,0], 1)
    'a5'
    >>> L7 = T0.tree_map_reduce(to_list, 0, map_any=get_credit_pos)
    >>> L7
    ['a7', 'a7', 'a7', 'a7', 'a7', 'a7', 'a7', 'a7', 'a7', 'a7', 'a7', 'a7', 'a7', 'a7', 'a7']
    >>> len(L7) == T0.size()
    True
    >>> L5 = T0.tree_map_reduce(to_list, 1, map_any=get_credit_pos)
    >>> L5
    ['a5', 'a5', 'a5', 'a5', 'a5', 'a5', 'a5', 'a5', 'a5', 'a5', 'a5', 'a5', 'a5', 'a5', 'a5']
    >>> len(L5) == T0.size() 
    True
    """
    if 'credit' in tree.metadata:
        tree.metadata['credit'].appendleft(agent.id)
    else:
        tree.metadata['credit'] = deque([agent.id], 4)

def init_reward_tree(tree: Tree, rew, *args, **kwargs):
    tree.metadata['init_reward'] = rew

def calculate_credit(tree: Tree, except_id, *args, **kwargs):
    """This gathers the assignment of credit from a single tree-node and compiles it into a dict
    of credit values due to agents *other* than the one that created the tree. A value of 0.5
    is assigned to the most recently credited agent (IF that agent is not the agent posting the
    tree), 0.25 to the agent before, and 0.125 to any agents listed before that.

    This function is designed to be used as a mapping function with Tree.tree_map_reduce

    >>> from test_materials import T0, AG, to_list, get_credit_pos
    >>> T0.apply(sign_tree, AG[5])
    >>> T0[0,0].apply(sign_tree, AG[8])
    >>> T0.apply(sign_tree, AG[3])
    >>> T0[1].apply(sign_tree, AG[1])
    >>> T0[0,0,1,0].apply(sign_tree, AG[9])
    >>> T0.apply(sign_tree, AG[4])
    >>> T0[0,0,1,0,1].apply(sign_tree, AG[9])
    >>> T0.apply(sign_tree, AG[1])
    >>> for pos in range(4):
    ...     print(T0.tree_map_reduce(to_list, pos, map_any=get_credit_pos))
    ['a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1']
    ['a4', 'a4', 'a4', 'a9', 'a4', 'a4', 'a4', 'a4', 'a4', 'a4', 'a4', 'a4', 'a4', 'a4', 'a4']
    ['a3', 'a3', 'a9', 'a4', 'a9', 'a3', 'a3', 'a3', 'a3', 'a1', 'a1', 'a1', 'a1', 'a1', 'a3']
    ['a8', 'a8', 'a3', 'a9', 'a3', 'a8', 'a8', 'a8', 'a5', 'a3', 'a3', 'a3', 'a3', 'a3', 'a5']
    >>> calculate_credit(T0[0,0,0,0], 'a10')
    {'a1': 0.5, 'a4': 0.25, 'a3': 0.125, 'a8': 0.125}
    >>> calculate_credit(T0[0,0,0,0], 'a1')
    {'a4': 0.25, 'a3': 0.125, 'a8': 0.125}
    >>> calculate_credit(T0[0,0,0,0], 'a3')
    {'a1': 0.5, 'a4': 0.25, 'a8': 0.125}
    >>> calculate_credit(T0[1], 'a10')
    {'a1': 0.625, 'a4': 0.25, 'a3': 0.125}
    >>> calculate_credit(T0[1], 'a1')
    {'a4': 0.25, 'a3': 0.125}
    >>> credit = calculate_credit(T0[1], 'a4')
    >>> credit
    {'a1': 0.625, 'a3': 0.125}
    >>> sum(credit.values())
    0.75
    """
    credits = tree.metadata.get('credit', [])
    cr_dict = MDict()
    rewards = [0.5, 0.25, 0.125, 0.125]
    for id, reward in zip(credits, rewards):
        if id != except_id:
            cr_dict[id] = cr_dict.get(id, 0) + reward
    return cr_dict

def sum_all(self_val, *child_vals, **kwargs):
    """This function sums values, and is intended for use as a reduce_func in 
    Tree.tree_map_reduce
    
    >>> from test_materials import T0, AG, to_list, get_credit_pos
    >>> T0.apply(sign_tree, AG[5])
    >>> T0[0,0].apply(sign_tree, AG[8])
    >>> # T0.tree_map_reduce(sum_all, 10, map_any=calculate_credit)
    {'a8': 4.0, 'a5': 5.5}
    >>> # T0.tree_map_reduce(sum_all, 8, map_any=calculate_credit)
    {'a5': 5.5}
    >>> T0.apply(sign_tree, AG[3])
    >>> T0[1].apply(sign_tree, AG[1])
    >>> # T0.tree_map_reduce(sum_all, 10, map_any=calculate_credit)
    {'a3': 5.25, 'a8': 2.0, 'a5': 2.0, 'a1': 2.5}
    >>> T0[0,0,1,0].apply(sign_tree, AG[9])
    >>> T0.apply(sign_tree, AG[4])
    >>> # T0.tree_map_reduce(sum_all, 10, map_any=calculate_credit)
    {'a4': 7.5, 'a3': 2.75, 'a8': 1.0, 'a5': 1.5, 'a9': 0.75 'a1': 1.25}
    >>> T0[0,0,1,0,1].apply(sign_tree, AG[9])
    >>> T0.apply(sign_tree, AG[1])
    >>> # T0.tree_map_reduce(sum_all, 10, map_any=calculate_credit)
    {'a1': 7,875, 'a4': 3.625, 'a3': 1.75, 'a8': 0.625, 'a9': 0.625, 'a5': 0.25}
    >>> # T0.tree_map_reduce(sum_all, 1, map_any=calculate_credit)
    {'a4': 3.625, 'a3': 1.75, 'a8': 0.625, 'a9': 0.625, 'a5': 0.25}
    >>> # T0.tree_map_reduce(sum_all, 8, map_any=calculate_credit)
    {'a1': 7,875, 'a4': 3.625, 'a3': 1.75, 'a9': 0.625, 'a5': 0.25}
    """
    return reduce(lambda x, y: x+y, child_vals + (self_val,))

def unions_for_all(*sets, **kwargs):
    """This function collects the union of `set` valuess, and is intended for 
    use as a reduce_func in Tree.tree_map_reduce. Also, Solidarity!
    
    """
    return reduce(lambda x, y: x|y, sets)

def get_operators(tree: Tree):
    """Gathers all the Operators on the NonTerminals of `tree`
    
    >>> from test_materials import T0
    >>> ops = get_operators(T0)
    >>> print(type(ops).__name__)
    set
    >>> opnames = [op.name for op in ops]
    >>> opnames.sort()
    >>> opnames
    ['PROD', 'SQ', 'SUM']
    """
    return tree.tree_map_reduce(
        reduce_func=unions_for_all, 
        map_non_terminal=lambda t: {t._operator}, 
        map_terminal=lambda t: set(), 
        map_subsite=lambda t: set()
    )


def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()