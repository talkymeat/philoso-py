from repository import *
from test_materials import AG, T0
from model_time import ModelTime
import numpy as np
from icecream import ic

mt=ModelTime()

mem = Archive(
    cols=['goodness'], 
    rows=10,
    model_time=mt,
    types={'goodness': np.float64},
    tables=2,
    value='goodness',
)

plos_one = Publication(
    cols=['goodness'], 
    rows=10,
    model_time=mt,
    users=AG,
    types={'goodness': np.float64},
    tables=2,
    value='goodness',
    reward='ranked'
)

# def aeq(a, b, eta=1e-6):
#     return abs(ic(ic(a)-ic(b))) < eta
def aeq(a, b, eta=1e-6):
    return abs(a-b) < eta

def main():
    agent_nos = [
        3, 5, 2, 4, 
        3, 8, 1, 
        2, 6, 7
    ]
    goodness = [
        0.5, 0.1, 0.4, 0.6, 
        0.8, 0.7, 1.0, 
        0.9, 1.3, 1.2
    ]
    expected_t = [
        8, 9, 6, 7,
        4, 5, 3,
        0, 2, 1
    ]
    expected_rew = [
        0.0, 1.0, 1.8, 2.0,
        1.0, 0.9, 1.0, 
        0.9, 0.9, 0.0
    ]
    exp_agent_order = [
        'a6', 'a7', 'a1', 'a2', 
        'a3', 'a8', 'a4', 
        'a3', 'a2', 'a5'
    ]

    for ag, gd in zip(agent_nos, goodness):
        plos_one.insert_tree(T0.copy(), AG[ag].name, journal=0, goodness=gd)
        mt.tick()


    assert aeq(plos_one._agents['reward'].copy().reset_index(drop=True), (pd.Series(expected_rew))).all(), f"""Expected:
    {expected_rew}
    Got:
    {list(plos_one._agents['reward'])}
    """ 

    assert aeq(plos_one.tables[0]['t'].copy().reset_index(drop=True), (pd.Series(expected_t))).all(), f"""Expected:
    {expected_t}
    Got:
    {list(plos_one.tables[0]['t'])}
    """ 

    for i in range(10):
        for a in range(10):
            if f"a{a}" == exp_agent_order[i]:
                assert plos_one.tables[0].loc[i, f"a{a}"], f"False positive at row {i}, col a{a}"
            else:
                assert not plos_one.tables[0].loc[i, f"a{a}"], f"False negative at row {i}, col a{a}"

    plos_one.insert_tree(T0.copy(), AG[0].name, journal=0, goodness=0.45)
    mt.tick()

    assert aeq(plos_one._agents.loc['a0', 'reward'], 0.2)
    assert aeq(plos_one._agents.loc['a5', 'reward'], 0.9-((0.95**9)*0.9)), f'Oops, {plos_one._agents.loc['a5', 'reward']}'

if __name__ == '__main__':
    main()