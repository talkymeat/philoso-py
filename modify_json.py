import json, os
from json_factory import IDCoder


def mod_json(world, gp_vars_more_fn, device, sources, idc):
    for src in sources:
        with open(f"model_json/{src}") as f:
            jsrc = json.load(f)
        jsrc['world'] = world
        jsrc['gp_vars_more'] = gp_vars_more_fn(jsrc['gp_vars_more'])
        pref = next(idc)
        jsrc['output_prefix'] = f"{pref}__"
        jsrc['out_dir'] = f"output/{pref}"
        jsrc['model_id'] = pref
        for k in jsrc['agent_templates'].keys():
            jsrc['agent_templates'][k]['device'] = device
        os.makedirs('mdls', exist_ok=True)
        with open(f'mdls/model_{pref}.json', 'w') as outf:
            json.dump(jsrc, outf, indent = 4)

if __name__ == '__main__':
    sources = ['model_aa.json', 'model_am.json', 'model_an.json', 'model_ao.json']
    idc = IDCoder(2, i=40)
    worlds = ['SineWorld2', 'SineWorld3']
    gvmfs = [
        lambda gvm: gvm[:-1],
        lambda gvm: gvm[:-3] + ['obs_centre', 'obs_radius']
    ]
    for w, g in zip(worlds, gvmfs):
        mod_json(w, g, 'mps', sources, idc)

# world:
# 1 - SineWorld2
# 2 - SineWorld3

# gp_vars_more:
# 1 - lose last (obs_num)
# 2 - obs_start, obs_stop -> obs_centre, obs_radius

# output_prefix bo__ - bv__
# out_dir output/bo - output/bv