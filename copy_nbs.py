from shutil import copyfile as cp

ags = [f'a_{i}' for i in range(8)]
mdls = [f'b{ch}' for ch in 'qrstuv']

for ag in ags:
    for mdl in mdls:
        cp('_SineWorldOutputAnalyser_.ipynb', f'_SineWorldOutputAnalyser_{mdl}_{ag}.ipynb')