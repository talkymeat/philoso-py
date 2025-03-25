from philoso_py import ModelFactory
from datetime import datetime

print('GO!')
t0 = datetime.now()
ModelFactory().run_json('model_json/model1_mps.json')
t1 = datetime.now()
ModelFactory().run_json('model_json/model1.json')
t2 = datetime.now()
print(t1-t0, 'mps')
print(t2-t1, 'cpu')