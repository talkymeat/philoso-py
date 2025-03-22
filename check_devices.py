from philoso_py import ModelFactory

model = ModelFactory().from_json('model_json/model1_gpu.json')

for name, net in model.agents[0].nn.policy_heads.items():
    print(name) 
    for param in net.parameters():
        print('---', param.device)

for name, net in model.agents[0].nn.policy_heads.items():
    print(name)
    for nom, head in net.items():
        print('---', nom) 
        for param in head.parameters():
            print('-------', param.device)