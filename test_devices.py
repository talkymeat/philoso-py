from philoso_py import ModelFactory

model = ModelFactory().from_json('model_json/model1_gpu.json')

for agent in model.agents:
    for name, net in model.agents[0].nn.policy_layers.items():
        for param in net.parameters():
            assert str(param.device)[:4] == 'cuda'

    for name, net in model.agents[0].nn.policy_heads.items():
        for nom, head in net.items():
            for param in head.parameters():
                assert str(param.device)[:4] == 'cuda'

print('devices test OK')