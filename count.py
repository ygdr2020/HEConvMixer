import torch
from thop import profile
from models import build_model

model = build_model()
dummy_input = torch.randn(1, 3, 32, 32)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
