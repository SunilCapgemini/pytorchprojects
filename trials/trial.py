import torch
outputs = torch.arange(2*3*4).reshape((2, 3, 4))
print(outputs)
permuted_outputs = outputs.permute(0,2,1)
print(permuted_outputs)
outputs = torch.arange(2*3*4).reshape((2, 3, 4))
reshaped_outputs = outputs.reshape((2,4,3))
print(reshaped_outputs)