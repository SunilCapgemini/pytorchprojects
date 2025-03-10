import torch

Q = torch.arange(6*48).reshape(6,48)
P = Q.view(6,3,16)
R = P.transpose(0,1)
J= Q.view(3,6,16)
print('Q',Q)
print('P',P)
print('R',R)
print('J',J)