import torch
from torchvision.datasets import FakeData
import numpy as np
from src.models.utils.patch_embed import PatchEmbed3D

fakedata = FakeData(size=10)

data = []

for d, label in fakedata:
    data.append(np.array(d).transpose(2, 0, 1))


concat = np.stack(data).transpose(1, 0, 2, 3)


data = []
for _ in range(4):
    data.append(concat)

concat = np.stack(data)

print(concat.shape)

net = PatchEmbed3D()

out = net(torch.tensor(concat).float())

print(out.shape)
