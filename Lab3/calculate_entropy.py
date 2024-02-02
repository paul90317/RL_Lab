import numpy as np
from torch.distributions import Categorical
import torch
temp = torch.tensor(np.array([0.0829, 0.1108, 0.1595, 0.0854, 0.1162, 0.0859, 0.1069, 0.1010, 0.1514]))
print(torch.sum(-temp*torch.log(temp)))
dist = Categorical(probs=temp)
print(dist.probs)
print(dist.entropy())