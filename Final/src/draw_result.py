from torch.utils.tensorboard import SummaryWriter
import os,sys
writer = SummaryWriter('log')
result = {}
for filename in os.listdir('models'):
    _,i,rew,_=filename.split('_')
    result[int(i)] = float(rew) / 100

for i in range(1, 101):
    writer.add_scalar('eval/reward', result[i], 4096 * 4 * i)



