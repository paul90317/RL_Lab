import matplotlib.pyplot as plt
plt.xlabel('episode')
plt.ylabel('score')
with open('log.txt','r') as f:
    y=[float(i) for i in f.readlines()[:100]]
x=[i*1000 for i in range(len(y))]
plt.plot(x,y)
plt.show()