
import numpy as np
import matplotlib.pyplot as plt
import torch

def GenerateData(N,mean,variance):
  data = []
  for i in range(len(mean)):
    print(mean[i])
    print(variance[i])
    temp = np.random.multivariate_normal(mean[i],variance[i],N)
    data.append(temp)
    print("=============")
  return data

dim = 2

mean1 = np.array([0,-5])
mean2 = np.array([0,5])
mean3 = np.array([5,0])
mean4 = np.array([5,10])

mean = [mean1,mean2,mean3,mean4]

var1 = np.diag(np.ones(dim))

var = [var1,var1,var1,var1]
data = GenerateData(1000,mean,var)
print(len(data))
print(data[0][0:5])
print(data[1][0:5])
print(data[2][0:5])
print(data[3][0:5])



plt.figure()
plt.plot(data[0][:,0],data[0][:,1],'.',color='r')
plt.plot(data[1][:,0],data[1][:,1],'.',color='y')
plt.plot(data[2][:,0],data[2][:,1],'.',color='g')
plt.plot(data[3][:,0],data[3][:,1],'.',color='b')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

