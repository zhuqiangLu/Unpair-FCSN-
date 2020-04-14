import matplotlib.pyplot as plt
import numpy as np 


a = np.array([1,2,3,4,5,5,6,7])
b = a+1
c = a-1

plt.plot(a, label = 'a')

plt.plot(b, label = 'b')

plt.plot(c, label = 'c')
plt.legend()
plt.savefig('foo.png')