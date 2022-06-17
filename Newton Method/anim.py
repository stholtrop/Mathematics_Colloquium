import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors

aW = 1
'''a = 0
b = 2.5
theta = np.pi/3'''#stability issues

a = -1
b = 2
theta = np.pi/2

def sin(xs):
  return np.sin(sum(xs))

  
def cos(xs):
  return np.cos(sum(xs))

f4 = lambda x : 1.5 + np.arctan(x)/np.pi
Df4 = lambda x : 1.0 / (np.pi * (1 + x**2))
'''
f  = lambda x : np.array([-f4(x[3])*sin(x[0:3]) - sin(x[0:2]) - sin(x[0:1]) - a, f4(x[3])*cos(x[0:3]) + cos(x[0:2]) + cos(x[0:1]) - b, (aW * (sum(x[0:3]) - theta)) % (2 * np.pi)])
Df = lambda x : np.array([[-Df4(x[3])*cos(x[0:3]) - cos(x[0:2]) - cos(x[0:1]), -Df4(x[3])*cos(x[0:3]) - cos(x[0:2]), -Df4(x[3])*cos(x[0:3]), - Df4(x[3]) * sin(x[0:3])],
                          [-Df4(x[3])*sin(x[0:3]) - sin(x[0:2]) - sin(x[0:1]), -Df4(x[3])*sin(x[0:3]) - sin(x[0:2]), -Df4(x[3])*sin(x[0:3]), Df4(x[3]) * cos(x[0:3])],
                          [aW, aW, aW, 0]])'''

f  = lambda x : np.array([-f4(x[3])*sin(x[0:3]) - sin(x[0:2]) - sin(x[0:1]) - a, f4(x[3])*cos(x[0:3]) + cos(x[0:2]) + cos(x[0:1]) - b, cos(x[0:3]) - np.cos(theta), sin(x[0:3]) - np.sin(theta)])
Df = lambda x : np.array([[-Df4(x[3])*cos(x[0:3]) - cos(x[0:2]) - cos(x[0:1]), -Df4(x[3])*cos(x[0:3]) - cos(x[0:2]), -Df4(x[3])*cos(x[0:3]), - Df4(x[3]) * sin(x[0:3])],
                          [-Df4(x[3])*sin(x[0:3]) - sin(x[0:2]) - sin(x[0:1]), -Df4(x[3])*sin(x[0:3]) - sin(x[0:2]), -Df4(x[3])*sin(x[0:3]), Df4(x[3]) * cos(x[0:3])],
                          [-sin(x[0:3]), -sin(x[0:3]), -sin(x[0:3]), 0],
                          [cos(x[0:3]), cos(x[0:3]), cos(x[0:3]), 0]])

# Newton
niter = 500

plt.xlim(-4,4)
plt.ylim(-4, 4)

def xyPos(x):
  return [[0, - sin(x[0:1]), - sin(x[0:2]) - sin(x[0:1]), -f4(x[3])*sin(x[0:3]) - sin(x[0:2]) - sin(x[0:1])],
         [0,   cos(x[0:1]),   cos(x[0:2]) + cos(x[0:1]),  f4(x[3])*cos(x[0:3]) + cos(x[0:2]) + cos(x[0:1])]]

line, = plt.plot([], [])

def start():
  line.set_data(*xyPos(X[0]))
  return line,

def animate(i):
  if (i % niter == 0):
    X[0] = np.array([0,1.33333 * np.pi,0,1.5])#np.random.randn(4)
    for k in range(niter-1):
      X[k+1] = X[k] - 1.0 * np.linalg.pinv(Df(X[k]))@f(X[k])
  line.set_data(*xyPos(X[i % niter]))
  print(X[i % niter][0], X[i % niter][1], X[i % niter][2], f4(X[i % niter][3]))
  return line,

X = np.zeros((niter,4))
anim = FuncAnimation(plt.gcf(), animate, init_func = start, interval = 200, blit=True)
plt.show()