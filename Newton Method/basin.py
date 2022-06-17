
# Import Libraries

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

aW = 1
'''a = 0
b = 2
theta = np.pi/3'''

'''a = 0
b = 2
theta = 0'''#mooi symmetrisch

a = 0
b = 2
theta = 0

def xyPos(x):
  return [[0, - sin(x[0:1]), - sin(x[0:2]) - sin(x[0:1]), -f4(x[3])*sin(x[0:3]) - sin(x[0:2]) - sin(x[0:1])],
         [0,   cos(x[0:1]),   cos(x[0:2]) + cos(x[0:1]),  f4(x[3])*cos(x[0:3]) + cos(x[0:2]) + cos(x[0:1])]]

def appendTikz(fName, x):
  with open(fName, "a") as f:
    pos = xyPos(x)
    f.write("\\begin{tikzpicture}[scale = 0.18]\n")
    f.write('''\\draw[thick, -] (-6,-3)--(6,-3);
\\foreach \\x in {-6,-5.5,...,5.5}
{\\draw[black, -] (\\x,-3) -- (\\x+0.5,-3.5);}\n''')
    x = [3 * p for p in pos[0]]
    y = [3 * p for p in pos[1]]
    f.write("\\draw(0,-3) node {\\textbullet};\n")
    for i in range(4):
      f.write("\\draw({},{}) node {{\\textbullet}};\n".format(x[i], y[i]))
    f.write("\\draw[black, -] (0,-3)--({},{})--({},{})--({},{})--({},{});\n".format(x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3]))
    f.write("\\end{tikzpicture}\n")

def plotPictures(fName, xs, size, labelx, labely):
  with open(fName, "a") as f:
    f.write("\\begin{{tikzpicture}}[scale = {}]\n".format(0.7/float(size)))
    f.write("\\node[] at ({},-13) {{${}$}};\n".format(20 * size - 12, labelx))
    f.write("\\node[] at (-13,{}) {{${}$}};\n".format(20 * size - 12, labely))
    f.write("\\node[] at (0,-13) {0};\n")
    f.write("\\node[] at (-13,0) {0};\n")
    for x in range(1, size):
      if (2 * x == np.gcd(2 * x, size)):
        if(size == np.gcd(2 * x, size)):
          f.write("\\node[] at ({}, -13) {{$\\pi$}};\n".format(x * 20))
        else:
          f.write("\\node[] at ({}, -13) {{$\\frac{{\\pi}}{{{}}}$}};\n".format(x * 20, size // np.gcd(2 * x, size)))
      else:
        if(size == np.gcd(2 * x, size)):
          f.write("\\node[] at ({}, -13) {{${}\\pi$}};\n".format(x * 20, (2 * x) // np.gcd(2 * x, size)))
        else:
          f.write("\\node[] at ({}, -13) {{$\\frac{{{}\\pi}}{{{}}}$}};\n".format(x * 20, (2 * x) // np.gcd(2 * x, size), size // np.gcd(2 * x, size)))
    for y in range(1, size):
      if (2 * y == np.gcd(2 * y, size)):
        if(size == np.gcd(2 * y, size)):
          f.write("\\node[] at (-13, {}) {{$\\pi$}};\n".format(y * 20))
        else:
          f.write("\\node[] at (-13, {}) {{$\\frac{{\\pi}}{{{}}}$}};\n".format(y * 20, size // np.gcd(2 * y, size)))
      else:
        if(size == np.gcd(2 * y, size)):
          f.write("\\node[] at (-13, {}) {{${}\\pi$}};\n".format(y * 20, (2 * y) // np.gcd(2 * y, size)))
        else:
          f.write("\\node[] at (-13, {}) {{$\\frac{{{}\\pi}}{{{}}}$}};\n".format(y * 20, (2 * y) // np.gcd(2 * y, size), size // np.gcd(2 * y, size)))

    f.write("\\draw[black, ->](-11,-10)--({},-10);\n".format(20 * size - 8))
    f.write("\\draw[black, ->](-10,-11)--(-10,{});\n".format(20 * size - 8))
    f.write("\\foreach \\x in {{10,30,...,{}}}{{\\draw[dashed, -](\\x, -10)--(\\x,{});}}\n".format(20 * size - 10, 20 * size - 10))
    f.write("\\foreach \\y in {{10,30,...,{}}}{{\\draw[dashed, -](-10, \\y)--({},\\y);}}\n".format(20 * size - 10, 20 * size - 10))
    for y in range(size):
      for x in range(size):
        f.write('''\\draw[thick, -] ({},{})--({},{});
\\foreach \\x in {{{},{},...,{}}}
{{\\draw[black, -] (\\x,{}) -- (\\x+0.5,{});}}\n'''.format(-6 + 20 * x, -3 + 20 * y, 6 + 20 * x, -3 + 20 * y, -6 + 20 * x, -5.5 + 20 * x, 5.5 + 20 * x, -3 + 20 * y, -3.5 + 20 * y))
        pos = xyPos(xs[x, y])
        xr = [3 * p + 20 * x for p in pos[0]]
        yr = [3 * p + 20 * y for p in pos[1]]
        f.write("\\draw({},{}) node {{\\textbullet}};\n".format(20 * x, 20 * y - 3))
        for i in range(4):
          f.write("\\draw({},{}) node {{\\textbullet}};\n".format(xr[i], yr[i]))
        f.write("\\draw[black, -] ({},{})--({},{})--({},{})--({},{})--({},{});\n".format(20 * x, 20 * y - 3, xr[0],yr[0],xr[1],yr[1],xr[2],yr[2],xr[3],yr[3]))

    f.write("\\end{tikzpicture}\n")



def sin(xs):
  return np.sin(sum(xs))

  
def cos(xs):
  return np.cos(sum(xs))

f4 = lambda x : 1.5 + np.arctan(x)/np.pi
Df4 = lambda x : 1.0 / (np.pi * (1 + x**2))

f  = lambda x : np.array([-f4(x[3])*sin(x[0:3]) - sin(x[0:2]) - sin(x[0:1]) - a, f4(x[3])*cos(x[0:3]) + cos(x[0:2]) + cos(x[0:1]) - b, cos(x[0:3]) - np.cos(theta), sin(x[0:3]) - np.sin(theta)])
Df = lambda x : np.array([[-Df4(x[3])*cos(x[0:3]) - cos(x[0:2]) - cos(x[0:1]), -Df4(x[3])*cos(x[0:3]) - cos(x[0:2]), -Df4(x[3])*cos(x[0:3]), - Df4(x[3]) * sin(x[0:3])],
                          [-Df4(x[3])*sin(x[0:3]) - sin(x[0:2]) - sin(x[0:1]), -Df4(x[3])*sin(x[0:3]) - sin(x[0:2]), -Df4(x[3])*sin(x[0:3]), Df4(x[3]) * cos(x[0:3])],
                          [-sin(x[0:3]), -sin(x[0:3]), -sin(x[0:3]), 0],
                          [cos(x[0:3]), cos(x[0:3]), cos(x[0:3]), 0]])

# Newton
#niter = 490
niter = 50


'''a = -1
b = 1.4
theta = np.pi/2

X = np.array([0, 0, 0, 1.5])
for k in range(niter-1):
  X = X - 0.005 * np.linalg.pinv(Df(X))@f(X)

appendTikz("WISM100_Robotics-master/lectures/out.tex", X)'''


'''a = 3
b = 1
theta = np.pi + np.pi/5'''

'''for k in range(niter):
  X = X - 0.01 * np.linalg.pinv(Df(X))@f(X)
  if ((k + 1) % 10 == 0):
    appendTikz("WISM100_Robotics-master/lectures/out.tex", X)

print("done")'''

reps = 1
singleres = 50
resolution = reps * singleres


x = np.linspace(-reps * np.pi, reps * np.pi, resolution)
y = np.linspace(-reps * np.pi, reps * np.pi, resolution)

x_2d, y_2d = np.meshgrid(x,y)

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')


ax.axes.set_xlim3d(left=-reps * np.pi, right = reps * np.pi)
ax.axes.set_ylim3d(bottom=-reps * np.pi, top = reps * np.pi)
ax.axes.set_zlim3d(bottom=-np.pi, top = np.pi)
# Surface Plot

z = np.zeros(shape=(resolution, resolution))

for iy in range(resolution):
  for ix in range(resolution):
    X = np.array([x_2d[iy, ix], y_2d[iy, ix], 0, 1.5])
    for k in range(niter-1):
      X = X - 1.0 * np.linalg.pinv(Df(X))@f(X)
    z[iy, ix] = np.cos(X[2])

ax.plot_surface(x_2d, y_2d, z, cmap=cm.jet)

# Labels


ax.set_xlabel('theta_1 (initial)')
ax.set_ylabel('theta_2 (initial)')
ax.set_zlabel('cos(theta_3)')


# Display
plt.show()



reps = 1
singleres = 10
resolution = reps * singleres


x = np.linspace(0, 2 * reps * np.pi, resolution + 1)
y = np.linspace(0, 2 * reps * np.pi, resolution + 1)

x_2d, y_2d = np.meshgrid(x,y)

Xs = np.zeros((singleres, singleres, 4))

for iy in range(resolution):
  for ix in range(resolution):
    X = np.array([x_2d[iy, ix], 0 , y_2d[iy, ix], 1.5])
    for k in range(niter-1):
      X = X - 1.0 * np.linalg.pinv(Df(X))@f(X)
    Xs[ix, iy] = X

plotPictures("WISM100_Robotics-master/lectures/out.tex", Xs, resolution, "\\theta_1", "\\theta_3")