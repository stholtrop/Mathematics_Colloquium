import numpy as np
import matplotlib.pyplot as plt

def pixel(loc):
    x = loc[0]
    y = loc[1]
    if x > 2 or x < -2: return (0, 0, 0)
    if (y - 2) > 0 and (y - 2)**2 + x**2 > 4: return (0, 0, 0)
    if (y - 1) < 0 and (y - 1)**2 + x**2 > 4: return (0, 0, 0)
    if (y - 2)**2 + x**2 <= 4 and (y - 1)**2 + x**2 <= 4: return (255, 0, 0)
    if (y - 2)**2 + x**2 > 4 and (y - 1)**2 + x**2 > 4: return (0, 255, 0)
    if (y - 2)**2 + x**2 <= 4 or (y - 1)**2 + x**2 <= 4: return (0, 0, 255)

    return (255, 255, 255)

def plot_ab():
    xleft, xright, ytop, ybottom = -3, 3, 5, -2
    grid = np.array([[(x, y) for x in np.linspace(xleft, xright, 1000)] for y in np.linspace(ytop, ybottom, 1000)])
    image = np.array([[pixel(grid[i,j]) for j in range(len(grid[0]))] for i in range(len(grid))])
    plt.imshow(image, extent=[xleft, xright, ybottom, ytop])
    plt.xlabel('a')
    plt.ylabel('b')
    plt.show()

def transform(t, b):
    return np.sin(t)*b, np.cos(t)*b

def plot_tb():
    tleft, tright, ytop, ybottom = 0, np.pi, 5, 0
    grid = np.array([[(x, y) for x in np.linspace(tleft, tright, 1000)] for y in np.linspace(ytop, ybottom, 1000)])
    image = np.array([[pixel(transform(*grid[i,j])) for j in range(len(grid[0]))] for i in range(len(grid))])
    plt.imshow(image, extent=[tleft, tright, ybottom, ytop])
    plt.xlabel('$\\theta$')
    plt.ylabel('b')
    plt.show()

def pixel2(loc):
    x = loc[0]
    y = loc[1]
    if x > 4 or x < -4 : return (0, 0, 0)
    if x**2 + (y - 1)**2 < 4 and x**2 + (y - 3)**2 < 4: return  (0, 0, 0)
    if y - 3 < 0 and y - 1 > 0 : return (255, 255, 255)
    if x**2 + (y - 1)**2 > 16 and x**2 + (y - 3)**2 > 16: return (0, 0, 0)
    return (255, 255, 255)

def plot_modified():
    tleft, tright, ytop, ybottom = -6, 6, 8, -4
    grid = np.array([[(x, y) for x in np.linspace(tleft, tright, 1000)] for y in np.linspace(ytop, ybottom, 1000)])
    image = np.array([[pixel2(grid[i,j]) for j in range(len(grid[0]))] for i in range(len(grid))])
    plt.imshow(image, extent=[tleft, tright, ybottom, ytop])
    plt.xlabel('a')
    plt.ylabel('b')
    plt.show()

# plot_tb()
# plot_ab()
plot_modified()
