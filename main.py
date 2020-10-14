import numpy as np
from matplotlib import pyplot as plt

N = 20  # Кол-во делений пространства
Per = 100  # Кол-ва делений времени

a = 5

dx = 2 * a / N
dy = 2 * a / N
dt = 0.1

f = np.zeros(N * N * 2 * 11 * 11).reshape(2, N, N, 11, 11)

for x in range(N):
    for y in range(N):
        for v_x in range(11):
            for v_y in range(11):
                f[0][x][y][v_x][v_y] = np.exp(-((dx * x - a)**2 +
                                      (dy * y - a)**2)/ 2)  # Начальные условия



x = np.linspace(2*dx, dx * (N-2), N)
y = np.linspace(2*dy, dy * (N-2), N)

X, Y = np.meshgrid(x, y)


concentration = np.zeros(N * N ).reshape( N, N)

for time in range(0, Per - 1, 2):
    for x in range(0, N - 1):  # Making step using only x
        for y in range(0, N - 1):
            for v_x in range(11):
                for v_y in range(11):
                    a = abs((v_x - 5) * dt / dx)
                    f[1][x][y][v_x][v_y] = max(f[0][x][y][v_x][v_y] + a * (f[0][x+np.sign(v_x-5)][y][v_x][v_y] -
                                   f[0][x][y][v_x][v_y]), 0)
                    concentration[x][y] += f[1][x][y][v_x][v_y]
            #concentration[x][y] = f[1][x][y][2][2]


    fig = plt.figure() # creating an image
    ax = plt.axes(xlim=(2*dx, dx * (N - 2)), ylim=(2*dy, dy * (N - 2)), zlim=(0, 100), projection='3d')
    ax.plot_surface(X, Y, concentration, cmap='Blues')
    fig.savefig('gas' + '{0:d}'.format(time + 1000) + '.jpg')
    plt.close('all')

    concentration = np.zeros_like(concentration)

    for x in range(0, N - 1):  # making step using only y
        for y in range(0, N - 1):
            for v_x in range(11):
                for v_y in range(11):
                    a = abs((v_y - 5)*dt/dy)
                    f[0][x][y][v_x][v_y] = max(f[1][x][y][v_x][v_y] + a * (f[1][x][y+np.sign(v_y - 5)][v_x][v_y] -
                                   f[1][x][y][v_x][v_y]), 0)
                    concentration[x][y] += f[0][x][y][v_x][v_y]
            #concentration[x][y] = f[0][x][y][2][2]


    fig = plt.figure() # creating an image
    ax = plt.axes(xlim=(2*dx, dx * (N - 2)), ylim=(2*dy, dy * (N - 2)), zlim=(0, 100), projection='3d')
    ax.plot_surface(X, Y, concentration, cmap='Blues')
    fig.savefig('gas' + '{0:d}'.format(time + 1001) + '.jpg')
    plt.close('all')
    concentration = np.zeros_like(concentration)