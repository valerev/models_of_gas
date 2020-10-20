import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

N = 20  # Кол-во делений пространства
Per = 100  # Кол-ва делений времени

a = 3

dx = 2 * a / N
dy = 2 * a / N
dt = 0.1

number_of_velocities = 11 # In one direction

f = np.zeros(N * N * 2 * number_of_velocities * number_of_velocities)
f = f.reshape(2, N, N, number_of_velocities, number_of_velocities)

space_x = np.linspace(2*dx, dx * (N-2), N)
space_y = np.linspace(2*dy, dy * (N-2), N)

X, Y = np.meshgrid(space_x, space_y)

concentration = np.zeros(N * N ).reshape( N, N)


def fill_space(is_one_velocity = False):
    if is_one_velocity:
        for x in range(N):
            for y in range(N):
                        f[0][x][y][2][2] = np.exp(-((dx * x - a)**2 +
                                              (dy * y - a)**2)/ 2)  # Начальные условия
    else:
        for x in range(N):
            for y in range(N):
                for v_x in range(number_of_velocities):
                    for v_y in range(number_of_velocities):
                        f[0][x][y][v_x][v_y] = np.exp(-((dx * x - a)**2 +
                                              (dy * y - a)**2)/ 2)  # Начальные условия


fig = plt.figure()  # creating an image
ax = plt.axes(xlim=(2 * dx, dx * (N - 2)), ylim=(2 * dy, dy * (N - 2)), zlim=(0, 100), projection='3d')


def draw_3d(c):
    ax.clear()
    return ax.plot_surface(X, Y, c, cmap='Blues')
    #fig.savefig('gas' + '{0:d}'.format(n + 1000) + '.jpg')


def print_plot(n):
    draw_3d()


def make_x_step(t, x, y, v_x, v_y):
    a = abs((v_x - number_of_velocities//2) * dt / dx)
    f[t][x][y][v_x][v_y] = max(f[0][x][y][v_x][v_y] +
                               a * (f[0][x + np.sign(v_x - number_of_velocities//2)][y][v_x][v_y] -
                               f[0][x][y][v_x][v_y]), 0)


def make_y_step(t, x, y, v_x, v_y):
    a = abs((v_y - number_of_velocities//2) * dt / dy)
    f[t][x][y][v_x][v_y] = max(f[1][x][y][v_x][v_y] +
                               a * (f[1][x][y + np.sign(v_y - number_of_velocities//2)][v_x][v_y] -
                               f[1][x][y][v_x][v_y]), 0)


def count_x_collisions_with_walls(time):
    pre_time = (time - 1)%2

    for y in range(0, N):
        for v_x in range(number_of_velocities // 2 + 1, number_of_velocities):
            a = abs((v_x - number_of_velocities // 2) * dt / dx)
            for v_y in range(number_of_velocities):
                f[time][0][y][v_x][v_y] = (1-a)*f[pre_time][0][y][v_x][v_y] + \
                                          a*f[pre_time][0][y][number_of_velocities - 1 - v_x][v_y];

    for y in range(0, N):
        for v_x in range(number_of_velocities // 2):
            a = abs((v_x - number_of_velocities // 2) * dt / dx)
            for v_y in range(number_of_velocities):
                f[time][N-1][y][v_x][v_y] = (1 - a) * f[pre_time][N-1][y][v_x][v_y] + \
                                          a * f[pre_time][N-1][y][number_of_velocities - 1 - v_x][v_y];


def count_y_collisions_with_walls(time):
    pre_time = (time - 1)%2

    for x in range(0, N):
        for v_y in range(number_of_velocities // 2 + 1, number_of_velocities):
            a = abs((v_y - number_of_velocities // 2) * dt / dy)
            for v_x in range(number_of_velocities):
                f[time][x][0][v_x][v_y] = (1 - a) * f[pre_time][x][0][v_x][v_y] + \
                                          a * f[pre_time][x][0][v_x][number_of_velocities - 1 - v_y];

    for x in range(0, N):
        for v_y in range(number_of_velocities // 2):
            a = abs((v_y - number_of_velocities // 2) * dt / dy)
            for v_x in range(number_of_velocities):
                f[time][x][N-1][v_x][v_y] = (1 - a) * f[pre_time][x][N-1][v_x][v_y] + \
                                          a * f[pre_time][x][N-1][v_x][number_of_velocities - 1 - v_y];


fill_space(is_one_velocity=True)


def count_two_steps(n_step):
    concentration = np.zeros(N * N ).reshape( N, N)
    for x in range(0, N - 1):  # Making step using only x
        for y in range(0, N):
            for v_x in range(number_of_velocities):
                for v_y in range(number_of_velocities):
                    make_x_step(1, x, y, v_x, v_y)
                    concentration[x][y] += f[1][x][y][v_x][v_y]
            # concentration[x][y] = f[1][x][y][2][2] # We need this only to check, how does it work with one speed
    count_x_collisions_with_walls(1)

    concentration = np.zeros(N * N).reshape(N, N)

    for x in range(0, N):  # making step using only y
        for y in range(0, N - 1):
            for v_x in range(number_of_velocities):
                for v_y in range(number_of_velocities):
                    make_y_step(0, x, y, v_x, v_y)
                    concentration[x][y] += f[0][x][y][v_x][v_y]
            # concentration[x][y] = f[0][x][y][2][2] # We need this only to check, how does it work with one speed
    count_y_collisions_with_walls(0)

    return draw_3d(concentration)


_animation = FuncAnimation(fig, count_two_steps, interval=5, repeat=True, frames=Per - 1)
plt.show()
