from datetime import datetime
from matplotlib import pyplot as plt    
from matplotlib.animation import FuncAnimation

import numpy

rx, ry, ax, ay = 10, 10, 50, 20
eps = 0.01

n_epoch = 150
n_x, n_y, n_vx, n_vy = 100, 100, 7, 7
dt, dx, dy, dv = 1, 1, 1, 0.2

data = numpy.zeros((n_x, n_y, n_vx, n_vy))
prev_data = data.copy()


fig = plt.figure()
axes = plt.axes(projection='3d', zlim=(0, 2))
space_x = numpy.linspace(0, dx * n_x, n_x)
space_y = numpy.linspace(0, dy * n_y, n_y)
X, Y = numpy.meshgrid(space_x, space_y)


def draw(surface):
    axes.clear()
    axes.set_zlim((0, 2))
    return axes.plot_surface(X, Y, surface, cmap='inferno')


def initial_f(i_x, i_y, i_vx, i_vy):
    if i_vx == 3 and i_vy == 6:
        return numpy.exp(- (i_x - ax) ** 2 / 2.0 / rx ** 2 - (i_y - ay) ** 2 / 2.0 / ry ** 2)
    return eps


def initialize():
    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            for i_x in range(n_x):
                for i_y in range(n_y):
                    prev_data[i_x, i_y, i_vx, i_vy] = initial_f(i_x, i_y, i_vx, i_vy)


def get_v(i_vx, i_vy):
    vx = (i_vx - (n_vx - 1) / 2.0) * dv
    vy = (i_vy - (n_vy - 1) / 2.0) * dv
    return vx, vy


S_x, S_y = 0, 0  # Я не придумал хороших названий, но это константы в знаментале, при высичлении диффузного отражения

for i_vx in range(n_vx//2):
    for i_vy in range(n_vy):
        v = get_v(i_vx, i_vy)
        S_x += numpy.exp(-0.5 * (v[0]**2 + v[1]**2))

for i_vx in range(n_vx):
    for i_vy in range(n_vy//2):
        v = get_v(i_vx, i_vy)
        S_y += numpy.exp(-0.5 * (v[0]**2 + v[1]**2))

assert S_x == S_y

def is_border(i_x, i_y, i_vx, i_vy):
    v = get_v(i_vx, i_vy)
    is_right = i_x + 1 == n_x and v[0] > 0
    is_left = i_x == 0 and v[0] < 0
    is_bottom = i_y + 1 == n_y and v[1] > 0
    is_top = i_y == 0 and v[1] < 0
    return any([is_right, is_left, is_bottom, is_top])


# Обычная формула
def formula_rough(prev_data, i_vx, i_vy, direction):
    v = get_v(i_vx, i_vy)
    v_projection = numpy.dot(v, direction)
    this_prev = prev_data[:, :, i_vx, i_vy]
    if v_projection > 0:
        right_prev = numpy.roll(this_prev, - numpy.array(direction), axis=(0, 1))  # Те, которые i+1
        diff = (right_prev - this_prev)
    else:
        left_prev = numpy.roll(this_prev, direction, axis=(0, 1))  # Те, которые i-1
        diff = (this_prev - left_prev)
    return this_prev + v_projection * dt / dx * diff


# Формула более точная
def formula_precise(prev_data, i_vx, i_vy, direction):
    v = get_v(i_vx, i_vy)
    gamma = numpy.dot(v, direction) * dt / dx

    this_prev = prev_data[:, :, i_vx, i_vy]
    left_prev = numpy.roll(this_prev, - numpy.array(direction), axis=(0, 1))  # Те, которые i-1
    right_prev = numpy.roll(this_prev, numpy.array(direction), axis=(0, 1))  # Те, которые i+1
    value = gamma * (1 + gamma) * left_prev / 2 \
            + (1 - gamma * gamma) * this_prev \
            - gamma * (1 - gamma) * right_prev / 2
    return value


def calculate_concentration():
    return numpy.add.reduce(data[:, :, :, :], (2, 3))


def save_to_file(filename, array):
    lines = []
    for i_x in range(n_x):
        for i_y in range(n_y):
            lines.append(f"{i_x}\t{i_y}\t{array[i_x, i_y]}\n")

    with open(filename, "w") as f:
        f.writelines(lines)


def calc_epoch(i):
    global data, prev_data
    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            v = get_v(i_vx, i_vy)
            data[1:-1, :, i_vx, i_vy] = formula_precise(prev_data, i_vx, i_vy, (1, 0))[1:-1, :]
            if v[0] < 0:
                # Скорость направлена вправо
                data[-1:, :, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (1, 0))[-1:, :]
            else:
                # Скорость направлена влево
                data[:1, :, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (-1, 0))[:1, :]
   
    # Зеркальное отражение по X
    #for i_y in range(n_y):
    #    for i_vx in range(n_vx):
    #        for i_vy in range(n_vy):
    #            v = get_v(i_vx, i_vy)
    #            if v[0] < 0:
    #                data[0, i_y, i_vx, i_vy] = prev_data[0, i_y, n_vx - i_vx - 1, i_vy]
    #            else:
    #                data[n_x - 1, i_y, i_vx, i_vy] = prev_data[n_x - 1, i_y, n_vx - i_vx - 1, i_vy]
    
    S_pos_x = numpy.add.reduce(prev_data[0, :, n_vx//2+1:, :], (1, 2))
    S_neg_x = numpy.add.reduce(prev_data[n_x - 1, :, :n_vx//2, :], (1, 2)) 
  
    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            v = get_v(i_vx, i_vy)
            if v[0] < 0:
                data[0, :, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_pos_x/S_x
            elif v[0] > 0:
                data[n_x - 1, :, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_neg_x/S_x
                

    data, prev_data = prev_data, data

    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            v = get_v(i_vx, i_vy)
            data[:, 1:-1, i_vx, i_vy] = formula_precise(prev_data, i_vx, i_vy, (0, 1))[:, 1:-1]
            if v[1] < 0:
                data[:, :-1, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (0, 1))[:, :-1]
            else:
                data[:, :1, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (0, 1))[:, :1]
   
    # Зеркальное отражение по Y
    #for i_x in range(n_x):
    #    for i_vx in range(n_vx):
    #        for i_vy in range(n_vy):
    #            v = get_v(i_vx, i_vy)
    #            if v[1] < 0:
    #                data[i_x, 0, i_vx, i_vy] = prev_data[i_x, 0, i_vx, n_vy - i_vy - 1]
    #            else:
    #                data[i_x, n_y - 1, i_vx, i_vy] = prev_data[i_x, n_y - 1, i_vx, n_vy - i_vy - 1]
    
    S_pos_y = numpy.add.reduce(prev_data[:, 0, :, n_vy//2+1:], (1, 2)) 
    S_neg_y = numpy.add.reduce(prev_data[:, n_y - 1, :, :n_vy//2], (1, 2))
    
    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            v = get_v(i_vx, i_vy)
            if v[1] < 0:
                data[:, 0, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_pos_y/S_y
            elif v[1] > 0:
                data[:, n_y - 1, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_neg_y/S_y
   

    print (numpy.add.reduce(data[:, 0, :, :3], (0, 1, 2)) / numpy.add.reduce(prev_data[:, 0, :, 4:], (0, 1, 2)))
    print ('\n')


    data, prev_data = prev_data, data

    concentration = calculate_concentration()
    total_count = numpy.add.reduce(concentration, (0, 1))
    #if i % 10 == 0:
    time = datetime.now().time()
    print(f"{time}. Step {i}. Total count: {total_count}")
    #save_to_file(f"out_{i:03}.dat", concentration)
    #return total_count 
    return draw(concentration)


def main():
    initialize()
    total_count_array = []
    #for i in range(n_epoch):
    #    total_count = calc_epoch(i)
    #    total_count_array.append(total_count)
    #value = numpy.std(total_count_array)
    #print("Standard deviation: ", value)


main()
_animation = FuncAnimation(fig, calc_epoch, repeat=False, frames=n_epoch)
#plt.show()

_animation.save('diffusion.gif', writer='imagemagic', fps=15)
