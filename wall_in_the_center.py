from datetime import datetime
from matplotlib import pyplot as plt    
from matplotlib.animation import FuncAnimation

import numpy

rx, ry, ax, ay = 3, 3, 5, 15
eps = 0.01

n_epoch = 1200
n_x, n_y, n_vx, n_vy = 50, 50, 15, 15
dt, dx, dy, dv = 2, 2, 2, 0.1  # It was 1, 1, 1, 0.2

data = numpy.zeros((n_x, n_y, n_vx, n_vy))
prev_data = data.copy()

concent = 100

wall_x_phy = 15
wall_y_phy = 50

wall_x = (wall_x_phy * n_x) // 100
wall_y = (wall_y_phy * n_y) // 100

x_data = []
y1_data = []
y2_data = []

fig = plt.figure()
axes = plt.axes(
                projection='3d'
                )
space_x = numpy.linspace(0, dx * n_x, n_x)
space_y = numpy.linspace(0, dy * n_y, n_y)
X, Y = numpy.meshgrid(space_x, space_y)
#X = numpy.arange(61 * dx, dx * n_x, dx)
#Y = numpy.arange(0 * dy, dy * n_y, dy)

levels = numpy.linspace(-0.01, 5, 500)

def draw(surface):
    axes.clear()
    axes.set_zlim((0, 4))
    return axes.plot_surface(X, Y, surface, cmap='inferno',
            #levels=levels
            )


def get_v(i_vx, i_vy):
    vx = (i_vx - (n_vx - 1) / 2.0) * dv
    vy = (i_vy - (n_vy - 1) / 2.0) * dv
    return vx, vy


S_all = 0
for i_vx in range(n_vx):
    for i_vy in range(n_vy):
        v = get_v(i_vx, i_vy)
        S_all += numpy.exp(-0.5 * (v[0]**2 + v[1]**2))


def initial_f(i_x, i_y, i_vx, i_vy):
    #if i_y < wall_y:
    #    return concent*numpy.exp(-0.5 * ((i_vx - n_vx//2)**2 + (i_vy - n_vy//2)**2))/S_all
    #if i_y > wall_y - 1:
    #    return 0.01*concent*numpy.exp(-0.5 * ((i_vx - n_vx//2)**2 + (i_vy - n_vy//2)**2))/S_all
    #return eps
    

    #if i_y < wall_y:
    #    if i_vx == 9 and i_vy == 0 and i_y:
    #        return numpy.exp(- (i_x - ax) ** 2 / 2.0 / rx ** 2 - (i_y - ay) ** 2 / 2.0 / ry ** 2)
    #    return eps * 5
    #return eps

    if i_y < wall_y:
        return concent
    return eps

def initialize():
    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            for i_x in range(n_x):
                for i_y in range(n_y):
                    prev_data[i_x, i_y, i_vx, i_vy] = initial_f(i_x, i_y, i_vx, i_vy)

Vx_plus = numpy.arange(1,  n_vx//2 + 1)*dv
Vx_minu = numpy.arange(-(n_vx//2), 0)*dv
Vy_plus = numpy.arange(1,  n_vy//2 + 1)*dv
Vy_minu = numpy.arange(-(n_vy//2), 0)*dv


S_x, S_y = 0, 0  # Я не придумал хороших названий, но это константы в знаментале, при высичлении диффузного отражения

for i_vx in range(n_vx//2):
    for i_vy in range(n_vy):
        v = get_v(i_vx, i_vy)
        S_x += numpy.exp(-0.5 * (v[0]**2 + v[1]**2)) * v[0]

for i_vx in range(n_vx):
    for i_vy in range(n_vy//2):
        v = get_v(i_vx, i_vy)
        S_y += numpy.exp(-0.5 * (v[0]**2 + v[1]**2)) * v[1]

#assert S_x == S_y

def is_border(i_x, i_y, i_vx, i_vy):
    v = get_v(i_vx, i_vy)
    is_right = all([i_x + 1 == n_x, v[0] > 0, i_y <= wall_y-1]) or all([i_x == wall_x - 1, v[0] > 0, i_y == wall_y])
    is_left = all([i_x == 0, v[0] < 0, i_y <= wall_y])
    is_bottom = all([i_y == wall_y - 1, v[1] > 0, i_x >= wall_x])
    is_outside = all([i_y + 1 == wall_y, v[1] < 0, i_x >= wall_x])
    is_top = i_y == 0 and v[1] < 0
    return any([is_right, is_left, is_bottom, is_top, is_outside])


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


def easy_x_diffusion():
    S_pos_x = numpy.add.reduce(prev_data[0, :, n_vx//2+1:, :], (1, 2))
    S_neg_x = numpy.add.reduce(prev_data[n_x - 1, :, :n_vx//2, :], (1, 2)) 
    S_wall_x = numpy.add.reduce(prev_data[wall_x - 1, wall_y, :n_vx//2, :], (0, 1))
 
    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            v = get_v(i_vx, i_vy)
            if v[0] < 0:
                data[0, :, i_vx, i_vy] = S_pos_x / ((n_vx//2) * n_vy)
            elif v[0] > 0:
                data[n_x - 1, :, i_vx, i_vy] = S_neg_x / ((n_vx//2) * n_vy)
                data[wall_x - 1, wall_y, i_vx, i_vy] = S_wall_x / ((n_vx//2) * n_vy)


def easy_y_diffusion():
    S_pos_y = numpy.add.reduce(prev_data[:, 0, :, n_vy//2+1:], (1, 2)) 
    S_neg_y = numpy.add.reduce(prev_data[:, n_y - 1, :, :n_vy//2], (1, 2))
    S_out_y = numpy.add.reduce(prev_data[wall_x:, wall_y + 1, :, n_vy//2+1:], (1, 2))
    S_wal_y = numpy.add.reduce(prev_data[wall_x:, wall_y - 1, :, :n_vy//2], (1, 2))

    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            v = get_v(i_vx, i_vy)
            if v[1] < 0:
                data[:, 0, i_vx, i_vy] = S_pos_y / (n_vx * (n_vy//2))
                data[wall_x:, wall_y + 1, i_vx, i_vy] = S_out_y / (n_vx * (n_vy//2))
            elif v[1] > 0:
                data[:, n_y - 1, i_vx, i_vy] = S_neg_y / (n_vx * (n_vy//2))
                data[wall_x:, wall_y - 1, i_vx, i_vy] = S_wal_y / (n_vx * (n_vy//2))


def x_reflection():
    for i_y in range(n_y):
        for i_vx in range(n_vx):
            for i_vy in range(n_vy):
                v = get_v(i_vx, i_vy)
                if v[0] < 0:
                    data[0, i_y, i_vx, i_vy] = prev_data[0, i_y, n_vx - i_vx - 1, i_vy]
                else:
                    data[n_x - 1, i_y, i_vx, i_vy] = prev_data[n_x - 1, i_y, n_vx - i_vx - 1, i_vy]
                    data[wall_x - 1, wall_y, i_vx, i_vy] = prev_data[wall_x - 1, wall_y, n_vx - i_vx - 1, i_vy]


def y_reflection():
    for i_x in range(n_x):
        for i_vx in range(n_vx):
            for i_vy in range(n_vy):
                v = get_v(i_vx, i_vy)
                if v[1] < 0:
                    data[i_x, 0, i_vx, i_vy] = prev_data[i_x, 0, i_vx, n_vy - i_vy - 1]
                else:
                    data[i_x, n_y - 1, i_vx, i_vy] = prev_data[i_x, n_y - 1, i_vx, n_vy - i_vy - 1]


def calc_epoch(i):
    global data, prev_data
    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            v = get_v(i_vx, i_vy)
            data[1:-1, :wall_y, i_vx, i_vy] = formula_precise(prev_data, i_vx, i_vy, (1, 0))[1:-1, :wall_y]
            data[1:wall_x - 1, wall_y, i_vx, i_vy] = formula_precise(prev_data, i_vx, i_vy, (1, 0))[1:wall_x - 1, wall_y]
            data[1:-1, wall_y + 1:, i_vx, i_vy] = formula_precise(prev_data, i_vx, i_vy, (1, 0))[1:-1, wall_y + 1:]
            if v[0] < 0:
                # Скорость направлена вправо
                data[-1:, :wall_y, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (1, 0))[-1:, :wall_y]
                data[wall_x - 1, wall_y, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (1, 0))[wall_x - 1, wall_y]
                data[-1:, wall_y + 1:, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (1, 0))[-1:, wall_y + 1:]
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
    
    #near_wall_x = numpy.swapaxes(prev_data, 2, 3)

    #S_pos_x = numpy.add.reduce(numpy.dot(near_wall_x[0, :wall_y + 1, :, n_vx//2+1:], Vx_plus), (1))
    #S_neg_x = numpy.add.reduce(numpy.dot(near_wall_x[n_x - 1, :wall_y, :, :n_vx//2], Vx_minu), (1)) 
    #S_wall_x = numpy.add.reduce(numpy.dot(near_wall_x[wall_x - 1, wall_y, :, :n_vx//2], Vx_minu), (0))
  
    #for i_vx in range(n_vx):
    #    for i_vy in range(n_vy):
    #        v = get_v(i_vx, i_vy)
    #        if v[0] < 0:
    #            data[0, :wall_y + 1, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_pos_x/S_x
    #        elif v[0] > 0:
    #            data[n_x - 1, :wall_y, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_neg_x/S_x
    #            data[wall_x - 1, wall_y, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_wall_x/S_x
     
    easy_x_diffusion()

    data, prev_data = prev_data, data

    for i_vx in range(n_vx):
        for i_vy in range(n_vy):
            v = get_v(i_vx, i_vy)
            data[:, 1:wall_y - 1, i_vx, i_vy] = formula_precise(prev_data, i_vx, i_vy, (0, 1))[:, 1:wall_y - 1]
            data[:wall_x, wall_y - 1:wall_y + 2, i_vx, i_vy] = formula_precise(prev_data, i_vx, i_vy, (0, 1))[:wall_x, wall_y-1:wall_y+2]
            data[:, wall_y + 2:-1, i_vx, i_vy] = formula_precise(prev_data, i_vx, i_vy, (0, 1))[:, wall_y + 2:-1]
            if v[1] < 0:
                data[wall_x:, wall_y - 1, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (0, 1))[wall_x:, wall_y - 1]
                data[:, -1, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (0, 1))[:, -1]
            else:
                data[:, :1, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (0, 1))[:, :1]
                data[wall_x:, wall_y + 1, i_vx, i_vy] = formula_rough(prev_data, i_vx, i_vy, (0, 1))[wall_x:, wall_y + 1]
   
    # Зеркальное отражение по Y
    #for i_x in range(n_x):
    #    for i_vx in range(n_vx):
    #        for i_vy in range(n_vy):
    #            v = get_v(i_vx, i_vy)
    #            if v[1] < 0:
    #                data[i_x, 0, i_vx, i_vy] = prev_data[i_x, 0, i_vx, n_vy - i_vy - 1]
    #            else:
    #                data[i_x, n_y - 1, i_vx, i_vy] = prev_data[i_x, n_y - 1, i_vx, n_vy - i_vy - 1]
    

    #S_pos_y = numpy.add.reduce(numpy.dot(prev_data[:, 0, :, n_vy//2+1:], Vy_plus),            (1)) 
    #S_neg_y = numpy.add.reduce(numpy.dot(prev_data[wall_x:, wall_y - 1, :, :n_vy//2], Vy_minu), (1))
    #S_out_y = numpy.add.reduce(numpy.dot(prev_data[wall_x:, wall_y + 1, :, n_vy//2+1:], Vy_plus), (1))
    
    #for i_vx in range(n_vx):
    #    for i_vy in range(n_vy):
    #        v = get_v(i_vx, i_vy)
    #        if v[1] < 0:
    #            data[wall_x:, wall_y + 1, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_out_y/S_y
    #            data[:, 0, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_pos_y/S_y
    #        elif v[1] > 0:
    #            data[wall_x:, wall_y - 1, i_vx, i_vy] = numpy.exp(-0.5*(v[0]**2 + v[1]**2))*S_neg_y/S_y
   
    easy_y_diffusion()

    #print (numpy.add.reduce(data[:, 0, :, :3], (0, 1, 2)) / numpy.add.reduce(prev_data[:, 0, :, 4:], (0, 1, 2)))
    print ('\n')

    x_data.append(i)
    y1_data.append(numpy.add.reduce(data[:, :wall_y, :, :], (0, 1, 2, 3)))
    y2_data.append(numpy.add.reduce(data[:, wall_y + 1:, :, :], (0, 1, 2, 3)))

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

#_animation.save('outflow_4.gif', writer='imagemagic', fps=15)

#for i in range(n_epoch):
#    calc_epoch(i)

#file1 = open('data1', 'w')
#for num in y1_data:
#    file1.write(str(num) + '\n')
#file1.close()

#file2 = open('data2', 'w')
#for num in y2_data:
#    file2.write(str(num) + '\n')
#file2.close()

#axes.plot(x_data, y1_data)
#axes.plot(x_data, y2_data)

plt.show()
