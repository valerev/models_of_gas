import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

n_x = 100
v_x_max = 3 
a = 2
dt = 0.01
dx = 2*a/n_x
X = np.linspace(0, dx*n_x, n_x)

f = np.zeros(n_x*2*v_x_max).reshape(2, n_x, v_x_max)

def fill_space():
    for x in range(n_x):
        f[0][x][0] = np.exp(- (dx*x-a)**2 / 2)
        #f[0][x][2] = 1
fig = plt.figure()
ax = plt.axes(xlim=(0, n_x*dx))

def draw(c):
    ax.clear()
    return ax.plot(X, c)


def count_sum(time):
    sum = 0
    for x in range(n_x):
        for v_x in range(v_x_max):
            sum += f[time][x][v_x]
    return sum


def make_bad_x_step(time, x, v_x):
    Y = abs((v_x - v_x_max//2)*dt/dx)
    
    if Y>=0 and x==n_x-1:
        f[(time+1)%2][x][v_x] = f[time][x][v_x] * (1 - Y)   
        return

    if Y<=0 and x==0:
        f[(time+1)%2][x][v_x] = f[time][x][v_x] * (1 - Y)  
        return

    f_this = f[time][x][v_x]
    f_next = f[time][x + int(np.sign(Y))][v_x]
    f[(time+1)%2][x][v_x] = f_this + Y * (f_next - f_this)


def make_x_step(time, x, v_x):
    Y = (v_x - v_x_max//2)*dt/dx
   
    if x==0 or x==n_x-1:
        make_bad_x_step(time, x, v_x)
        return
    
    f_prev = f[time][x-1][v_x]
    f_next = f[time][x+1][v_x]

    f[(time+1)%2][x][v_x] = max(f[time][x][v_x] * (1- Y*Y) + f_next * (Y*Y - Y) * 0.5 + f_prev * (Y*Y + Y) * 0.5, 0)


def x_reflection(time):
    for v_x in range(v_x_max//2):
        Y = abs((v_x - v_x_max//2)*dt/dx)
        f[(time+1)%2][0][v_x_max - v_x - 1] += f[time][0][v_x] * Y
    for v_x in range(v_x_max//2 + 1, v_x_max):
        f[(time+1)%2][n_x-1][v_x_max - v_x - 1] += f[time][n_x - 1][v_x] * Y


def count_concentration(time):
    c = np.zeros(n_x)
    for x in range(n_x):
        for v_x in range(v_x_max):
            c[x] += f[time][x][v_x]
    return c


def main(step):
    for x in range(n_x):
        for v_x in range(v_x_max):
            make_x_step(step%2, x, v_x)
    #x_reflection(step%2)
    #print (count_sum(step%2) / first_sum)
    #return draw(count_concentration(step%2))
    return draw(f[step%2])


fill_space()
first_sum = count_sum(0)

_animation = FuncAnimation(fig, main, repeat=False, frames=100)

plt.show()
#plt.close()


#for i in range(300):
#    main(i)
#print(count_sum(0) / first_sum)

#_animation.save('gas.gif', writer='imagemagick', fps=15)

