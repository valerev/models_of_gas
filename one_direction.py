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
        f[0][x][2] = np.exp(- (dx*x-a)**2 / 2)

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
    Y = (v_x - v_x_max//2)*dt/dx
    
    if Y>=0 and x==n_x-1:
        f[(time+1)%2][x][v_x] = f[time][x][v_x] * Y   
        return

    if Y<=0 and x==0:
        f[(time+1)%2][x][v_x] = f[time][x][v_x] * Y  
        return

    f_this = f[time][x][v_x]
    f_next = f[time][x +int(np.sign(Y))][v_x]
    f[(time+1)%2][x][v_x] = f_this + Y * (f_next - f_this)


def _make_bad_x_step(time, x, v_x):
    Y = (v_x - v_x_max//2)*dt/dx

    f_this = f[time][x][v_x]
    if Y<0:
        try:
            f_next = f[time][x+1][v_x]
        except:
            f_next = f[time][0][v_x]
        f[(time+1)%2][x][v_x] = f_this + Y * (f_next - f_this)
            
    else:
        try:
            f_prev = f[time][x-1][v_x]
        except:
            f_prev = f[time][n_x-1][v_x] 
        f[(time+1)%2][x][v_x] = f_this + Y * (f_prev - f_this)
            

def make_x_step(time, x, v_x):
    Y = (v_x - v_x_max//2)*dt/dx
    if x==0:
        f_prev = 0
    else:  
        #f_prev = f[time][-1][v_x]
        f_prev = f[time][x-1][v_x]

    if x==n_x-1:
        f_next = 0
    else:  
        #f_next = f[time][0][v_x]
        f_next = f[time][x+1][v_x]

    f[(time+1)%2][x][v_x] = f[time][x][v_x] * (1- Y**2) + f_next * (Y**2 - Y) * 0.5 + f_prev * (Y**2 + Y) * 0.5


def main(step):
    for x in range(n_x):
        make_bad_x_step(step%2, x, 2)
    #print (count_sum(step%2) / first_sum)
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

