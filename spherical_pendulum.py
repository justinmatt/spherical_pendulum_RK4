"""The following program numerically solves the equation of motion of a spherical pendulum using Runge-Kutta method,
the final result simulate the path traced by bob of the spherical pendulum"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
 
Length = 1 #length of pendulum
g = 9.8 #acceleration due to gravity
def F(t,y):
    dtheta,dphi,theta,phi = y[0],y[1],y[2],y[3]
    
    d2theta = np.sin(theta)*np.cos(theta)*dphi**2 - g/Length * np.sin(theta)   #second order ODE for theta coordinate
    d2phi = -2.0*dphi*dtheta/np.tan(theta)          #second order ode for phi coordinate
    
    return np.array([d2theta,d2phi,dtheta,dphi])

def Rk4(t,y,dt):
    k1 = F(t, y)
    k2 = F(t+0.5*dt, y+k1*dt*0.5)
    k3 = F(t+0.5*dt, y+k2*dt*0.5)
    k4 = F(t+dt, y+k3*dt)
    
    return dt*(k1+2*k2+2*k3+k4)/6

dt = 0.01     #step size for numerical calculation
t0,tf = 0,50  #initial time and final time
time = np.arange(t0,tf,dt)
y0 = np.array([0,1,1,0])    #[initial velocity (dtheta,dphi),initial positions(theta,phi)]
th = []
ph = []
for t in time:
    y0 = y0 + Rk4(t,y0,dt)
    
    th.append(y0[2])
    ph.append(y0[3])

# coordinates transformed to cartesian     
x = Length*np.sin(th)*np.cos(ph)
y = Length*np.sin(th)*np.sin(ph)
z = -Length*np.cos(th)

#plt.plot(x,y) #uncomment this for path in X-Y plane


def func(num, CoordXYZ, line, ani_speed=1):     #animation speed is adjusted in ani_speed

    line.set_data(CoordXYZ[0:2, :num*ani_speed])    
    line.set_3d_properties(CoordXYZ[2, :num*ani_speed])
    
    return line
 
#Animation part
Coords = np.array([x, y, z])
Coords_size = len(time)
 
fig = plt.figure()
ax = Axes3D(fig)


line = plt.plot(Coords[0], Coords[1], Coords[2],color='blue', lw=2)[0] #line plot

 
ax.set_xlim3d([-Length, Length])
ax.set_title('Trajectory of Spherical pendulum')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.text2D(0.05, 0.95, "g = %s $m/s^2$"%(g), transform=ax.transAxes)
ax.text2D(0.05, 0.90, "Length = %s $m$"%(Length), transform=ax.transAxes)
 
# Creating the Animation object
path_animation = animation.FuncAnimation(fig, func, frames=Coords_size, fargs=(Coords,line), interval=10, blit=False)
 
 
plt.show()
