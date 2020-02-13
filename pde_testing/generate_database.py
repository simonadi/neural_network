# @date 2019/10/18
# @author adi_simon, angla_celestine, chassat_perrine, goyeau_larry, laurendeau_matthieu, lehmann_fanny
# @brief create a database on the solutions of simple pde equations

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
np.set_printoptions(threshold=sys.maxsize)

############ DIFFUSION ############

def diffusion_1D(dx, dt, c, T, u_0):
    # @return solve PDE of a diffusion equation in 1D
    # @param: dx space step
    #         dt time step
    #         c constant advsection velocity
    #         T final time
    #         u_0 initial condition

    Nt = int(T/dt)
    Nx = int(1/dx)
    u = np.zeros((Nt+1, Nx+1))
    X = np.linspace(0, 1, Nx+1)

    # Initial time conditions
    u[0, :] = np.array([u_0(x) for x in X])

    coef = c * (dt/dx**2)
    # Euler in time, explicit second order in space
    for t in range(0, Nt):
        u[t+1, 1:-1] = coef*u[t,:-2] + coef*u[t,2:] + (1-2*coef)*u[t,1:-1]
        # Boundary conditions
        u[t+1,0] = 0
        u[t+1,-1] = 0

    u_final = u[-1,]

    return X, u_final


def diffusion_2D(dx, dy, dt, c, T, u_0):
    # @return solve PDE of a diffusion equation in 2D
    # @param: dx,dy spaces step
    #         dt time step
    #         c constant advsection velocity
    #         T final time
    #         u_0 initial condition

    def k(i,j, n):
        return i+(n+1)*j

    Nt = int(T/dt)
    Nx = int(1/dx)
    Ny = int(1/dy)
    N = (Nx+1)**2
    u = np.zeros((Nt+1, N))

    X = np.linspace(0, 1, Nx+1)

#    #Initialisation
#     for x in range(0,Nx):
#         for y in range(0,Ny):
#             u[0,x,y] = u_0(X[x],X[y])

#     # Propagate with forward-difference in time, central-difference in space
#     for t in range(1,Nt):
#         u[t, 1:-1, 1:-1] = u[t-1, 1:-1, 1:-1] + c * dt * (
#           (u[t-1,2:, 1:-1] - 2*u[t-1,1:-1, 1:-1] + u[t-1,:-2, 1:-1])/dx**2
#           + (u[t-1,1:-1, 2:] - 2*u[t-1,1:-1, 1:-1] + u[t-1,1:-1, :-2])/dy**2 )

    # IMPLICIT SCHEME: SLOW BUT WORKING
    # u_final = np.zeros((Nt+1, Nx+1, Nx+1))
    # u[0, :] = np.array([u_0(x,y) for x in X for y in X])
    # # Build the matrix A defined by A * U_n+1 = Un
    # # A is intialized as the identity matrix NxN
    # A = np.eye(N)
    # # loop over points that are not on the border
    # for x in range(1,Nx):
    #     for y in range(1,Nx):
    #         A[k(x,y, Nx),k(x,y, Nx)] = 1 + 4*c

    #         A[k(x,y, Nx),k(x,y+1, Nx)] = -c
    #         A[k(x,y, Nx),k(x,y-1, Nx)] = -c

    #         A[k(x,y, Nx),k(x-1,y, Nx)] = -c
    #         A[k(x,y, Nx),k(x+1,y, Nx)] = -c

    # for t in range(0, Nt):
    #     u[t+1,:] = linalg.solve(A, u[t,:])
    #     u_final[t] = u[t,:].reshape(Nx+1,Nx+1)

    # return X, u_final

    # EXPLICIT SCHEME: WORKING BUT NOT UNCONTIONALLY STABLE
    Nt = int(T/dt)
    Nx = int(1/dx)
    Ny = int(1/dy)
    u = np.zeros((Nt+1, Nx+1, Ny+1))
    X = np.linspace(0, 1, Nx+1)

    #Initialisation time conditions
    for x in range(0,Nx):
        for y in range(0,Ny):
            u[0,x,y] = u_0(X[x],X[y])

    # Propagate with forward-difference in time, central-difference in space
    for t in range(1,Nt):
        u[t, 1:-1, 1:-1] = u[t-1, 1:-1, 1:-1] + c * dt * (
          (u[t-1, 2:, 1:-1] - 2*u[t-1, 1:-1, 1:-1] + u[t-1, :-2, 1:-1])/(dx**2)
          + (u[t-1, 1:-1, 2:] - 2*u[t-1, 1:-1, 1:-1] + u[t-1, 1:-1, :-2])/(dy**2) )
        # Boundary conditions
        u[t,0,0] = 0
        u[t,0,-1] = 0
        u[t,-1,0] = 0
        u[t,-1,-1] = 0

    return X, u








############ WAVE ############

def wave_1D(dx, dt, c, T, u_0, v_0=0):
    # @return solve PDE of a wave equation in 1D
    # @param: dx space step
    #         dt time step
    #         c constant advsection velocity
    #         T final time
    #         u_0 initial condition
    #         v_0=0 initial condition on the derivate
    #         scheme_order order of the scheme

    Nt = int(T/dt)
    Nx = int(1/dx)
    u = np.zeros((Nt+1, Nx+1))
    X = np.linspace(0, 1, Nx+1)

    # Initial time conditions
    u[0, :] = np.array([u_0(x) for x in X])

    coef = c * (dt/dx**2)
    # Euler forward scheme
    for x in range(Nx):
        u[1,x] = u[0,x]+v_0(x*dx)*dt

    for t in range(2,Nt):
        u[t, 1:-1] = -u[t-2,1:-1] + 2*u[t-1,1:-1] + \
            (coef**2)*(u[t-1,:-2]-2*u[t-1,1:-1]+u[t-1,2:])

        # Boundary conditions (Neumann)
        u[t+1,0] = -u[t-1,0] + 2*u[t,0] + 2*(coef**2)*(u[t,1]-u[t,0])
        u[t+1,-1] = -u[t-1,-1] + 2*u[t,-1] + 2*(coef**2)*(u[t,-2]-u[t,-1])

    u_final = u[-2,]

    return X, u_final

def wave_2D(dx, dy, dt, c, T, u_0, v_0=0):
    Nt = int(T/dt)
    Nx = int(1/dx)
    Ny = int(1/dy)
    u = np.zeros((Nt+1, Nx+1, Ny+1))
    v = np.zeros((Nx+1, Ny+1))
    X = np.linspace(0, 1, Nx+1)

    ax = c*dt/dx
    ay = c*dt/dy

   #Initialisation
    for x in range(0,Nx):
        for y in range(0,Ny):
            u[0,x,y] = u_0(X[x],X[y])
            v[x,y] = v_0(X[x],X[y])

    u[1,1:-1,1:-1] = u[0,1:-1,1:-1] + v[1:-1,1:-1]*dt \
        + (1/2) * ax**2 * (u[0,2:,1:-1] - 2*u[0,1:-1,1:-1] + u[0,:-2,1:-1]) \
        + (1/2) * ay**2 * (u[0,1:-1,2:] - 2*u[0,1:-1,1:-1] + u[0,1:-1,:-2])

    for t in range(2, Nt):
        u[t,1:-1,1:-1] = 2*u[t-1,1:-1,1:-1] - u[t-2,1:-1,1:-1]\
            +  ax**2 * (u[t-1,2:,1:-1] - 2*u[t-1,1:-1,1:-1] + u[t-1,:-2,1:-1]) \
            +  ay**2 * (u[t-1,1:-1,2:] - 2*u[t-1,1:-1,1:-1] + u[t-1,1:-1,:-2])

        # TO ADD Boundary conditions (Neumann) (check the coefficient)
        u[t+1,0,0] = -u[t-1,0,0] + 2*u[t,0,0] + 2*(ax**2*ay**2)*(u[t,1,1]-u[t,0,0])
        u[t+1,0,-1] = -u[t-1,0,-1] + 2*u[t,0,-1] + 2*(ax**2*ay**2)*(u[t,1,-2]-u[t,0,-1])
        u[t+1,-1,0] = -u[t-1,-1,0] + 2*u[t,-1,0] + 2*(ax**2*ay**2)*(u[t,-2,1]-u[t,-1,0])
        u[t+1,-1,-1] = -u[t-1,-1,-1] + 2*u[t,-1,-1] + 2*(ax**2*ay**2)*(u[t,-2,-2]-u[t,-1,-1])

    return X, u

############ ANIMATION ############

def animate_1D(nature, X, u, Nt):
    # @return animation of a $nature equation evolution in 1 dimension
    # @param: nature 'diffusion' or 'wave' equation
    #         X space discretization vector
    #         u matrix of the evolution modelising the solution
    #         Nt number of frames

    if (nature=='diffusion'):
        print('Animate diffusion equation in 1D')
    elif(nature=='wave'):
        print('Animate wave equation in 1D')
    else:
        print('Error nature in animate_1D. Only possible  \
            diffusion or wave / '+str(nature)+'given.')


    fig = plt.figure()
    ax = plt.axes(xlim=(0,1), ylim=(-1,1))
    label = nature+' equation'
    fig.suptitle(label, fontsize=15)
    ax.set_xlabel('Space grid', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    line, = ax.plot([], [])

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(X, u[i,:])
        ax.set_title(u"T={}".format(round(i/Nt,3)))
        return line,

    # call the animator
    anim = animation.FuncAnimation(fig, animate, init_func=init,
        frames=Nt+1, interval=50, blit=False)

    plt.show()

def animate_2D(nature, X, u, Nt, Nx):
    # @return animation of a $nature equation evolution in 2 dimensions
    # @param: nature 'diffusion' or 'wave' equation
    #         x,y space meshgrid
    #         u matrix of the evolution modelising the solution
    #         Nt number of frames
    #         Nx space discretization

    if (nature=='diffusion'):
        print('Animate diffusion equation in 2D')
    elif(nature=='wave'):
        print('Animate wave equation in 2D')
    else:
        print('Error nature in animate_2D. Only possible  \
            diffusion or wave / '+str(nature)+'given.')

    fig = plt.figure()
    label = nature+' equation'
    fig.suptitle(label, fontsize=15)
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect('equal')
    x,y = np.meshgrid(X,X)

    # animation function
    def animate(i):
        ax.set_title(u"T={}".format(round(i/Nt,3)))
        z = u[i]
        cont = plt.contourf(x, y, z, 25, vmin=0, vmax=1)
        return cont

    # call the animator
    anim = animation.FuncAnimation(fig, animate, frames=Nx)
    plt.show()


if __name__ == "__main__":
    print('Test diffusion_1D:', end=' ')

    # Numerical parameters
    c = 1/(np.pi**2)
    dt = 0.01
    dx = np.sqrt(2*dt*c) * 2
    T = 1

    # Initial condition
    u_0 = lambda x : np.sin(np.pi*x)



    # Resolution
    X, u_final = diffusion_1D(dx, dt, c, T, u_0)

    u_init = [u_0(x) for x in X]

    plt.plot(X, u_init)
    plt.plot(X, u_final)

    plt.show()


    print('Test wave_1D:', end=' ')

    # Numerical parameters
    c = 0.01
    dt = 0.001
    dx = np.sqrt(c*dt)
    T = 1

    # Initial condition
    u_0 = lambda x : np.exp(-100*(x-0.5)**2) # gaussian
    # u_0 = lambda x : 0.5 - abs(x-0.5) # non smooth initial function
    v_0 = lambda x : 0


    # Resolution
    X, u_final = wave_1D(dx, dt, c, T, u_0, v_0)
    u_init = [u_0(x) for x in X]

    print(u_final)

    plt.plot(X, u_init)
    plt.plot(X, u_final, label='final')

    plt.legend()
    plt.show()

    # Animate
    Nt = int(T/dt)
    # animate_1D('wave', X, u, Nt)




    # print('Test diffusion_2D:', end=' ')
    #
    # # Numerical parameters
    # dx = 0.04
    # dy = dx
    # dt = 0.003
    # T = 0.5
    # c = 0.3
    #
    #
    # # Initial condition
    # # u_0 = lambda x,y : np.exp(-100*(x-0.5)**2) * np.exp(-100*(y-0.5)**2) # gaussian
    # u_0 = lambda x,y : np.sin(np.pi*x)*np.sin(np.pi*y)
    #
    # # Resolution
    # X, u = diffusion_2D(dx, dy, dt, c, T, u_0)
    #
    # # Animate
    # Nt = int(T/dt)
    # Nx = int(1/dx)
    # # animate_2D('diffusion', X, u, Nt, Nx)
    #
    #
    #
    #
    # print('Test wave_2D:', end=' ')
    #
    # # Numerical parameters
    # c = 1
    # dx = 0.01
    # dy = dx
    # dt = 0.005
    # # dt = (1/float(c))*(1/np.sqrt(1/dx**2 + 1/dy**2))
    # T = 1
    #
    # # Initial condition
    # # u_0 = lambda x,y : np.sin(np.pi*x)*np.sin(np.pi*y)
    # u_0 = lambda x,y : np.exp(-100*(x-0.5)**2) * np.exp(-100*(y-0.5)**2) # gaussian
    # v_0 = lambda x,y : 0
    #
    # # Resolution
    # X, u = wave_2D(dx, dy, dt, c, T, u_0,v_0)
    #
    # # Animate
    # Nt = int(T/dt)
    # Nx = int(1/dx)
    # # animate_2D('wave', X, u, Nt, Nx)
