# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 07:51:28 2015

@author: leo
"""
from __future__ import division

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from numba import jit

#Símbolo / se torna divisão verdadeira (e não apenas a inteira)


#Rossler system ##############################################  
##############################################################

#
#Explore system with N initial conditions
#

def explore_rossler(N=10, angle=0.0, max_time=4.0, 
    a=0.398, b=2.0, c=4.0, 
    Plot=False):
    """
    You can use it with animated widgets:
    
        from IPython.html.widgets import interact, interactive
        w = interactive(explore_rossler, angle=(0.,360.), N=(0,50), sigma=(0.0,50.0), rho=(0.0,50.0),Plot=True)
        display(w)
        
    Or use it alone or with another code:
    
        t, x_t = explore_rossler(angle=0, N=10, max_time=40, Plot=False)
        leo.GPU_plot_3D(x_t) 
        
        Obs.: to remove initial transiente use e.g. x_t[:,1000:,:].
    """
        
    def rossler_deriv(x_y_z, t0, a=a, b=b, c=c):
        """Compute the time-derivative of the cord attractor
        (Lorenz'84 modified)."""
        x, y, z = x_y_z
        return [-y-z, x+a*y, b+z*(x-c)]

    # Choose random starting points, uniformly distributed from -15 to 15
    np.random.seed(1)
    x0 = -2 + 1 * np.random.random((N, 3))

    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250*max_time))
    x_t = np.asarray([integrate.odeint(rossler_deriv, x0i, t)
                      for x0i in x0])
    
    #
    #Figure
    #
    if Plot == True:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.axis('off')

        # prepare the axes limits
        #ax.set_xlim((-25, 25))
        #ax.set_ylim((-35, 35))
        #ax.set_zlim((5, 55))

        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, N))

        for i in range(N):
            x, y, z = x_t[i,:,:].T
            lines = ax.plot(x, y, z, '-', c=colors[i])
            plt.setp(lines, linewidth=2)

        ax.view_init(30, angle)
        plt.show()

    return t, x_t
    
    
#
#Uncoupled  
@jit(nopython=True)
def rossler_system(Y, t, w,a,b,c):
        """Compute the time-derivative of the Rossler system."""
        x, y, z=Y
        return [-w*y -z, w*x + a*y, b + z*(x-c)]
    
    
def solve_rossler(w=1, a=0.3, b=2, c=4, max_time=400,
                  dt=0.1, s_time=0.2, xyz0=[2,0,0], Plot=False):
    """Solve the Rossler system with parameters a, b, c, natural
    frequency w and initinal condition xyz0. 
    
    Call as: t, xyz = solve_rossler(**args_dict)
    
       args_dict={'a':0.4,'b':2.,
           'c':4.,
           'max_time':4000,
           'dt':0.01,
           's_time':0.01}
    
    Returns a 1D (time) and 2D array with 3 columns (x, y and z time series).
    Returns the Phase Space plot if Plot=True.
    
       w, DOUBLE = natural frequency of the oscillator
       a,b, c, INT = parameters.
       max_time, INT = lenght of the simulation, in Rossler time units
       dt, DOUBLE = integration step
       s_time > dt, DOUBLE = sampling time
       xyz0, DOUBLE = initial conditions array [x0, y0, z0]
       Plot, True/False: if True, returns the phase space plot.
    """
    from scipy import integrate
    
    

    #Condicao inicial
    xyz0 = np.array(xyz0)

    #Solucao das trajetorias
    t = np.linspace(0, max_time, int((1/dt)*max_time))
    xyz_t = integrate.odeint(rossler_system, xyz0, t,args=(w,a,b,c))
    
    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    xyz_t = xyz_t[::int(s_time/dt), :]
    
    #figura
    if Plot == True:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        #axes labels
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Rossler Attractor")
        ax.plot(xs=xyz_t[:,0], ys=xyz_t[:,1], zs=xyz_t[:,2])
        ax.scatter3D(xs=xyz0[0], ys=xyz0[1], zs=xyz0[2], color='r')
    
    
        plt.show()
    
    return t, xyz_t



#
#Coupled
#
@jit(nopython=True)
def Network_RosslerSystem(Y,t, F,w, A, b, c, Nsys):
    """Defines a set of coupled Rossler Systems in a Network structure.
        Coupling is done through the y variable.
        The coupling and the network topology is defined by the matrix A.
        The sistem is (where (i) * is multiplication element wise
           and (ii) AY is the diag(kroneker product(A*Y))):
           
        X_dot = -w*Y - Z
        Y_dot = X +AY
        Z_dot = b_f + Z*(X-c)
        
        Y, 1D array: state of the systems of the form
            [x1,x2, ...,xn, y1,y2, ...,yn, z1,z2,...,zn].
        t is needed to the function ODEinte.
        w, 1D array of floats: frequencies of the oscilators.
        A, 2D array: matrix defining the coupling (explicity 
            Y_dot = X +AY)
        b, c, FLOATs: see the ODE equations
        Nsys, INT: number of systems, must be iqual to int(len(Y0)/3)
            because the each system has 3 dimentions (x, y, z)"""
    # com Jit: 
    #    15 segundos -> 3.4 segundos com Sys.shape: (1000000, 6)

    #X_dot = -w*Y - Z
    F[:Nsys] = -w*Y[Nsys:2*Nsys] - Y[2*Nsys:3*Nsys]
    
    #Y_dot = X +AY: ayj +c(y_{j+1} -2yj +2y_{j-1}) == y_{j+1} -(a-2c)yj +2y_{j-1}
    F[Nsys:2*Nsys] = w*Y[:Nsys] + np.dot(Y[Nsys:2*Nsys],A) #y_{j+1} -(a-2c)yj +2y_{j-1}
    
    #Z_dot = b_f + Z*(X-c)
    F[2*Nsys:3*Nsys] = b + Y[2*Nsys:3*Nsys]*(Y[:Nsys]-c)
    
    return F
    #return np.r_[dotX, dotY, dotZ]                               34.9 micro_s.
    #return np.array([dotX, dotY, dotZ]).ravel()                  16.3 micro_s.
    #return [dotX, dotY, dotZ]                                    12.6 micro_s.
    #@jit                                                         19.8 micro_s.
    #@jit(float64[:](float64[:],float64... limite serah 3x 3 micro_s =  9 micro_s 
    #   => nao compensa usar jit.
    # nao atribuindo dotX etc, tudo direto no return              16.1 micro_s
    #trocar dotX, dotY etc por dotY[:Nys], dotY[Nsys:2*Nsys]      15.2 micro_s*
    
    
    


def Network_solve_RosslerSystem(args):
    """Returns a 2D-array of the trajectories of the coupled systems in the form
    [X1...n, Y1...n, Z1...n]
    
        Y0, 1D array of DOUBLE or INT: inital state of the form 
            Y0 = [x1,x2, ...,xn, y1,y2, ...,yn, z1,z2,...,zn]
        dt, DOUBLE = integration step
        s_time > dt, DOUBLE = sampling time"""
#    w, A, b=0.1, c=8.5,
 #                               Y0 = [2,2, 0,-2, 0,0],
  #                              max_time=400, dt=0.1, s_time=0.2
    from scipy import integrate
    
    w, A, b, c, Y0, max_time, dt, s_time = args 
    Nsys = int(len(Y0)/3) #number of coupled Rossler systems
     
    F=np.empty_like(Y0)
    t = np.linspace(0, max_time, int((1/dt)*max_time))

    Yt = integrate.odeint(Network_RosslerSystem, Y0, t,
                                 args=(F,w, A, b, c, Nsys))

    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    #return Nsys, t, Yt[::int(s_time/dt), :]
    return Yt[::int(s_time/dt), :]
    
    
    
    
    
# Network Rossler, choosing the kind of coupling ################
@jit(nopython=True)
def Network_RosslerSystemChooseCoup(Y,t,F, 
                w, a, b, c,Agm,
                Nsys,coup_vector=[0.0,1.0,0.0],coup_strength=0.0):
    x=Y[:Nsys]
    y=Y[Nsys:2*Nsys]
    z=Y[2*Nsys:3*Nsys]
    
    beta_x,beta_y,beta_z=coup_vector
    
    F[:Nsys] = -w*y - z +beta_x*coup_strength*np.dot(Agm,x)
    F[Nsys:2*Nsys] = w*x + a*y +beta_y*coup_strength*np.dot(Agm,y)
    F[2*Nsys:3*Nsys] = b + z*(x-c) +beta_z*coup_strength*np.dot(Agm,z)
    
    return F


def Network_solve_RosslerSystemChooseCoup(args):
    from scipy import integrate
    #w, A, b, c, Y0, max_time, dt, s_time = args 
    w, a, b, c, Agm,coup_vector,coup_strength,Y0, max_time, dt, s_time=args
    Nsys = int(len(Y0)/3) #number of coupled Rossler systems
     
    F=np.empty_like(Y0)
    t = np.linspace(0, max_time, int((1/dt)*max_time))

    Yt = integrate.odeint(Network_RosslerSystemChooseCoup, Y0, t,
                                 args=(F,w, a, b, c, Agm,Nsys,coup_vector,coup_strength))

    return Yt[::int(s_time/dt), :]
    

    
    
    
    
    
    
    
####################################
#Antigos do Rossler
####################################
def solve_coupled_rossler(w1=0.98, w2=1.02, a=0.3, d=0.05, b_f=0.1, c=8.5,
                          max_time=400, dt=0.1, s_time=0.2,
                          xyz0=[2,10,10, 2,-5,0], Plot=False):
    """w1, w2, DOUBLE = natural frequency of the oscillators
       d, DOUBLE = is the coupling strength -> 3 routes do chaotic phase syncronization:
           1) strong coherence
           2) intermediate coherence
           3) strong noncoherence
        a, DOUBLE = parameter in [0.15; 0.3] in ref [1], pg. 69.
        f, DOUBLE = ???
        dt, DOUBLE = integration step
        s_time > dt, DOUBLE = sampling time"""
    
    
    def coupled_rossler_system((x1, y1, z1, x2, y2, z2), t0, w1=w1, w2=w2, a=a, d=d, b_f=b_f, c=c):
        """Compute the time-derivative of the Rossler system.
            ref [1], pg. 69, eq. 4.25"""
        return [-w1*y1 -z1, w1*x1 + a*y1 + d*(y2-y1), b_f + z1*(x1-c),
               -w2*y2 -z2, w2*x2 + a*y2 + d*(y1-y2), b_f + z2*(x2-c)]
    
    #Condicao inicial (dos dois sistemas)
    xyz0 = np.array(xyz0)

    #Solucao das trajetorias
    t = np.linspace(0, max_time, int((1/dt)*max_time))
    xyz_1_2_t = integrate.odeint(coupled_rossler_system, xyz0, t)
    
    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    xyz_1_2_t = xyz_1_2_t[::int(s_time/dt), :]
    
    #figura
    if Plot == True:
        Color = ('blue', 'green')
        fig = plt.figure(figsize=(16,8))
        
        for i in [0,1]:
            ax = fig.add_subplot(1,2,1+i, projection='3d')
    
            #Trajetoria
            ax.plot(xs=xyz_1_2_t[:,0+3*i], ys=xyz_1_2_t[:,1+3*i], zs=xyz_1_2_t[:,2+3*i], alpha=0.5, color=Color[i])
    
            #Condicao inicial
            ax.scatter3D(xs=xyz0[3*i], ys=xyz0[1+3*i], zs=xyz0[2+3*i], color='r')
    
            #axes labels
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            ax.set_title("Rossler Attractor \n System "+str(i+1)+
                         ", (a,d)=("+str(a)+", "+str(d)+")"+
                         "\n "+str(xyz_1_2_t.shape[0])+
                         " points, Sample frequency "+
                         str(1/s_time))
    
    
        plt.show()
    
    return t, xyz_1_2_t
    
    
    
def CriticalCouplingCurvesRossler(a=0.25, d=0.05):
    
    image_file = "../references/Critical coupling curves of the Rossler systems.PNG"
    image = plt.imread(image_file)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.imshow(image)
    ax.axis('off') # clear x- and y-axes

    px1, px2 = 165, 715
    py1, py2 = 55, 280

    p_sincronismo =[3666.667*a - 385, -1125*d + 280]

    ax.plot([px1, px2, px2, px1], [py1, py1, py2, py2], 'r+')

    ax.plot(p_sincronismo[0], p_sincronismo[1], 'ro')
    ax.set_title('(a, d) = ('+str(a)+', '+str(d)+')')
    
    fig.tight_layout()
    





























# Lorenz system ##############################################
##############################################################

#
#Explore system with N initial conditions
#

def explore_lorenz(N=10, angle=0.0, max_time=4.0, 
   sigma=10.0, beta=8./3, rho=28.0, Plot=False):
    
    """
    You can use it with animated widgets:
    
        from IPython.html.widgets import interact, interactive
        w = interactive(explore_lorenz, angle=(0.,360.), N=(0,50), sigma=(0.0,50.0), rho=(0.0,50.0),Plot=True)
        display(w)
        
    Or use it alone or with another code:
    
        t, x_t = explore_lorenz(angle=0, N=10, max_time=40, Plot=False)
        leo.GPU_plot_3D(x_t) 
        
        Obs.: to remove initial transiente use e.g. x_t[:,1000:,:].
    """

    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        """Compute the time-derivative of a Lorenz system."""
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    # Choose random starting points, uniformly distributed from -15 to 15
    np.random.seed(1)
    x0 = -15 + 30 * np.random.random((N, 3))

    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250*max_time))
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0i, t)
                      for x0i in x0])
    
    #
    #Figure
    #
    if Plot == True:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.axis('off')

        # prepare the axes limits
        ax.set_xlim((-25, 25))
        ax.set_ylim((-35, 35))
        ax.set_zlim((5, 55))

        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, N))

        for i in range(N):
            x, y, z = x_t[i,:,:].T
            lines = ax.plot(x, y, z, '-', c=colors[i])
            plt.setp(lines, linewidth=2)

        ax.view_init(30, angle)
        plt.show()

    return t, x_t
    
    
#
#Uncoupled  
def solve_lorenz(sigma=10.0, beta=8./3, rho=28.0, max_time=400,
                  dt=0.1, s_time=0.2, xyz0=[2,0,0], Plot=False):
    """Solve the Rossler system with parameters a, b, c, natural
    frequency w and initinal condition xyz0. 
    
    Call as: t, xyz = solve_rossler(*args)
    
    Returns a 1D (time) and 2D array with 3 columns (x, y and z time series).
    Returns the Phase Space plot if Plot=True.
    
       w, DOUBLE = natural frequency of the oscillator
       a,b, c, INT = parameters.
       max_time, INT = lenght of the simulation, in Rossler time units
       dt, DOUBLE = integration step
       s_time > dt, DOUBLE = sampling time
       xyz0, DOUBLE = initial conditions array [x0, y0, z0]
       Plot, True/False: if True, returns the phase space plot.
    """
    
    def lorenz_system((x, y, z), t0,sigma=sigma, beta=beta, rho=rho):
        """Compute the time-derivative of the Rossler system."""
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    #Condicao inicial
    xyz0 = np.array(xyz0)

    #Solucao das trajetorias
    t = np.linspace(0, max_time, int((1/dt)*max_time))
    xyz_t = integrate.odeint(lorenz_system, xyz0, t)
    
    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    xyz_t = xyz_t[::int(s_time/dt), :]
    
    #figura
    if Plot == True:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        #axes labels
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("lorenz Attractor")
        ax.plot(xs=xyz_t[:,0], ys=xyz_t[:,1], zs=xyz_t[:,2])
        ax.scatter3D(xs=xyz0[0], ys=xyz0[1], zs=xyz0[2], color='r')
    
    
        plt.show()
    
    return t, xyz_t


#
#Coupled
#

def Network_LorenzSystem(Y,t, A, sigma,beta,rho, Nsys):
    """
    Compute the time-derivative of a Lorenz system.
    Coupling is done through the z variable, as in Bialonsk (2006). Note that
        in his paper the author exchanges x<->y. So, there the coupling is
        through x and here is through z!!!
    """
    
    #X_dot =  sigma*(y - x)
    dotX = sigma*(Y[Nsys:2*Nsys]-Y[:Nsys]) #+ np.dot(Y[:Nsys],A)
    
    #Y_dot = x*(rho-z) - y
    dotY = Y[:Nsys]*(rho-Y[2*Nsys:3*Nsys])-Y[Nsys:2*Nsys]
    
    #Z_dot = x*y - beta*z
    dotZ = Y[:Nsys]*Y[Nsys:2*Nsys] -beta*Y[2*Nsys:3*Nsys] + np.dot(Y[2*Nsys:3*Nsys],A)
    
    return np.array([dotX, dotY, dotZ]).ravel()




    
def Network_solve_LorenzSystem(args):
    """
        Y0, 1D array of DOUBLE or INT: inital state of the form 
            Y0 = [x1,x2, ...,xn, y1,y2, ...,yn, z1,z2,...,zn]
        dt, DOUBLE = integration step
        s_time > dt, DOUBLE = sampling time
        
        Obs.: The number of systems is defined by the length of Y0/3
        
    """
    A, sigma, beta, rho, Y0, max_time, dt, s_time = args
    Nsys = int(len(Y0)/3) #number of coupled Rossler systems
     

    t = np.linspace(0, max_time, int((1/dt)*max_time))

    Yt = integrate.odeint(Network_LorenzSystem, Y0, t,
                                 args=(A, sigma,beta,rho, Nsys))

    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    return Yt[::int(s_time/dt), :]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Cord attractor (modified Lorenz'84 system ##################  
##############################################################

#Segundo e-mail do Aguirre, b eh a freq. de oscilacao

#
#Explore system with N initial conditions
#

def explore_CordAttractor(N=10, angle=0.0, max_time=4.0, 
    a=0.25, b=4.0, F=8.0, G=1.0, 
    Plot=False):
    """
    You can use it with animated widgets:
    
        from IPython.html.widgets import interact, interactive
        w = interactive(solve_lorenz, angle=(0.,360.), N=(0,50), sigma=(0.0,50.0), rho=(0.0,50.0),Plot=True)
        display(w)
        
    Or use it alone or with another code:
    
        t, x_t = solve_CordAttractor(angle=0, N=10, max_time=40, Plot=False)
        leo.GPU_plot_3D(x_t) 
        
        Obs.: to remove initial transiente use e.g. x_t[:,1000:,:].
    """
    
    def CordAttractor_deriv(x_y_z, t0, a=a, b=b, F=F, G=G):
        """Compute the time-derivative of the cord attractor (Lorenz'84 modified)."""
        x, y, z = x_y_z
        return [-y -z -a*x + a*F, x*y -b*x*z -y +G, b*x*y +x*z -z]

    # Choose random starting points, uniformly distributed from -15 to 15
    np.random.seed(1)
    x0 = -15 + 30 * np.random.random((N, 3))

    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250*max_time))
    x_t = np.asarray([integrate.odeint(CordAttractor_deriv, x0i, t)
                      for x0i in x0])
    
    #
    #Figure
    #
    if Plot == True:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.axis('off')

        # prepare the axes limits
        #ax.set_xlim((-25, 25))
        #ax.set_ylim((-35, 35))
        #ax.set_zlim((5, 55))

        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, N))

        for i in range(N):
            x, y, z = x_t[i,:,:].T
            lines = ax.plot(x, y, z, '-', c=colors[i])
            plt.setp(lines, linewidth=2)

        ax.view_init(30, angle)
        plt.show()

    return t, x_t
    
    
#
#Uncoupled  
def solve_CordAttractor(w=1, a=0.25, b=4.0, F=8.0, G=1.0, max_time=400,
                  dt=0.1, s_time=0.2, xyz0=[2,0,0], Plot=False):
    """Solve the CordAttractor system with parameters a, b, c, natural
    frequency w and initinal condition xyz0. 
    
    Call as: t, xyz = solve_rossler(*args)
    
    Returns a 1D (time) and 2D array with 3 columns (x, y and z time series).
    Returns the Phase Space plot if Plot=True.
    
       w, DOUBLE = natural frequency of the oscillator
       a,b, c, INT = parameters.
       max_time, INT = lenght of the simulation, in Rossler time units
       dt, DOUBLE = integration step
       s_time > dt, DOUBLE = sampling time
       xyz0, DOUBLE = initial conditions array [x0, y0, z0]
       Plot, True/False: if True, returns the phase space plot.
    """
    
    def CordAttractor((x, y, z), t0, a=a, b=b, F=F, G=G):
        """Compute the time-derivative of the cord attractor."""
        return [-y -z -a*x + a*F, x*y -b*x*z -y +G, b*x*y +x*z -z]

    #Condicao inicial
    xyz0 = np.array(xyz0)

    #Solucao das trajetorias
    t = np.linspace(0, max_time, int((1/dt)*max_time))
    xyz_t = integrate.odeint(CordAttractor, xyz0, t)
    
    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    xyz_t = xyz_t[::int(s_time/dt), :]
    
    #figura
    if Plot == True:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        #axes labels
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Cord Attractor")
        ax.plot(xs=xyz_t[:,0], ys=xyz_t[:,1], zs=xyz_t[:,2])
        ax.scatter3D(xs=xyz0[0], ys=xyz0[1], zs=xyz0[2], color='r')
    
    
        plt.show()
    
    return t, xyz_t


#
#Coupled
#

def Network_CordAttractor(Y,t, A, a,b,F,G, Nsys):
    """
    Coupling is done through the x variable
    """
    
    #X_dot =  -y-z-ax+aF
    dotX = -Y[Nsys:2*Nsys] -Y[2*Nsys:3*Nsys] -a*Y[:Nsys] +a*F +np.dot(Y[:Nsys],A)
    
    #Y_dot = xy-bxz-y+G
    dotY = Y[:Nsys]*Y[Nsys:2*Nsys] -b*Y[:Nsys]*Y[2*Nsys:3*Nsys] -Y[Nsys:2*Nsys] +G
    
    #Z_dot = bxy+xz-z
    dotZ = b*Y[:Nsys]*Y[Nsys:2*Nsys] +Y[:Nsys]*Y[2*Nsys:3*Nsys] -Y[2*Nsys:3*Nsys]
    
    return np.array([dotX, dotY, dotZ]).ravel()
    
    


def Network_solve_CordAttractor(args):
    """
        Y0, 1D array of DOUBLE or INT: inital state of the form 
            Y0 = [x1,x2, ...,xn, y1,y2, ...,yn, z1,z2,...,zn]
        dt, DOUBLE = integration step
        s_time > dt, DOUBLE = sampling time
        
        Obs.: The number of systems is defined by the length of Y0/3
        
    """
    from scipy import integrate

    A, a, b, F, G, Y0, max_time, dt, s_time = args
    Nsys = int(len(Y0)/3) #number of coupled Rossler systems
     

    t = np.linspace(0, max_time, int((1/dt)*max_time))

    Yt = integrate.odeint(Network_CordAttractor, Y0, t,
                                 args=(A, a,b,F,G, Nsys))

    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    return Yt[::int(s_time/dt), :]
    
   
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#Van der Pol oscillator ##############################################  
##############################################################

#
#Explore system with N initial conditions
#

def explore_VanderPol(N=10, max_time=4.0, w=1, mi=7, Plot=False):
    """
    You can use it with animated widgets:
    
        from IPython.html.widgets import interact, interactive
        w = interactive(explore_VanderPol, N=(0,50),w=(0.1,10.0), mi=(0.1,10),Plot=True)
        display(w)
        
    Or use it alone or with another code:
    
        t, x_t = explore_VanderPol(N=10, max_time=4.0, w=1, mi=7, Plot=False)
        leo.GPU_plot_3D(x_t) 
        
        Obs.: to remove initial transiente use e.g. x_t[:,1000:,:].
    """
        
    def VanderPol_deriv(x_y, t0, w=w, mi=mi):
        """Compute the time-derivative of a Van der Pol system."""
        x, y = x_y
        return [y, -(w**2)*x + mi*(1-x**2)*y]

    # Choose random starting points, uniformly distributed from -15 to 15
    np.random.seed(1)
    x0 = -0.5 + 1.2 * np.random.random((N, 2))

    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250*max_time))
    x_t = np.asarray([integrate.odeint(VanderPol_deriv, x0i, t)
                      for x0i in x0])
    
    #
    #Figure
    #
    if Plot == True:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        #ax.axis('off')

        # prepare the axes limits
        #ax.set_xlim((-4, 4))
        #ax.set_ylim((-4, 4))
        
        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, N))

        for i in range(N):
            x, y = x_t[i,:,:].T
            lines = ax.plot(x, y, '-', c=colors[i])
            plt.setp(lines, linewidth=2)

        plt.show()

    return t, x_t
    
    
#
#Uncoupled  
def solve_VanderPol(w=1, mi=0.1, max_time=400,
                  dt=0.1, s_time=0.2, xy0=[0.1,0.1], Plot=False):
    """Solve the Rossler system with parameters a, b, c, natural
    frequency w and initinal condition xyz0. 
    
    Call as: t, xyz = solve_rossler(*args)
    
    Returns a 1D (time) and 2D array with 3 columns (x, y and z time series).
    Returns the Phase Space plot if Plot=True.
    
       w, DOUBLE = natural frequency of the oscillator
       a,b, c, INT = parameters.
       max_time, INT = lenght of the simulation, in Rossler time units
       dt, DOUBLE = integration step
       s_time > dt, DOUBLE = sampling time
       xyz0, DOUBLE = initial conditions array [x0, y0, z0]
       Plot, True/False: if True, returns the phase space plot.
    """
    
    def VanderPol_deriv((x,y), t0, w=w, mi=mi):
        """Compute the time-derivative of a Van der Pol system."""
        return [y, -(w**2)*x + mi*(1-x**2)*y]

    #Condicao inicial
    xy0 = np.array(xy0)

    #Solucao das trajetorias
    t = np.linspace(0, max_time, int((1/dt)*max_time))
    xy_t = integrate.odeint(VanderPol_deriv, xy0, t)
    
    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    xy_t = xy_t[::int(s_time/dt), :]
    
    #figura
    if Plot == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        #axes labels
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title("Van der Pol oscillator")
        ax.plot(xy_t[:,0], xy_t[:,1])
           
    
        plt.show()
    
    return t, xy_t



#
#Coupled
#

def Network_VanderPol(Y,t, w, A, b, c, Nsys):
    """Defines a set of coupled Rossler Systems in a Network structure.
        The coupling and the network topology is defined by the matrix A.
        The sistem is (where (i) * is multiplication element wise
           and (ii) AY is the diag(kroneker product(A*Y))):
           
        X_dot = -w*Y - Z
        Y_dot = X +AY
        Z_dot = b_f + Z*(X-c)
        
        Y, 1D array: state of the systems of the form
            [x1,x2, ...,xn, y1,y2, ...,yn, z1,z2,...,zn].
        t is needed to the function ODEinte.
        w, 1D array of floats: frequencies of the oscilators.
        A, 2D array: matrix defining the coupling (explicity 
            Y_dot = X +AY)
        b, c, FLOATs: see the ODE equations
        Nsys, INT: number of systems, must be iqual to int(len(Y0)/3)
            because the each system has 3 dimentions (x, y, z)"""
    
    #X_dot = -w*Y - Z
    dotX = -w*Y[Nsys:2*Nsys] - Y[2*Nsys:3*Nsys]
    
    #Y_dot = X +AY: ayj +c(y_{j+1} -2yj +2y_{j-1}) == y_{j+1} -(a-2c)yj +2y_{j-1}
    dotY = w*Y[:Nsys] + np.dot(Y[Nsys:2*Nsys],A) #y_{j+1} -(a-2c)yj +2y_{j-1}
    
    #Z_dot = b_f + Z*(X-c)
    dotZ = b + Y[2*Nsys:3*Nsys]*(Y[:Nsys]-c)
    
    return np.array([dotX, dotY, dotZ]).ravel()
    #return np.r_[dotX, dotY, dotZ]                               34.9 micro_s.
    #return np.array([dotX, dotY, dotZ]).ravel()                  16.3 micro_s.
    #return [dotX, dotY, dotZ]                                    12.6 micro_s.
    #@jit                                                         19.8 micro_s.
    #@jit(float64[:](float64[:],float64... limite serah 3x 3 micro_s =  9 micro_s 
    #   => nao compensa usar jit.
    # nao atribuindo dotX etc, tudo direto no return              16.1 micro_s
    #trocar dotX, dotY etc por dotY[:Nys], dotY[Nsys:2*Nsys]      15.2 micro_s*
    
    
    


def Network_solve_VanderPol(w, A, b=0.1, c=8.5,
                                Y0 = [2,2, 0,-2, 0,0],
                                max_time=400, dt=0.1, s_time=0.2):
    """Returns a 2D-array of the trajectories of the coupled systems in the form
    [X1...n, Y1...n, Z1...n]
    
        Y0, 1D array of DOUBLE or INT: inital state of the form 
            Y0 = [x1,x2, ...,xn, y1,y2, ...,yn, z1,z2,...,zn]
        dt, DOUBLE = integration step
        s_time > dt, DOUBLE = sampling time"""
    
    Nsys = int(len(Y0)/3) #number of coupled Rossler systems
     

    t = np.linspace(0, max_time, int((1/dt)*max_time))

    Yt = integrate.odeint(Network_RosslerSystem, Y0, t,
                                 args=(w, A, b, c, Nsys))

    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    return Nsys, t, Yt[::int(s_time/dt), :]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# HR neuron model
########################################################################
def HRneuron_model_single(Y,t0,
            a=2.8, alpha=1.6, c=5, b=9, mu=0.001):
    
    """
    Criado por Luis A. Aguirre, adaptado para Python por Leonardo L. Portes.
    
    xyz_dot=HRneuron([x,y,z],t)
    Simulates the HR neuron model published in 

    A MODEL OF NEURONAL BURSTING USING 3 COUPLED 1ST ORDER 
        DIFFERENTIAL-EQUATIONS
    By: HINDMARSH, JL; ROSE, RM
    PROCEEDINGS OF THE ROYAL SOCIETY SERIES B-BIOLOGICAL SCIENCES
        Volume: 221   Issue: 1222   Pages: 87-102   Published: 1984

    x is the membrane potential
    y associated to fast current 
    z associated to slow current

    parameters: a=2.8; alpha=1.6; c=5; b=9; mu=0.001; as in

    Synchronization of bursting neurons: What matters in the network topology
    By: Belykh, I; de Lange, E; Hasler, M
    PHYSICAL REVIEW LETTERS   Volume: 94   Issue: 18     
        Article Number: 188101   Published: MAY 13 2005
    """
    x = Y[0]
    y = Y[1]
    z = Y[2]
    
    # Differential equations
    xd=a*(x**2)-x**3-y-z
    yd=(a+alpha)*x**2-y
    zd=mu*(b*x+c-z)
    
    return [xd,yd,zd]




def HRneuron_model_single_solve(args):
    """
    args = (a,alpha,c,b,mu,
        max_time,dt,s_time,
        xyz_0)
    xyz_t = solve_HRneuron(args)
    
    Generates the time series xyz_t for the HRneuron model.
    
    Args as in Belykh et al.:
    args = (2.8,1.6,5,9,0.001,
       80000,0.1,0.2,
       [2,0,0])
    
    """
    
    # Parameters
    a,alpha,c,b,mu,max_time,dt,s_time,Y0 = args

    t = np.linspace(0, max_time, int((1/dt)*max_time))

    xyz_t = integrate.odeint(HRneuron_model_single, Y0, t,
                                 args=(a,alpha,c,b,mu))

    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    return xyz_t[::int(s_time/dt), :]
    
    
    
#
# Coupled HR Neurons
#
    
def HRneuron_model_Network(Y,t,C,gs,
        a=2.8, alpha=1.6, c=5, b=9, mu=0.001,
        Theta_s=-0.25, Vs=2,lambda_hr=10,
        Nsys=2):
    """
    Coupling is done through the x variable
    gs: synaptic coupling strength
    Vs: reversal potential (if Vs>xi(t) for all xi and all t => excitatory synapse)
    C: counectivity matrix (principal diagonal should be 0)
    """

    x = Y[:Nsys] # x component of each neuron
    y = Y[Nsys:2*Nsys] # y component of each neuron
    z = Y[2*Nsys:3*Nsys] # z component of each neuron
        
    # synaptic coupling function (fast threshold modulation)
    Gamma = 1./(1.+np.exp(-lambda_hr*(x-Theta_s)))
    
    # Differential equations
    xd=a*(x**2)-x**3-y-z - (gs*(x-Vs))*np.dot(C,Gamma)
    yd=(a+alpha)*x**2-y
    zd=mu*(b*x+c-z)

    return np.array([xd, yd, zd]).ravel()




def HRneuron_model_Network_solve(args):
    """
    """
    
    # Parameters
    C,gs,  a,alpha,c,b,mu,max_time,dt,s_time,Y0 = args
    Nsys = int(len(Y0)/3) #number of coupled neurons
    Theta_s, Vs, lambda_hr = -0.25, 2, 10
        

    t = np.linspace(0, max_time, int((1/dt)*max_time))

    xyz_t = integrate.odeint(HRneuron_model_Network, Y0, t,
                                 args=(C,gs,  a,alpha,c,b,mu,
                                       Theta_s, Vs,lambda_hr,
                                       Nsys))
   
        

    #sampling (decimating):
    #All rows: jumping s_time/dt, All columns
    return xyz_t[::int(s_time/dt), :]
    
    
    





#########################################################################
# Lyapunovo spectrum ####################################################
#########################################################################

# based on Sandri, M. (1996). Numerical Calculation of Lyapunov Exponents.
# The Mathematica Journal. 
# Retrieved from http://library.wolfram.com/infocenter/Articles/2902/

from scipy.linalg import norm
from sympy import *
import pandas as pd

def RKStep(F,Y0,dt):
    """Runge-Kutta integration
    F is the vector field """
    
    
    k1F=dt*F(Y0)[0] # ravel because the sympy-numpy function returns [[x1],[y1],[z1]]
    
    k2F=dt*F(Y0+k1F/2)[0]
    
    k3F=dt*F(Y0+k2F/2)[0]

    k4F=dt*F(Y0+k3F)[0]

    return Y0+(k1F+2*k2F+2*k3F+k4F)/6.


def gram_schmidt_columns(u):
    """ Returns an orthogonal (and not normalized) base, using
        the Gram-Schmidt method.
        
        Vectors are considered as the columns of the matrix u.
        """
    
    w=np.copy(u)
    v=np.empty_like(w)

    # w1
    w[:,0]=u[:,0]
    v[:,0]=w[:,0]/norm(w[:,0])
    N_vectors=u.shape[1] # number of vectors=number of columns of the matrix u.

    # wk, for k>0
    for k in range(1,N_vectors):
        for i in range(k):
            w[:,k]-=np.dot(u[:,k],v[:,i])*v[:,i]
        v[:,k]=w[:,k]/norm(w[:,k])

    return w     


def LCEs(F,Vars,Nsys,
        Y0_n,dt_n,T,K,
        Transient=0):

    # State vector (symbolic): None x1 y1 z1 x2 y2 z2 x3 y3 z3
    sys_i=[str(i) for i in np.arange(1,Nsys+1)]

    labels_temp=[j+i+' ' for i in sys_i for j in Vars]
    labels_temp='None '+''.join(labels_temp) # none para que X[1] corresponda ao simbolo x1

    # state variables
    X=symbols(labels_temp, real=True)
    vecX=Matrix(X[1:])
    
    # Jacobian
    DF=F.jacobian(vecX)
    
    # Matrix Phi
    Phi=Matrix(np.array(['phi'+str(i)+str(j) 
        for i in range(1,Nsys*3+1)
        for j in range(1,Nsys*3+1)]).reshape(Nsys*3,Nsys*3))
    
    # dPhi/dt
    DPhi=Phi*DF.T


    # Symbolic vector [x ..., y..., z..., phi11, phi12, ...
    vecY0_Phi0=flatten(vecX)+flatten(Phi)

    # Creating the functions (joining F and DPhi)
    f_dphi=lambdify([vecY0_Phi0],(DPhi.T.vec().row_insert(0,F)).T,'numpy') # flattened function [F, DF.Phi]
    
    
    # Simulation ##########################
    
    # Numeric vectors and matrices
    Phi0_n=np.eye(Nsys*3)
    vecY0_Phi0_n=np.hstack((Y0_n,Phi0_n.ravel()))
    yt=vecY0_Phi0_n.copy()
    
    # Transient
    if Transient>0:
        N_n=int(Transient/dt_n)
        

        for i in xrange(N_n-1):
            yt=RKStep(f_dphi,yt,dt_n)
        
    # Permanent regime
    N=int(T/dt_n)
    
    # If the transient was evaluated, we need to return the final
    #   part of the array, in respect to Phi, to the identity matrix (Phi0_n)
    yt=np.hstack((yt[:-Phi0_n.ravel().shape[0]],Phi0_n.ravel()))
    
    w_norms_list=np.zeros((K,Nsys*3))

    for k in xrange(K):
        for i in xrange(N):
            yt=RKStep(f_dphi,yt,dt_n)

        Phi_t=yt[Nsys*3:].reshape(Nsys*3,Nsys*3).T

        # Gram-Schmidt orthogonalization
        #print Phi_t
        
        W=gram_schmidt_columns(Phi_t)
        norms_W=norm(W,axis=0)

        w_norms_list[k,:]=norms_W

        # Normalizing Phi_t
        Phi_t=W/norms_W
        yt[Nsys*3:]=Phi_t.T.ravel()
    
    integration_time=np.arange(1,K+1)*T
    cols_labels=['L'+str(i_lyap)+'_sys'+str(i_sys) 
                 for i_sys in range(1,Nsys+1) 
                 for i_lyap in range(1,len(Vars)+1)]
    lces=pd.DataFrame(np.log(w_norms_list).cumsum(axis=0)/integration_time[:,None],columns=cols_labels)

    
    
    return lces
