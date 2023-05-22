
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import norm

class diff_schemes:
    
    def __init__(self,dt,dx,r,x,t,u1,u2,initial_st):
        self.dt = dt
        self.dx = dx
        self.r = r # alpha*tol/h^2
        self.x = x
        self.t = t
        self.u1 = u1
        self.u2 = u2
        self.initial_st = initial_st
        
    def make_figure(self,matx,title_msg):
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        x, y = np.meshgrid(self.x, self.t)
        z = matx
        ax.plot_surface(x, y, z, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title(title_msg)
        plt.show()
        
    def init_condition(self):
        return self.initial_st
    
    def forward_diff_scheme(self):
        matx = np.zeros([len(self.t),len(self.x)])
        matx[0,:] = self.initial_st
        matx[:,0] = 0
        matx[:,-1] = 0
        for i in range(1, len(self.t)):
            for j in range(1,len(self.x)-1):
                dd1 = norm.pdf(self.x[j], np.pi/3, self.dx/(2**0.5))
                dd2 = norm.pdf(self.x[j], 2*np.pi/3, self.dx/(2**0.5))
                matx[i,j] = self.r*(matx[i-1,j-1]-2*matx[i-1,j]+matx[i-1,j+1]) + self.dt*(50*(np.exp(-4/(1+matx[i-1,j]))-np.exp(-4))) - self.dt*2*matx[i-1,j] + self.dt*2*((dd1/2)*self.u1+self.u2*(dd2/2)) + matx[i-1,j]
                if matx[i,j]<0:
                    matx[i,j] = 0
        return matx
