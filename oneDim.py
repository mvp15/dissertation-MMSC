# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:36:57 2016

@author: arminkekic
"""

from types import *
import warnings
import time
import pickle
import os
import peakutils
import copy

# packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from scipy.optimize import curve_fit, minimize
try:
    from matplotlib2tikz import save as tikz_save
except:
    warnings.warn('matplotlib2tikz could not be imported')
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode']=False
#matplotlib.rcParams['backend']='ps'


try:
    from scipy.integrate import ode
    from scipy.signal import find_peaks_cwt
    import scipy.ndimage as ndimage
except:
    warnings.warn('SciPy could not be imported')

## FEniCS and dolfin
#try:
#    from dolfin import *
#except ImportError:
#    warnings.warn('FEniCS could not be imported')


# functions that return the RHS of the equation of motion
def f_inner(u_left, u, u_right, a_left, a_right, m, Delta_left, Delta_right, n_left, n_right, beta, mu, force):
    return (mu*(beta**2.0)/(m))*( n_left*a_left*np.maximum(Delta_left + u_left - u,0.0)**(n_left - 1.0) - n_right*a_right*np.maximum(Delta_right + u - u_right,0.0)**(n_right - 1.0) ) + ((beta**2.0)/m)*force

# vectorize
#v_f_inner = np.vectorize(f_inner, excluded=['Delta', 'n'])

def f(t, u, Fargs):
    # unwrap function arguments
    m = Fargs[0]
    a = Fargs[1]
    Delta = Fargs[2]
    n = Fargs[3]
    beta = Fargs[5]
    mu = Fargs[6]
    force = Fargs[7]
    
    # prepare arguments for main function evaluation
    a_left      = a[:-1]
    a_right     = a[1:]
    Delta_left  = Delta[:-1]
    Delta_right = Delta[1:]
    n_left      = n[:-1]
    n_right     = n[1:]
    u_left      = u[:-2]
    u_right     = u[2:]
    u           = u[1:-1]
    
    return (mu*(beta**2.0)/(m))*( n_left*a_left*np.maximum(Delta_left + u_left - u,0.0)**(n_left - 1.0) - n_right*a_right*np.maximum(Delta_right + u - u_right,0.0)**(n_right - 1.0) ) + ((beta**2.0)/m)*force
#    return f_inner(u_left, u, u_right, a_left, a_right, m, Delta_left, Delta_right, n_left, n_right, beta, mu, force)

# function for the RHS of the e.o.m. in the expanded system
def F(t, z, Fargs):
    N = Fargs[4]
    vec_return = np.zeros(2*N+4)
       
    # u' = v
    vec_return[0] = 0
    vec_return[1:N+1] = z[N+3:2*N+3]
    vec_return[N+1] = 0
    
    # v' = f(u)
    vec_return[(N+3):(2*N+3)] = f(t, z[:N+2], Fargs)
    return vec_return

# jacobian of f
def jac(t, u, Fargs):
    # unwrap function arguments
    m       = Fargs[0]
    a       = Fargs[1]
    Delta   = Fargs[2]
    n       = Fargs[3]
    N       = Fargs[4]
    beta    = Fargs[5]
    mu      = Fargs[6]
    
    # prepare arguments for main function evaluation
    a_left      = a[:-1]
    a_right     = a[1:]
    Delta_left  = Delta[:-1]
    Delta_right = Delta[1:]
    n_left      = n[:-1]
    n_right     = n[1:]
    u_left      = u[:-2]
    u_right     = u[2:]
    u           = u[1:-1]
    
    # set up jacobian and indices for relevant diagonals
    j           = np.zeros((N,N))
    rows, cols  = np.indices((N,N))
    rows_diag   = np.diag(rows, k=0)
    cols_diag   = np.diag(cols, k=0)
    rows_sub    = np.diag(rows, k=-1)
    cols_sub    = np.diag(cols, k=-1)
    rows_sup    = np.diag(rows, k=1)
    cols_sup    = np.diag(cols, k=1)
    
    # calculate entries
    j[rows_diag, cols_diag] = (mu*(beta**2.0)/(m))*(
        - n_left*(n_left-1.)*a_left*np.maximum( Delta_left + u_left - u,0.0)**(n_left - 2.0) 
        - n_right*(n_right-1.)*a_right*np.maximum(Delta_right + u - u_right,0.0)**(n_right - 2.0) )
    
    j[rows_sub, cols_sub] = ((mu*(beta**2.0)/(m))*(
        n_left*(n_left-1.)*a_left*np.maximum(Delta_left + u_left - u,0.0)**(n_left - 2.0) ))[1:]
    
    j[rows_sup, cols_sup] = ((mu*(beta**2.0)/(m))*(
        n_right*(n_right-1.)*a_right*np.maximum(Delta_right + u - u_right,0.0)**(n_right - 2.0) ))[:-1]
    
    return j

# jacobian of F
def Jac(t, z, Fargs):
    # unwrap function arguments
    N       = Fargs[4]
    
    
    J = np.zeros((2*N+4,2*N+4))
    J[0:N+2,N+2:]  = np.eye(N+2)
    J[N+3:-1,1:N+1]    = jac(t, z[:N+2], Fargs)
    
    return J
    
#    # unwrap function arguments
#    m       = Fargs[0]
#    a       = Fargs[1]
#    Delta   = Fargs[2]
#    n       = Fargs[3]
#    N       = Fargs[4]
#    beta    = Fargs[5]
#    mu      = Fargs[6]
#    
#    # prepare arguments for main function evaluation
#    u           = z[:N+2]
#    a_left      = a[:-1]
#    a_right     = a[1:]
#    Delta_left  = Delta[:-1]
#    Delta_right = Delta[1:]
#    n_left      = n[:-1]
#    n_right     = n[1:]
#    u_left      = u[:-2]
#    u_right     = u[2:]
#    u           = u[1:-1]
#    
##    J = bsr_matrix((2*N+4,2*N+4), blocksize=(N+2,N+2), dtype=np.float64)
#    
#    # tridiagonal part
#    rows, cols  = np.indices((N,N))
#    rows_diag   = np.diag(rows, k=0)
#    cols_diag   = np.diag(cols, k=0)
#    rows_sub    = np.diag(rows, k=-1)
#    cols_sub    = np.diag(cols, k=-1)
#    rows_sup    = np.diag(rows, k=1)
#    cols_sup    = np.diag(cols, k=1)
#    
#    # shift to correct position in J matrix
##    print 'rows_diag', rows_diag
#    rows_diag.setflags(write=True)
#    cols_diag.setflags(write=True)
#    rows_sub.setflags(write=True)
#    cols_sub.setflags(write=True)
#    rows_sup.setflags(write=True)
#    cols_sup.setflags(write=True)
#    
#    rows_diag   += (N+3)
#    cols_diag   += 1
#    rows_sub    += (N+3)
#    cols_sub    += 1
#    rows_sup    += (N+3)
#    cols_sup    += 1
#    
#    diag_data   = (mu*(beta**2.0)/(m))*(
#        - n_left*(n_left-1.)*a_left*np.maximum( Delta_left + u_left - u,0.0)**(n_left - 2.0) 
#        - n_right*(n_right-1.)*a_right*np.maximum(Delta_right + u - u_right,0.0)**(n_right - 2.0) )
#    
#    sub_data    = ((mu*(beta**2.0)/(m))*(
#        n_left*(n_left-1.)*a_left*np.maximum(Delta_left + u_left - u,0.0)**(n_left - 2.0) ))[1:] 
#    
#    sup_data    = ((mu*(beta**2.0)/(m))*(
#        n_right*(n_right-1.)*a_right*np.maximum(Delta_right + u - u_right,0.0)**(n_right - 2.0) ))[:-1]
#    
#    # unit matrix part
#    rows2, cols2    = np.indices((N+2,N+2))
#    rows2_diag      = np.diag(rows2, k=0)
#    cols2_diag      = np.diag(cols2, k=0)
#    cols2_diag.setflags(write=True)
#    cols2_diag      += (N+2)
#    unit_data       = np.ones_like(cols2_diag)
#    
#    # collect data
#    data    = np.concatenate((diag_data,
#                              sub_data,
#                              sup_data,
#                              unit_data))
#    row     = np.concatenate((rows_diag,
#                              rows_sub,
#                              rows_sup,
#                              rows2_diag))
#    
#    col     = np.concatenate((cols_diag,
#                              cols_sub,
#                              cols_sup,
#                              cols2_diag))
#    
#    return csr_matrix((data,(row,col)))

# other functions
def order_of_magnitude(x):
    return int(np.log10(x))

# chain class
class Chain(object):
    """
    
    Class of granular chains.
    
    Members
    -------
    
    N : int
        Number of beads in the chain. (default=2)
    m : array-like
        Masses of the beads in kg. (default=np.zeros(N))
    R : array-like
        Radii of the beads in meters. (default=np.zeros(N))
    a : array-like
        Prefactors containing elastic properties of the beads, always in simulation units. (default=np.zeros(N-1))
    force : array-like
        External force acting on the beads in SI units. (default=np.zeros(N))
    Delta : array-like
        Bead precompression in meters. (default=np.zeros(N - 1))
    n : array-like
        Exponents of interaction potential. (default=2.5*np.ones(N - 1))
    beta : float
        Time scaling parameter. Has to be set as sqrt(m0/(n*mu)) for some base mass m0. (default=1.0)
    m0 : float
        Base mass used for calculation of beta. (default=1.0)
    mu : float
        Prefactor scaling parameter. (default=1.0)
    
    """
    N = 2                           # number of beads
    m = np.zeros(N)                 # masses of beads
    R = np.zeros(N)                 # bead radii
    a = np.zeros(N - 1)             # prefactors
    force = np.zeros(N)             # external force
    Delta = np.zeros(N - 1)         # precompression
    n = 2.5*np.ones(N - 1)          # powers in Hertz potential
    beta = 1.0                      # time scaling
    m0 = 1.0                        # base mass
    mu = 1.0                        # prefactor scaling
    
    # initialization      
    def __init__(self, N_value=2, m_values=None, R_values=None, a_values=None, force_values=None, n_values=None, Delta_values=None, beta_value=1.0, m0_value=1.0, mu_value=1.0):
        assert type(N_value) is IntType
        self.N = N_value
        
        if m_values is None:
            self.m = np.zeros(self.N)
        else:
            self.m = m_values
        
        if R_values is None:
            self.R = np.zeros(self.N)
        else:
            self.R = R_values
        
        if a_values is None:
            self.a = np.zeros(self.N-1)
        else:
            self.a = a_values
        
        if force_values is None:
            self.force = np.zeros(self.N)
        else:
            self.force = force_values
        
        if n_values is None:
            self.n = 2.5*np.ones(self.N-1)
        else:
            self.n = n_values
        
        if Delta_values is None:
            self.Delta = np.zeros(self.N-1)
        else:
            self.Delta = Delta_values
        
        self.beta = beta_value
        self.m0 = m0_value
        self.mu = mu_value
    
    def print_properties(self):
        print 'chain properties:'
        print 'N =', self.N
        print 'm =', self.m
        print 'R =', self.R
        print 'force =', self.force
        print 'Delta =', self.Delta
        print 'n =', self.n
        print 'beta =', self.beta
        print 'm0 =', self.m0
        print 'mu =', self.mu
    
    def save(self, filename='test'):
        """
        
        Function that saves the instance of the chain to a file.
        
        """
        file = open(filename+'.txt','w')
        pickle.dump(self, file)
        file.close()
    
    def load(self, filename='test'):
        """
        
        Function that loads the instance of the chain from a file.
        
        """
        file = open(filename+'.txt','r')
        loaded_chain = pickle.load(file)
        
        self.N      = loaded_chain.N
        self.m      = loaded_chain.m
        self.R      = loaded_chain.R
        self.a      = loaded_chain.a
        self.force  = loaded_chain.force
        self.Delta  = loaded_chain.Delta
        self.n      = loaded_chain.n
        self.beta   = loaded_chain.beta
        self.m0     = loaded_chain.m0
        self.mu     = loaded_chain.mu
        
        file.close()
    
    def is_complete(self):
        """
        
        Function checking whether the chain was set up correctly.
        
        Returns
        -------
        
        is_complete : bool
        
        """
        is_complete = True
        
        if (type(self.N) is not IntType) or self.N < 2:
            warnings.warn('N not set up properly.')
            is_complete = False
        
        if self.m is None or len(self.m) != self.N:
            warnings.warn('m not set up properly.')
            is_complete = False
        
        if self.R is None or len(self.R) != self.N:
            warnings.warn('R not set up properly.')
            is_complete = False
        
        if self.a is None or len(self.a) != self.N - 1:
            warnings.warn('a not set up properly.')
            is_complete = False
        
        if self.force is None or len(self.force) != self.N:
            warnings.warn('force not set up properly.')
            is_complete = False
        
        if self.Delta is None or len(self.Delta) != self.N - 1:
            warnings.warn('Delta not set up properly.')
            is_complete = False
        
        if self.n is None or len(self.n) != self.N - 1:
            warnings.warn('n not set up properly.')
            is_complete = False
        
        if self.beta < 0.0:
            warnings.warn('beta not set up properly.')
            is_complete = False
        
        if self.m0 < 0.0:
            warnings.warn('m0 not set up properly.')
            is_complete = False
        
        if self.mu < 0.0:
            warnings.warn('mu not set up properly.')
            is_complete = False
        
        return is_complete
    
    def set_scalings(self, beta_value=1.0, mu_value=1.0):
        self.beta = beta_value
        self.mu = mu_value
    
    # set unscaled prefactors
    def set_prefactors(self, a_values):
        self.a = a_values/self.mu
    
    # set already scaled prefactors
    def set_scaled_prefactors(self, a_values):
        self.a = a_values
    
    def set_alternating_n(self, n1, n2, n_unit=1):
        """
        
        Setting alternating n-values in the chain.
        
    
        Parameters
        ----------
        
        n1 : float
            n value of odd bead numbers in the chain.
        
        n2 : float
            n value of even bead numbers the chain.
        
        n_unit : int, optional
            Length of units with same n. (default=1)
            
        """
        n_first     = np.repeat(n1, n_unit)
        n_second    = np.repeat(n2, n_unit)
        no_rep      = np.ceil( float(self.N)/float(2*n_unit) )
        self.n      = np.tile( np.concatenate((n_first, n_second)), no_rep )[:self.N-1]
#        self.n = np.ones(self.N - 1)
#        self.n[::2] = n1
#        self.n[1::2] = n2
    
    
    def set_tapered_n(self, n_from, n_to):
        """
        
        Function setting the n-values linearly increasing (decreasing) from n_from to n_to from left to right in the chain.
        
    
        Parameters
        ----------
        
        n_from : float
            n value at the left end of the chain.
        
        n_to : float
            n value at the right end of the chain.
            
        """
        self.n = np.linspace(start=n_from, stop=n_to, num=self.N-1)
    
    def n_noise(self, sigma, distr='gauss', from_bead=None):
        """
        
        Function adding noise to the interaction potential exponent of the beads.
        
    
        Parameters
        ----------
        
        sigma : float
            Standard deviation of the added noise.
        
        distr : string, optional
            Distribution for random noise. Options: 'gauss', 'uni'. (default='gauss')
        
        from_bead : int, optional
            Bead number from which to start the noise. (default=None)
            
        """
        assert sigma>=0
        if from_bead == None:
            from_bead=0
        if distr=='gauss':
            noise = np.random.normal(loc=0.0, scale=sigma, size=(self.N-1))
        elif distr=='uni':
            noise = np.random.uniform(low=-sigma, high=sigma, size=(self.N-1))
        self.n[from_bead:] += noise[from_bead:]
        
        #check if n went below 2
        small = np.sum( self.n[self.n<2.0] )
        self.n[self.n<2.0] = 2.0
        if small>0:
            warnings.warn('n_noise: n values smaller than 2 detected and set to 2')
        
    
    def m_noise(self, sigma):
        """
        
        Function adding noise to the masses of the beads.
        
    
        Parameters
        ----------
        
        sigma : float
            Standard deviation of the added noise.
            
        """
        noise = np.random.normal(loc=0.0, scale=sigma, size=(self.N))
        self.m += noise
    
    # calculate prefactors from elastic properties
    def calculate_prefactors(self, R_values=1.0, Y_values=1.0, sigma_values=1.0):
        # make arrays for arguments that were given as floats
        if type(R_values) is float or int:
            R = R_values*np.ones(self.N)
        else:
            R = R_values
        
        if type(Y_values) is float or int:
            Y = Y_values*np.ones(self.N)
        else:
            Y = Y_values
        
        if type(sigma_values) is float or int:
            sigma = sigma_values*np.ones(self.N)
        else:
            sigma = sigma_values
        
        self.a = np.ones(self.N-1)
        for i in range(len(self.a)):
            D = (3.0/4.0)*( (1.0 - sigma[i]**2.0)/Y[i] + (1.0 - sigma[i+1]**2.0)/Y[i+1] )
            R_eff = np.sqrt(R[i]*R[i+1]/(R[i] + R[i+1]))
            self.a[i] = (0.4/D)*R_eff/self.mu
    
    def set_Delta_from_force(self, F_value, from_bead=None, to_bead=None):
        """
        
        Function setting the apropriate precompression delta for a given force assuming a uniform chain. The force has to be give in SI units.
    
        Parameters
        ----------
        
        F_value : double
            Externally applied force in SI units (N).
        from_bead : int
            Index of first bead of the section of the chain for which to apply precompression. (default=None)
        to_bead : int
            Index of last bead of the section of the chain for which to apply precompression. (default=None)
        
        """
        # set whole chain as precompression section by default
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.N - 1
        
        # radii, masses and prefactors in precompression section
        R_section = self.R[from_bead:to_bead]
        m_section = self.m[from_bead:to_bead]
        a_section = self.a[from_bead:to_bead]
        n_section = self.n[from_bead:to_bead]
        
        # assertations for uniform chain in precompression section
        assert np.all( R_section == R_section[0] )
        assert np.all( m_section == m_section[0] )
        assert np.all( a_section == a_section[0] )
        assert np.all( n_section == n_section[0] )
        
        n = n_section[0]
        mu = self.mu
        a = a_section[0]
        
        Delta_value = (F_value/(n*mu*a))**(1.0/(n - 1.0))
        self.Delta[from_bead:to_bead] = Delta_value*np.ones_like(R_section)
    
    def apply_precompression(self, F_value, from_bead=None, to_bead=None):
        """
        
        Function setting the apropriate precompression delta and constant external force for a given precompression force. The force has to be give in SI units.
    
        Parameters
        ----------
        
        F_value : double
            Externally applied force in SI units (N).
        from_bead : int
            Index of first bead of the section of the chain for which to apply precompression. (default=None)
        to_bead : int
            Index of last bead of the section of the chain for which to apply precompression. (default=None)
        
        """
        # set whole chain as precompression section by default
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.N - 1
        
        self.set_Delta_from_force(F_value, from_bead, to_bead)
#        self.force = np.zeros(self.N)
        self.force[from_bead] = F_value
        self.force[to_bead] = -F_value
        
    # check scalings
    def analyze_scalings(self):
        # masses
        m_max = np.max(self.m)
        m_max_order = order_of_magnitude( m_max )
        m_min_order = order_of_magnitude( np.min(self.m) )
        if m_max_order != m_min_order:
            print 'masses are of order 10^', m_min_order, 'to order 10^', m_max_order
        else:
            print 'masses are of order 10^', m_min_order
        
        # prefactors
        a_max_order = order_of_magnitude( np.max(self.a) )
        a_min_order = order_of_magnitude( np.min(self.a) )
        if a_max_order != a_min_order:
            print 'prefactors are of order 10^', a_min_order, 'to order 10^', a_max_order
        else:
            print 'prefactors are of order 10^', a_min_order
    
    def set_wall(self, mass_factor):
        assert self.m is not None
        self.m[-1] = mass_factor*self.m[-2]
    
    # get function arguments for simulation
    def get_Fargs(self):
        return [self.m, np.concatenate(([0], self.a, [0]), axis=0), np.concatenate(([0], self.Delta, [0]), axis=0), np.concatenate(([self.n[0]], self.n, [self.n[-1]])), self.N, self.beta, self.mu, self.force]
    
    def get_bead_numbers(self):
        return np.arange(self.N)
    

# 1D simulation class
class Simulation_1D(object):
    """
    
    Class for one-dimensional simulations of granular chains.
    
    Members
    -------
    
    sim_chain : Chain object
        Chain on which to carry out the simulation. (default=None)
    
    t0 : float
        Start time of simulation. (default=0.0)
    
    t1 : float
        End time of simulation. (default=1.0)
    
    dt : float
        Time step after which the simulation results are stored. Non necessarily the timestep for the numerical integrator. (default=0.01)
    
    z0 : array-like
        Initial value for displacement (z0[:N]]) and velocity (z0[N:]]) of beads in chain, where N is the number of beads in sim_chain. (default=None)
    
    data : array-like
        2D array with size storing displacements, velocities and time at each time step. The first index corresponds to time in the simulation and the second index selects the stored value. (default=None)
    
    """
    sim_chain = None                # chain
    t0 = 0.0                        # start time
    t1 = 1.0                        # end time
    dt = 0.01                       # time step
    z0 = None                       # initial conditions
    data = None                     # simulation data
    
    # initialization
    def __init__(self, sim_chain_val=None, t0_value=0.0, t1_value=1.0, dt_value=0.01, z0_values=None):
        if sim_chain_val is None:
            self.sim_chain = Chain()
        else:
            self.sim_chain = sim_chain_val
        
        self.t0 = t0_value
        self.t1 = t1_value
        self.dt = dt_value
        
        if z0_values is None:
            self.z0 = np.zeros(self.sim_chain.N*2)
        else:
            self.z0 = z0_values
    
    def initial_velocity_noise(self, sigma):
        """
        
        Function adding noise to the initial velocity of the beads.
        
    
        Parameters
        ----------
        
        sigma : float
            Standard deviation of the added noise.
            
        """
        N = self.get_N()
        noise = np.random.normal(loc=0.0, scale=sigma, size=(N))
        self.z0[N:2*N] += noise
    
    def set_alternating_n(self, n1, n2, n_unit=1):
        """
        
        Setting alternating n-values in the chain.
        
    
        Parameters
        ----------
        
        n1 : float
            n value of odd bead numbers in the chain.
        
        n2 : float
            n value of even bead numbers the chain.
        
        n_unit : int, optional
            Length of units with same n. (default=1)
            
        """
        self.sim_chain.set_alternating_n(n1,n2, n_unit=n_unit)
    
    def set_tapered_n(self, n_from, n_to):
        """
        
        Function setting the n-values linearly increasing (decreasing) from n_from to n_to from left to right in the chain.
        
    
        Parameters
        ----------
        
        n_from : float
            n value at the left end of the chain.
        
        n_to : float
            n value at the right end of the chain.
            
        """
        self.sim_chain.set_tapered_n(n_from, n_to)
    
    def n_noise(self, sigma, distr='gauss', from_bead=None):
        """
        
        Function adding noise to the interaction potential exponent of the beads.
        
    
        Parameters
        ----------
        
        sigma : float
            Standard deviation of the added noise.
        
        distr : string, optional
            Distribution for random noise. Options: 'gauss', 'uni'. (default='gauss')
        
        from_bead : int, optional
            Bead number from which to start the noise. (default=None)
            
        """
        self.sim_chain.n_noise(sigma, distr=distr, from_bead=from_bead)
    
    def m_noise(self, sigma):
        """
        
        Function adding noise to the interaction potential exponent of the beads.
        
    
        Parameters
        ----------
        
        sigma : float
            Standard deviation of the added noise.
            
        """
        self.sim_chain.m_noise(sigma)
    
    def print_properties(self):
        print 'simulation properties:'
        print 't0 =', self.t0
        print 't1 =', self.t1
        print 'dt =', self.dt
        print 'z0 =', self.z0
    
    # overlap variables
    def overlap(self, t_measure=None):
        assert self.data is not None
        time = self.get_time()
        N = self.get_N()
        
        if t_measure is not None:
            # find index for time closest to t_measure
            idx = self._find_index_for_time(t_measure)
            
            # calculate overlap
            u = self.data[idx,:N]
            u_left = u[:-1]
            u_right = u[1:]
            overlap = np.maximum( u_left - u_right,0 )
            return overlap
            
        else:
            overlaps = np.zeros((len(time),N-1))
            ctr = 0
            for tau in time:
                overlaps[ctr,:] = self.overlap(tau)
                ctr += 1
            return overlaps
    
    def save(self, dirname='testdir', chainname='testchain', dataname='data'):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.sim_chain.save(dirname+'/'+chainname)
        self.save_data_as_txt(dirname+'/'+dataname)
    
    def save_data_as_npy(self, filename='test'):
        np.save( filename, self.data )
    
    def save_data_as_txt(self, filename='test', fmt='%.4e'):
        np.savetxt( filename, self.data, fmt=fmt )
        
    # shrink data
    def shrink_data(self, factor):
        assert self.data is not None
        self.data = self.data[::factor, :]
    
    def shrink_data_to(self, n_data):
        assert self.data is not None
        factor = int( float(self.data.shape[0])/float(n_data) )
        if factor < 1:
            warnings.warn('Data is already smaller than suggested size.')
        else:
            self.data = self.data[::factor, :]
    
    # run simulation
    def run(self, use_jac=False, atol=1e-20, rtol=1e-10, n_steps_max=500, print_runtime=False, exit_check=False, mono_check=False, mono_rtol=1e-5, exit_bead=None, exit_threshold=0.0):
        """
        
        Runs the simulation and stores results in data.
        
        Parameters
        ----------
        
        use_jac : bool, optional
            Use jacobian in integration. (default=True)
        atol : float, optional
            Absolute tolerance for solution. (default=1e-10)
        rtol : float, optional
            Relative tolerance for solution. (default=1e-6)
        print_runtime : bool, optional
            Print runtime of solver. (default=False)
        exit_check : bool, optional
            Stop simulation when wave reaches end of chain. (default=False)
        mono_check : bool, optional
            Stop simulation when bead velocities reach steady state, i.e. drift apart without ever touching each other again. (default=False)
        mono_rtol : float, optional
            Relative tolareance for determining the steady state of the velocities. (default=1e-5)
        exit_bead : int, optional
            Index of bead at which to check if the wave has arrived. (default=None)
        exit_threshold : float, optional
            Threshold for the bead velocity for detection at the end of the chain. (default=0.0)
        
        """
        if print_runtime:
            start = time.time()
        
        assert self.sim_chain is not None
        assert self.sim_chain.is_complete()
        
        if exit_check:
            if exit_bead==None:
                exit_bead=self.get_N() - 1
            assert self.get_N() > exit_bead >= 0
        
        # get function arguments
        Fargs = self.sim_chain.get_Fargs()
        N = Fargs[4]
        
        # set up solver
#        solver = ode(F, None).set_integrator('dopri5', verbosity=1, first_step=self.dt, atol=rtol)
        if use_jac:
            solver = ode(F, Jac).set_integrator('vode', first_step=self.dt, rtol=rtol, atol=atol, nsteps=n_steps_max)
        else:
            solver = ode(F, None).set_integrator('vode', first_step=self.dt, rtol=rtol, atol=atol, nsteps=n_steps_max)
#        solver = ode(F, None).set_integrator('lsoda', first_step=self.dt, rtol=rtol, ixpr=True)
        #solver = ode(F, None).set_integrator('lsoda', first_step=self.dt)
        
        # initial condition in simulation form
        u_sim = np.concatenate(([0], self.z0[:N], [0]), axis=0)
        v_sim = np.concatenate(([0], self.z0[N:], [0]), axis=0)
        z0_sim = np.concatenate((u_sim, v_sim), axis=0)
        solver.set_initial_value(z0_sim, self.t0)
        
        # function arguments
        Fargs = self.sim_chain.get_Fargs()
        solver.set_f_params(Fargs)
        
        # jacobian arguments
        solver.set_jac_params(Fargs)
        
        # data storage
        N = Fargs[4]
        n_steps = int(self.t1/self.dt + 1)
        self.data = np.zeros((n_steps, 2*N + 1))
        
        # store initial condition
        self.data[0,:-1] = self.z0
        self.data[0,-1] = self.t0
        
        # run time integration
        ctr = 1
        while solver.successful() and solver.t < self.t1:
            # integrate
            solver.integrate(solver.t + self.dt)
            
            # store results
            self.data[ctr,:N] = solver.y[1:N+1]             # u values
            self.data[ctr,N:2*N] = solver.y[N+3:2*N+3]      # v values
            self.data[ctr,-1] = solver.t                    # time
            
            # stop integration if next step would take it too far in time
            if solver.t + self.dt > self.t1:
                break
            
            # exit check
            if exit_check:
                # velocity of last bead
                vel_last_bead = self.data[ctr,N+exit_bead]
                
                if vel_last_bead > exit_threshold:
                    # cut unnecessary zeros
                    self.data = self.data[:ctr+1,:]
#                    np.delete(self.data, np.arange(ctr+1,n_steps), axis=0)
                    
                    # adjust simulation end time
                    self.t1 = self.data[ctr,-1]
                    
                    print "run: wave reached end of chain and the simulation was ended before the sugggested simulation end time."
            
            # emono check
            if mono_check:
                tol = mono_rtol*np.abs(self.data[ctr,N:2*N]).max()
                if np.all( np.diff(self.data[ctr,N:2*N])>=(-tol) ):
                    # cut unnecessary zeros
                    self.data = self.data[:ctr+1,:]
#                    np.delete(self.data, np.arange(ctr+1,n_steps), axis=0)
                    
                    # adjust simulation end time
                    self.t1 = self.data[ctr,-1]
                    
                    print "run: chain reached velocitiy equilibrium and the simulation was ended before the sugggested simulation end time."
                
                
            ctr += 1
        
        if solver.t < self.t1 - self.dt:
            print "Simulation didn't work! t_end =", solver.t
        
        if print_runtime:
            end = time.time()
            print 'runtime:', (end - start), 's'
    
    
    # get length of chain
    def get_N(self):
        assert self.sim_chain is not None
        return self.sim_chain.N
    
    # get array of times
    def get_time(self):
        assert self.data is not None
        return self.data[:,-1]
    
    # get velocities of beads at a given time
    def get_velocities(self, t_measure):
        self._check_time(t_measure)
        idx = self._find_index_for_time(t_measure)
        N = self.get_N()
        return self.data[idx,N:2*N]
    
    def get_displacement(self, bead_number, from_time, to_time):
        self._check_time(from_time)
        self._check_time(to_time)
        assert from_time < to_time
        
        from_idx    = self._find_index_for_time(from_time)
        to_idx      = self._find_index_for_time(to_time)
        return self.data[from_idx:to_idx+1, bead_number]
    
    def v_max(self):
        N = self.get_N()
        return np.max( self.data[:,N:2*N] )
    
    def v_min(self):
        N = self.get_N()
        return np.min( self.data[:,N:2*N] )
    
    # bead velocity for a given bead
    def get_bead_velocity(self, bead_number, t_1=None, t_2=None):
        if t_1 is None:
            t_1 = self.t0
        if t_2 is None:
            t_2 = self.t1
        
        N = self.get_N()
        if bead_number >= N:
            warnings.warn('bead number out of range')
            return np.nan
        
        self._check_time(t_1)
        self._check_time(t_2)
        idx_1 = self._find_index_for_time(t_1)
        idx_2 = self._find_index_for_time(t_2)
        return self.data[idx_1:idx_2+1,N+bead_number]
    
    # get time difference for one index step in self.data
    def get_dt_per_index(self):
        """
        Get time difference for one index step in self.data assuming uniform time steps.
        """
        dt = self.data[1,-1] - self.data[0,-1]
        return dt
        
    # measure kinetic energy
    def measure_kinetic_energy(self, t_measure=None):
        """
        
        Function measuring the total kinetic energy for all times in the simulation (t_measure is None) or for one particular time t_measure. In the re-scaled equation of motion the kinetic energy is calculated by E_kin = 0.5*(m/m0)*v**2, where m0 is the base mass used for the time scaling parameter beta.
        
        
        Returns
        -------
        
        kinetic_energies: array-like (t_measure is None)
            Total kinetic energy of the chain for all time steps in the simulation.
        
        kinetic_energy : double (t_measure is not None)
            Total kinetic energy of the chain at time t_measure.
        
    
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the kinetic energy.
            
        """
        assert self.data is not None
        time = self.get_time()
        N = self.get_N()
        m0 = self.sim_chain.m0
        #print 'm0', m0
        
        if t_measure is not None:
            # find index for time closest to t_measure
            idx = self._find_index_for_time(t_measure)
            
            # calculate kinetic energy
            velocities = self.data[idx,N:2*N]
            masses = self.sim_chain.m
            kinetic_energy = 0.5*np.sum( np.multiply(masses/m0, velocities**2.0) )
            return kinetic_energy
        else:
            kinetic_energies = np.zeros(len(time))
            ctr = 0
            for tau in time:
                kinetic_energies[ctr] = self.measure_kinetic_energy(tau)
                ctr += 1
            return kinetic_energies
    
    # measure potential energy
    def measure_potential_energy(self, t_measure=None):
        """
        
        Function measuring the total potential energy of the chain for all times in the simulation (t_measure is None) or for one particular time t_measure. In the re-scaled equation of motion the potential energy is calculated from the potential V(overlap) = (a/n)*overlap**n .
        
        Returns
        -------
        
        potential_energies: array-like (t_measure is None)
            Total potential energy of the chain for all time steps in the simulation.
        potential_energy : double (t_measure is not None)
            Total potential energy of the chain at time t_measure.
        
    
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the potential energy.
        
        """
        assert self.data is not None
        time = self.get_time()
        
        if t_measure is not None:
            self._check_time(t_measure)
            
            # calculate potential energy
            overlap = self.overlap(t_measure)
            a = self.sim_chain.a
            n = self.sim_chain.n
            potential_energy = np.sum( np.multiply( (1.0/n), np.multiply( a, np.power(overlap,n) ) ) )
            
            return potential_energy
        
        else:
            # calculate potential energy
            overlaps = self.overlap()
            a = self.sim_chain.a
            n = self.sim_chain.n
            potential_energies = np.zeros(len(time))
            for i in range(len(time)):
                potential_energies[i] = np.sum( np.multiply( (1.0/n), np.multiply( a, np.power(overlaps[i,:],n) ) ) )
            
            return potential_energies
    
    # measure kinetic energy distribution
    def measure_kinetic_energy_distr(self, t_measure=None):
        """
        
        Function measuring the total kinetic energy for all times in the simulation (t_measure is None) or for one particular time t_measure. In the re-scaled equation of motion the kinetic energy is calculated by E_kin = 0.5*(m/m0)*v**2, where m0 is the base mass used for the time scaling parameter beta.
        
        
        Returns
        -------
        
        kinetic_energy_distrs: array-like (t_measure is None)
            Kinetic energy distributions of the chain for all time steps in the simulation.
        
        kinetic_energy_distr : double (t_measure is not None)
            Kinetic energy distribution of the chain at time t_measure.
        
    
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the kinetic energy.
            
        """
        assert self.data is not None
        time = self.get_time()
        N = self.get_N()
        m0 = self.sim_chain.m0
        #print 'm0', m0
        
        if t_measure is not None:
            # find index for time closest to t_measure
            idx = self._find_index_for_time(t_measure)
            
            # calculate kinetic energy
            velocities = self.data[idx,N:2*N]
            masses = self.sim_chain.m
            kinetic_energy_distr = 0.5*np.multiply(masses/m0, velocities**2.0)
            return kinetic_energy_distr
        else:
            kinetic_energy_distrs = np.zeros((len(time),N))
            ctr = 0
            for tau in time:
                kinetic_energy_distrs[ctr,:] = self.measure_kinetic_energy_distr(tau)
                ctr += 1
            return kinetic_energy_distrs
    
    # measure potential energy distribution
    def measure_potential_energy_distr(self, t_measure=None):
        """
        
        Function measuring the potential energy distribution of the chain. In the re-scaled equation of motion the potential energy is calculated from the potential V(overlap) = (a/n)*overlap**n .
        
        
        Returns
        -------
        
        potential_energy_distrs: array-like (t_measure is None)
            Potential energy distribution of the chain for all time steps in the simulation.
            
        potential_energy_distr : double (t_measure is not None)
            Potential energy distribution of the chain at time t_measure.
        
        
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the potential energy distribution.
        
        """
        assert self.data is not None
        time = self.get_time()
        
        a = self.sim_chain.a
        n = self.sim_chain.n
        N = self.get_N()
        
        if t_measure is not None:
            self._check_time(t_measure)
            
            # calculate potential energy
            overlap = self.overlap(t_measure)
            potential_energy_distr = np.multiply( (1./n), np.multiply(a, overlap**n) )
            
            return potential_energy_distr
        
        else:
            # calculate potential energy
            #overlaps = self.overlap()
            potential_energy_distrs = np.zeros((len(time),N-1))
            ctr = 0
            for t in time:
                potential_energy_distrs[ctr,:] = self.measure_potential_energy_distr(t)
                ctr += 1
            
            return potential_energy_distrs
    
    # measure total energy
    def measure_total_energy(self, t_measure=None):
        """
        
        Function measuring the total energy of the chain for all times in the simulation (t_measure is None) or for one particular time t_measure.
        
        Returns
        -------
        
        total_energies: array-like (t_measure is None)
        total_energy : double (t_measure is not None)
        
    
        Parameters
        ----------
        
        t_measure : double, optional
            Time in the simulation at which to measure the total energy.
        
        """
        assert self.data is not None
        
        if t_measure is not None:
            self._check_time(t_measure)
            
            # calculate total energy
            kinetic_energy = self.measure_kinetic_energy(t_measure)
            potential_energy = self.measure_potential_energy(t_measure)
            total_energy = kinetic_energy + potential_energy
            
            return total_energy
            
        else:
            # calculate total energy
            kinetic_energies = self.measure_kinetic_energy()
            potential_energies = self.measure_potential_energy()
            total_energies = kinetic_energies + potential_energies
            
            return total_energies
    
    def calculate_energy_conservation_error(self):
        """
        
        Function calculating the relative error in energy conservation between the start and the end of the simulation.
        
        Returns
        -------
        
        error : float
        
        """
        assert self.data is not None
        # calculate total energy at start and end of simulation
        energy_start = self.measure_total_energy(self.t0)
        energy_end = self.measure_total_energy(self.t1)
        
        # calculate accuracy
        error = abs(1.0 - energy_start/energy_end)
        
        return error
    
    def plot_velocity(self, t_plot, title="", latexsave=False, filename='test_plot', figureheight='0.8\\textwidth', figurewidth='\\textwidth', y_max=None, y_min=None, from_bead=None, to_bead=None, interface=None, legend=True, loc=1, x_min=None, x_max=None, x_shift=0):
        """
        
        Function plotting the bead velocities of the chain
    
        Parameters
        ----------
        
        t_plot : float or list
            Times at which to plot the velocities.
        
        title : str, optional
            Plottitle. (default='')
        
        latexsave : bool, optional
            Whether to save a tikz-file for the plot to be included in a LaTeX document. (default=False)
        
        filename : str, optional, (default='test_plot')
        
        figureheight : str, optional 
            Figureheight for specification in tikz file. Needs to be a string compatible with length units in LaTeX. (default='5cm')
        
        figurewidth : str, optional 
            Figureheight for specification in tikz file. Needs to be a string compatible with length units in LaTeX. (default='6cm')
        
        y_max : float, optional
            Maximum value on y-axis. (default=None)
        
        y_min : float, optional
            Minimum value on y-axis. (default=None)
        
        interface : int, optional
            Interface beads. (default=None)
        
        loc : int, optional
            Legend location. (default=1)
            
        """
        if not isinstance(t_plot, list):
            t_plot = [t_plot]
        if (not isinstance(interface, list)) and (interface != None):
            interface = [interface]
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.get_N() - 1
        assert 0<=from_bead<to_bead<self.get_N()
        N = self.get_N()
        bead_numbers = self.sim_chain.get_bead_numbers()[from_bead:to_bead+1]
        
        fig = plt.figure()
        ax = plt.subplot(111)        
        #        plt.tight_layout()
        if y_max == None: y_max = self.v_max()
        if y_min == None: y_min = self.v_min()
        if x_max == None: x_max = bead_numbers.max()
        if x_min == None: x_min = bead_numbers.min()
        
        if latexsave:
            for t in t_plot:
                idx = self._find_index_for_time(t)
                t_plot_real = self.get_time()[idx]
                velocity = self.get_velocities(t)[from_bead:to_bead+1]
                plt.plot(bead_numbers+x_shift, velocity, lw=2, label="\\tiny{"+'t='+str(t_plot_real)+"}")
            ax.set_xlabel("\\footnotesize{bead number}")
            ax.set_ylabel("\\footnotesize{$v$ (c.u.)}")
            if title=='' or title==None:
                pass
            else:
                ax.set_title("\\footnotesize{"+title+"}")
            ax.set_ylim(ymin=y_min, ymax=y_max)
            ax.set_xlim(xmin=x_min, xmax=x_max)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            if legend: ax.legend(loc=loc)
            plt.tight_layout()
            
            # interface
            if not interface == None:
                for i_face in interface:
                    delta_y = y_max-y_min
                    plt.plot((i_face,i_face),(y_min+0.0*delta_y,y_max-0.0*delta_y),'r--', lw=2)
            tikz_save(filename+'.tex', figureheight=figureheight, figurewidth=figurewidth)
        else:
            for t in t_plot:
                idx = self._find_index_for_time(t)
                t_plot_real = self.get_time()[idx]
                velocity = self.get_velocities(t)[from_bead:to_bead+1]
                plt.plot(bead_numbers+x_shift, velocity, marker='.', ms=3, label='t='+str(t_plot_real))
            ax.set_xlabel("bead number")
            ax.set_ylabel("velocity (computer units)")
            ax.set_title(title)
            ax.set_ylim(ymin=y_min, ymax=y_max)
            ax.set_xlim(xmin=x_min, xmax=x_max)
            ax.legend(loc=loc)
            
            # interface
            if not interface == None:
                delta_y = y_max-y_min
                plt.plot((interface,interface),(y_min+0.0*delta_y,y_max-0.0*delta_y),'r--', lw=2)
        plt.show()
    
    def plot_overlap(self, t_plot, title="", latexsave=False, filename='test_plot', figureheight='0.8\\textwidth', figurewidth='\\textwidth', y_max=None, y_min=None, from_bead=None, to_bead=None, interface=None, legend=True, loc=1, x_min=None, x_max=None, x_shift=0):
        """
        
        Function plotting the bead velocities of the chain
    
        Parameters
        ----------
        
        t_plot : float or list
            Times at which to plot the velocities.
        
        title : str, optional
            Plottitle. (default='')
        
        latexsave : bool, optional
            Whether to save a tikz-file for the plot to be included in a LaTeX document. (default=False)
        
        filename : str, optional, (default='test_plot')
        
        figureheight : str, optional 
            Figureheight for specification in tikz file. Needs to be a string compatible with length units in LaTeX. (default='5cm')
        
        figurewidth : str, optional 
            Figureheight for specification in tikz file. Needs to be a string compatible with length units in LaTeX. (default='6cm')
        
        y_max : float, optional
            Maximum value on y-axis. (default=None)
        
        y_min : float, optional
            Minimum value on y-axis. (default=None)
        
        interface : int, optional
            Interface beads. (default=None)
        
        loc : int, optional
            Legend location. (default=1)
            
        """
        if not isinstance(t_plot, list):
            t_plot = [t_plot]
        if (not isinstance(interface, list)) and (interface != None):
            interface = [interface]
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.get_N() - 1
        assert 0<=from_bead<to_bead<self.get_N()
        N = self.get_N()
        bead_numbers = self.sim_chain.get_bead_numbers()[from_bead:to_bead]
        
        fig = plt.figure()
        ax = plt.subplot(111)        
        #        plt.tight_layout()
        if y_max == None: y_max = self.v_max()
        if y_min == None: y_min = self.v_min()
        if x_max == None: x_max = bead_numbers.max()
        if x_min == None: x_min = bead_numbers.min()
        
        if latexsave:
            for t in t_plot:
                idx         = self._find_index_for_time(t)
                t_plot_real = self.get_time()[idx]
                overlap     = self.overlap(t)[from_bead:to_bead+1]
                plt.plot(bead_numbers+x_shift, overlap, lw=2, label="\\tiny{"+'t='+str(t_plot_real)+"}")
            ax.set_xlabel("\\footnotesize{bead number}")
            ax.set_ylabel("\\footnotesize{$\\delta$ (c.u.)}")
            if title=='' or title==None:
                pass
            else:
                ax.set_title("\\footnotesize{"+title+"}")
            #ax.set_ylim(ymin=y_min, ymax=y_max)
            #ax.set_xlim(xmin=x_min, xmax=x_max)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            if legend: ax.legend(loc=loc)
            plt.tight_layout()
            
            # interface
            if not interface == None:
                for i_face in interface:
                    delta_y = y_max-y_min
                    plt.plot((i_face,i_face),(y_min+0.0*delta_y,y_max-0.0*delta_y),'r--', lw=2)
            tikz_save(filename+'.tex', figureheight=figureheight, figurewidth=figurewidth)
        else:
            for t in t_plot:
                idx         = self._find_index_for_time(t)
                t_plot_real = self.get_time()[idx]
                overlap     = self.overlap(t)[from_bead:to_bead+1]
                plt.plot(bead_numbers+x_shift, overlap, marker='.', ms=3, label='t='+str(t_plot_real))
            ax.set_xlabel("bead number")
            ax.set_ylabel("overlap (computer units)")
            ax.set_title(title)
            #ax.set_ylim(ymin=y_min, ymax=y_max)
            #ax.set_xlim(xmin=x_min, xmax=x_max)
            ax.legend(loc=loc)
            
            # interface
            if not interface == None:
                delta_y = y_max-y_min
                plt.plot((interface,interface),(y_min+0.0*delta_y,y_max-0.0*delta_y),'r--', lw=2)
        plt.show()
    
    def plot_gap(self, left_beads, title="", latexsave=False, filename='test_plot_gap', figureheight='0.8\\textwidth', figurewidth='\\textwidth', y_max=None, y_min=None, from_time=None, to_time=None, loc=1):
        """
        
        Function plotting the bead velocities of the chain
    
        Parameters
        ----------
        
        left_beads : float or list
            Times at which to plot the velocities.
        
        title : str, optional
            Plottitle. (default='')
        
        latexsave : bool, optional
            Whether to save a tikz-file for the plot to be included in a LaTeX document. (default=False)
        
        filename : str, optional, (default='test_plot')
        
        figureheight : str, optional 
            Figureheight for specification in tikz file. Needs to be a string compatible with length units in LaTeX. (default='5cm')
        
        figurewidth : str, optional 
            Figureheight for specification in tikz file. Needs to be a string compatible with length units in LaTeX. (default='6cm')
        
        y_max : float, optional
            Maximum value on y-axis. (default=None)
        
        y_min : float, optional
            Minimum value on y-axis. (default=None)
        
        interface : int, optional
            Interface beads. (default=None)
        
        loc : int, optional
            Legend location. (default=1)
            
        """
        
        if not isinstance(left_beads, list):
            left_beads = [left_beads]
        
        fig = plt.figure()
        ax = plt.subplot(111)        
        #        plt.tight_layout()
        
        if from_time == None: 
            from_time=0
        if to_time == None: 
            to_time=self.t1
        
#        if y_max == None: y_max = gap.max()
#        if y_min == None: y_min = gap.min()
        
        from_idx            = self._find_index_for_time(from_time)
        to_idx              = self._find_index_for_time(to_time)
        time                = self.get_time()[from_idx:to_idx+1]
        
        if latexsave:
            for left_bead in left_beads:
                left_bead_displ = self.get_displacement(left_bead, from_time, to_time)
                right_bead_displ= self.get_displacement(left_bead+1, from_time, to_time)
                gap             = np.maximum(right_bead_displ - left_bead_displ, 0.0)
                plt.plot(time, gap, label='\\tiny gap '+str(left_bead)+','+str(left_bead+1))
            
            ax.set_xlabel("\\footnotesize time (c.u.)")
            ax.set_ylabel("\\footnotesize gap (c.u.)")
            ax.set_title(title)
            if (y_min != None) and (y_max != None):
                ax.set_ylim(ymin=y_min, ymax=y_max)
            ax.set_xlim(xmin=time.min(), xmax=time.max())
            ax.legend(loc=loc)
            plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
            plt.tight_layout()
            tikz_save(filename+'.tex', figureheight=figureheight, figurewidth=figurewidth)
        else:
            for left_bead in left_beads:
                left_bead_displ = self.get_displacement(left_bead, from_time, to_time)
                right_bead_displ= self.get_displacement(left_bead+1, from_time, to_time)
                gap             = np.maximum(right_bead_displ - left_bead_displ, 0.0)
                plt.plot(time, gap, label='gap '+str(left_bead)+','+str(left_bead+1))
            
            ax.set_xlabel("time (c.u.)")
            ax.set_ylabel("gap (c.u.)")
            ax.set_title(title)
            if (y_min != None) and (y_max != None):
                ax.set_ylim(ymin=y_min, ymax=y_max)
            ax.set_xlim(xmin=time.min(), xmax=time.max())
            ax.legend(loc=loc)
            plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        plt.show()
    
    def plot_distance(self, left_beads, title="", latexsave=False, filename='test_plot_gap', figureheight='0.8\\textwidth', figurewidth='\\textwidth', y_max=None, y_min=None, from_time=None, to_time=None, loc=1):
        """
        
        Function plotting the bead distances in the chain
    
        Parameters
        ----------
        
        left_beads : float or list
            Times at which to plot the velocities.
        
        title : str, optional
            Plottitle. (default='')
        
        latexsave : bool, optional
            Whether to save a tikz-file for the plot to be included in a LaTeX document. (default=False)
        
        filename : str, optional, (default='test_plot')
        
        figureheight : str, optional 
            Figureheight for specification in tikz file. Needs to be a string compatible with length units in LaTeX. (default='5cm')
        
        figurewidth : str, optional 
            Figureheight for specification in tikz file. Needs to be a string compatible with length units in LaTeX. (default='6cm')
        
        y_max : float, optional
            Maximum value on y-axis. (default=None)
        
        y_min : float, optional
            Minimum value on y-axis. (default=None)
        
        interface : int, optional
            Interface beads. (default=None)
        
        loc : int, optional
            Legend location. (default=1)
            
        """
        
        if not isinstance(left_beads, list):
            left_beads = [left_beads]
        
        fig = plt.figure()
        ax = plt.subplot(111)        
        #        plt.tight_layout()
        
        if from_time == None: 
            from_time=0
        if to_time == None: 
            to_time=self.t1
        
#        if y_max == None: y_max = gap.max()
#        if y_min == None: y_min = gap.min()
        
        from_idx            = self._find_index_for_time(from_time)
        to_idx              = self._find_index_for_time(to_time)
        time                = self.get_time()[from_idx:to_idx+1]
        
        if latexsave:
            for left_bead in left_beads:
                left_bead_displ = self.get_displacement(left_bead, from_time, to_time)
                right_bead_displ= self.get_displacement(left_bead+1, from_time, to_time)
                dist            = right_bead_displ - left_bead_displ
                plt.plot(time, dist, label='\\tiny gap '+str(left_bead)+','+str(left_bead+1))
            
            ax.set_xlabel("\\footnotesize time (c.u.)")
            ax.set_ylabel("\\footnotesize distance (c.u.)")
            ax.set_title(title)
            if (y_min != None) and (y_max != None):
                ax.set_ylim(ymin=y_min, ymax=y_max)
            ax.set_xlim(xmin=time.min(), xmax=time.max())
            ax.legend(loc=loc)
            plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
            plt.tight_layout()
            tikz_save(filename+'.tex', figureheight=figureheight, figurewidth=figurewidth)
        else:
            for left_bead in left_beads:
                left_bead_displ = self.get_displacement(left_bead, from_time, to_time)
                right_bead_displ= self.get_displacement(left_bead+1, from_time, to_time)
                dist            = right_bead_displ - left_bead_displ
                plt.plot(time, dist, label='gap '+str(left_bead)+','+str(left_bead+1))
            
            ax.set_xlabel("time (c.u.)")
            ax.set_ylabel("distance (c.u.)")
            ax.set_title(title)
            if (y_min != None) and (y_max != None):
                ax.set_ylim(ymin=y_min, ymax=y_max)
            ax.set_xlim(xmin=time.min(), xmax=time.max())
            ax.legend(loc=loc)
            plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        plt.show()
    
    def animate(self, interval=100, title='test', medium_from=None, medium_to=None, from_time=None, to_time=None):
        """
        
        Function creating an animation of the eave propagation in the chain.
        
        Returns
        -------
        
        anim : animation object        
    
        Parameters
        ----------
        
        interval : int, optional
            Time between frames in milliseconds. (default=100)
        
        title : str, optional
            Title of plot. (default='test')
        
        medium_from : int, optional
            Bead number at which a different medium in the chain is starting. (default=None)
        
        medium_to : int, optional
            Bead number at which a different medium in the chain is ending. (default=None)
        
        from_time : float, optional
            Animation start time. (default=None)
        
        to_time : float, optional
            Animation end time. (default=None)
        
        """
        if from_time == None:
            from_time = 0
        if to_time == None:
            to_time = self.t1
        
        from_idx = self._find_index_for_time(from_time)
        to_idx = self._find_index_for_time(to_time)
        
        # data input
        N = self.get_N()
        plotting_data = self.data[from_idx:to_idx+1,N:2*N]
        time = self.get_time()[from_idx:to_idx+1]
        y_min = np.min(plotting_data)
        y_max = np.max(plotting_data)
        n_frames = plotting_data.shape[0]
        
        # initialization of plots
        fig = plt.figure()
        ax = plt.axes(xlim=(0, N), ylim=(y_min, y_max))
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        time_text.set_text('')
        l, = plt.plot([], [], '.-')
        plt.xlabel('bead number')
        plt.ylabel('velocity (computer units)')
        plt.title(title)
        
        # visualize interfaces
        if medium_from is not None:
            if medium_to is None:
                medium_to = N
            
            plt.fill_between(range(medium_from, medium_to+1), y_min, y_max, facecolor='red', alpha=0.5)
        
        # updat4e function for animation
        def update_line(num, plotting_data, time, line):
            dat = plotting_data[num,:]
            line.set_data([range(len(dat)), dat])
            time_text.set_text('time = %.1f' % time[num])
            line.set_label('t= 10')
            return line,
        
        line_ani = animation.FuncAnimation(fig, update_line, n_frames, fargs=(plotting_data, time, l), interval=interval, blit=False)
        return line_ani
    
    def monotoneous(self, t_measure, rtol=1e-5):
        vel = self.get_velocities(t_measure)
        tol = rtol*np.abs(vel).max()
        print 'np.abs(vel).max()', np.abs(vel).max()
        print 'tol', tol
        return np.all( np.diff(vel)>= (-tol) )
        
    
    # peak finding routine 
    def find_peaks(self, t_measure):
        """
        Find the peaks of solitary waves. Only reliable for a single wave with no other disturbances.
        
        Parameters
        ----------
        
        t_measure : float
            Time at which to find the peaks.
        """
        self._check_time(t_measure)
        #widths = np.arange(2,7)  # range of widths to check by find_peaks_cwt
        #peak_nodes = find_peaks_cwt(self.get_velocities(t_measure), widths, min_snr=2.0,noise_perc=30.0)
        peak_beads = peakutils.peak.indexes(self.get_velocities(t_measure), thres=0.75, min_dist=7)
        return peak_beads
    
    
    # measure wave velocity
    def measure_wave_velocity(self, t_1, t_2, units='computer', shrink_range=False, max_tries=10):
        """
        
        Function measuring the wave velocity of a solitary wave by calculating the average speed in the time interval [t1,t2]. It is assumed and asserted that all bead radii are equal. This function is only guaranteed to work for a single wave propagating through a uniform medium with no other disturbances.
        
        Returns
        -------
        
        vel : float        
    
        Parameters
        ----------
        
        t_1 : double
            Start time for speed measurement. 
        
        t_2 : double
            Start time for speed measurement.
        
        units : str, optional
            Units in which to return the speed: 'computer' for computer units and 'SI' for SI units. (default='computer')
        
        shrink_range : bool, optional
            If velocity not determinable in given range iteratively try to determine on shrinked range. (default=False)
        
        max_tries : int, optional
            Maximum number of tries for shrinking the range. (default=10)
        
        """
        assert t_1<t_2
        assert units=='computer' or units == 'SI'
        assert np.all(self.sim_chain.R==self.sim_chain.R[0])
        
        # find the peaks at first time
        peak1 = self.find_peaks(t_1)
        
        if len(peak1) > 1:
            warnings.warn('Velocity cannot be determined unambiguously. More than one peak detected at first time.')
            return np.nan
        elif len(peak1) == 0:
            warnings.warn('Velocity cannot be determined. No peaks could be detected at first time.')
            return np.nan
        
        # find the peaks at second time
        peak2 = self.find_peaks(t_2)
        
        if len(peak2) > 1 or len(peak2) == 0 or peak2[0] > self.get_N()-10:
            if shrink_range == True:
                try_ctr = 0
                time_diff = t_2 - t_1
                alpha = 0.5                 # shrinking factor
                while try_ctr < max_tries:
                    # shrink range 
                    t_2 = t_1 + alpha*time_diff
                    peak2 = self.find_peaks(t_2)
                    time_diff = t_2 - t_1
                    try_ctr += 1
                    
                    # check if time_diff is too small
                    if time_diff < 20*self.dt:
                        warnings.warn('Time difference too small to determine wave speed.')
                        return np.nan
                    
                    # check if velocity determinable
                    if len(peak2) > 1 or len(peak2) == 0 or peak2[0] > self.get_N()-10:
                         pass
                    else:
                        # calculate velocity
                        vel = (peak2[0] - peak1[0])/time_diff
                        if units == 'computer':
                            return vel
                        elif units == 'SI':
                            R = self.sim_chain.R[0]
                            beta = self.sim_chain.beta
                            return vel*2.0*R/beta
                
                # if not succeeded by now return warning
                warnings.warn('Velocity cannot be determined unambiguously in given range.')
                return np.nan
            
            else:
                if len(peak2) > 1:
                    warnings.warn('Velocity cannot be determined unambiguously. More than one peak detected at seocond time.')
                    return np.nan
                elif len(peak2) == 0:
                    warnings.warn('Velocity cannot be determined. No peaks could be detected at second time.')
                    return np.nan
                elif peak2[0] > self.get_N()-10:
                    warnings.warn('The second peak is too close to the boundary. Wave could have left the chain.')
                    return np.nan
        else:
            # calculate velocity
            time_diff = t_2 - t_1
            vel = (peak2[0] - peak1[0])/time_diff
            if units == 'computer':
                return vel
            elif units == 'SI':
                R = self.sim_chain.R[0]
                beta = self.sim_chain.beta
                return vel*2.0*R/beta
        
    
    def measure_forces(self, from_time=None, to_time=None, units='computer'):
        """
        
        Function measuring the net forces on the beads by calculating their acceleration.
        
        Returns
        -------
        
        forces : array-like        
    
        Parameters
        ----------
        
        from_time : double
            Start time for speed measurement. 
        
        to_time : double
            Start time for speed measurement.
        
        units : str, optional
            Units in which to return the forces: 'computer' for computer units and 'SI' for SI units. (default='computer')
        
        """
        assert units=='computer' or units == 'SI'
        if from_time == None:
            from_time = self.t0
        if to_time == None:
            to_time = self.t1
        
        from_idx = self._find_index_for_time(from_time)
        to_idx = self._find_index_for_time(to_time)
        
        dt_ind = self.get_dt_per_index()
        #dt_ind = 1.0
        
        N = self.get_N()
        forces = np.zeros((self.data[from_idx:to_idx+1,:].shape[0], N))
        for bead in range(N):
            vel = self.get_bead_velocity(bead, from_time, to_time)
            acc = np.gradient(vel, dt_ind)
            mass = self.sim_chain.m[bead]
            forces[:,bead] = mass*acc
        
        if units=='computer':
            return forces
        elif units=='SI':
            return forces/(self.sim_chain.beta**2)
    
    
    def measure_max_forces(self, from_time=None, to_time=None, units='computer'):
        assert units=='computer' or units == 'SI'
        return np.max( self.measure_forces(from_time, to_time, units=units), axis=1 )
    
    
    def measure_force_at_wall(self, from_time=None, to_time=None, units='computer'):
        assert units=='computer' or units == 'SI'
        if from_time == None:
            from_time = self.t0
        if to_time == None:
            to_time = self.t1
        N = self.get_N()
        vel = self.get_bead_velocity(N-1, from_time, to_time)
        dt_ind = self.get_dt_per_index()
        acc = np.gradient(vel, dt_ind)
        mass = self.sim_chain.m[N-1]
        force = mass*acc
        
        if units=='computer':
            return force
        elif units=='SI':
            return force/(self.sim_chain.beta**2)
    
    
    def measure_wavelength(self, t_measure, use='overlap'):
        """
        
        Function measuring the wavelength of a wave in the granular chain. This is only working reliably for a single wave with no other disturbances.
        
        Returns
        -------
        
        wavelength : int
            Number of beads spanning the wave.
            
    
        Parameters
        ----------
        
        t_measure : double
            Time at which to measure the wavelength.
        
        use : str, optional
            Wave property to use for wavelength measurement (options: 'overlap', 'vel'). (default='overlap')
        
        """
        assert (use=='overlap') or (use=='vel')
        if use=='overlap':
            overlap = self.overlap(t_measure)
            
            # set all overlap entries below 0.1% of the maximum overlap to zero
            overlap[overlap<0.001*np.max(overlap)] = 0.0
            nonzero = np.nonzero(overlap)[0]
            
            # find connected section of chain with nonzero overlap
            consecutives = np.split(nonzero, np.where(np.diff(nonzero) != 1)[0]+1)
            if len(consecutives) != 1:
                warnings.warn('Wavelength could not be determined unambiguously.')
                return np.nan
            else:
                # add 1 since overlap involves two beads each
                wavelength = len(consecutives[0]) + 1
                return wavelength
    
    
    def measure_wavelength_avg(self, from_time, to_time, print_n=False, use='overlap'):
        """
        
        Function measuring the average wavelength of a wave in the granular chain. This is only working reliably for a single wave with no other disturbances.
        
        Returns
        -------
        
        wavelength_avg : float
            Average number of beads spanning the wave.
    
        Parameters
        ----------
        
        from_time : float
            Time at which to start the average wavelength measurement.
        
        to_time : float
            Time at which to end the average wavelength measurement.
        
        print_n : bool, optional
            Print number of wavelength measurements used for average. (default=False)
        
        use : str, optional
            Wave property to use for wavelength measurement (options: 'overlap', 'vel'). (default='overlap')
        
        """
        assert (use=='overlap') or (use=='vel')
        
        from_idx = self._find_index_for_time(from_time)
        to_idx = self._find_index_for_time(to_time)
        
        n_measurements = to_idx - from_idx + 1
        
        if use=='overlap':
            # calculate overlap
            overlap = self.overlap()[from_idx:to_idx+1,:]
            
            # intitialize wavelength storage
            wavelengths = np.zeros(overlap.shape[0])
            
            # counter for NaNs
            nanctr = 0
            for i in range(overlap.shape[0]):
                this_overlap = overlap[i,:]
                
                # set all overlap entries below 0.1% of the maximum overlap to zero
                this_overlap[this_overlap<0.001*np.max(this_overlap)] = 0.0
                
                nonzero = np.nonzero(this_overlap)[0]
                
                #print 'this_overlap', this_overlap
                #print 'nonzero', nonzero
                #print 'this ovelap shape', this_overlap.shape
                #print 'np.diff(nonzero) != 1', np.argwhere(np.diff(nonzero) != 1)
                
                # find connected section of chain with nonzero overlap
                consecutives = np.split(nonzero, np.where(np.diff(nonzero) != 1)[0]+1)
                #print 'consecutives:', consecutives
                
                if len(consecutives) != 1:
                    warnings.warn('Wavelength could not be determined unambiguously.')
                    wavelengths[i] = np.nan
                    nanctr += 1
                else:
                    # add 1 since overlap involves two beads each
                    wavelengths[i] = float( len(consecutives[0]) ) + 1
        
        else:
            # get velocities
            N = self.get_N()
            vel = self.data[from_idx:to_idx+1,N:2*N]
            
            # clean out backscattered beads
            #vel[vel<0] = 0
            
            # intitialize wavelength storage
            wavelengths = np.zeros(vel.shape[0])
            
            # counter for NaNs
            nanctr = 0
            for i in range(vel.shape[0]):
                this_vel = vel[i,:]
                
                # set all vel entries below 0.1% of the maximum vel to zero
                this_vel[this_vel<0.001*np.max(this_vel)] = 0.0
                
                nonzero = np.nonzero(this_vel)[0]
                
                #print 'this_overlap', this_overlap
                #print 'nonzero', nonzero
                #print 'this ovelap shape', this_overlap.shape
                #print 'np.diff(nonzero) != 1', np.argwhere(np.diff(nonzero) != 1)
                
                # find connected section of chain with nonzero overlap
                consecutives = np.split(nonzero, np.where(np.diff(nonzero) != 1)[0]+1)
                #print 'consecutives:', consecutives
                
                if len(consecutives) != 1:
                    warnings.warn('Wavelength could not be determined unambiguously.')
                    wavelengths[i] = np.nan
                    nanctr += 1
                else:
                    # add 1 since overlap involves two beads each
                    wavelengths[i] = float( len(consecutives[0]) )
        
        # print number of measurements
        if print_n:
            print 'successful measurements:', n_measurements - nanctr
            
        if nanctr == n_measurements:
            return np.nan
        else:
            return np.nanmean( wavelengths )
    
    def collapse_wave(self, wavespeed, from_time, to_time, max_datapoints=None):
        from_idx    = self._find_index_for_time(from_time)
        to_idx      = self._find_index_for_time(to_time)
        #print 'from_idx',from_idx
        #print 'to_idx', to_idx
        N           = self.get_N()
        vel         = self.data[from_idx:to_idx+1, N:2*N]
        time        = self.get_time()[from_idx:to_idx+1]
        
        # shrink data
        if (max_datapoints != None) and (len(time)>max_datapoints):
            fac     = np.ceil(vel.shape[0]/max_datapoints)
            vel     = vel[::fac]
            time    = time[::fac]

        
        wave_values = np.zeros(20*vel.shape[0])
        bead_values = np.zeros(20*vel.shape[0])
        value_ctr   = 0
        for i in range(vel.shape[0]):
            this_vel    = vel[i,:]
            
            # set all vel entries below 0.1% of the maximum vel to zero
            this_vel_bound = copy.deepcopy(this_vel)
            this_vel_bound[this_vel<0.001*np.max(this_vel)] = 0.0
            nonzero = np.nonzero(this_vel_bound)[0]
            #print 'nonzero', nonzero
            
            # find connected section of chain with nonzero vel
            consecutives = np.split(nonzero, np.where(np.diff(nonzero) != 1)[0]+1)[0]
            consecutives = np.concatenate(([consecutives.min()-1], consecutives, [consecutives.max()+1]))
            #print 'cons', consecutives
            
            # find wave values at those beads
            this_wave   = this_vel[consecutives]
            wave_values[value_ctr:value_ctr+len(this_wave)] = this_wave
            
            # shift back to from_time
            time_shift = time[i] - time[0]
            bead_shift = time_shift*wavespeed
            this_beads = consecutives - bead_shift
            bead_values[value_ctr:value_ctr+len(this_beads)] = this_beads
            
            assert len(this_wave) == len(this_beads)
            value_ctr += len(this_wave)
        
        # truncate zeros and shift beads to around 0
        wave_values = wave_values[0:value_ctr]
        bead_values = bead_values[0:value_ctr]
        bead_values -= np.mean(bead_values)
        
        # sort bead values
        sort_ind    = np.argsort(bead_values)
        bead_values = bead_values[sort_ind]
        wave_values = wave_values[sort_ind]
        
            
        return (bead_values, wave_values)
    
    def measure_wave_velocity2(self, from_time, to_time, max_datapoints=None, gtol=1e-15, y_cut=0.5):
        speed0      = self.measure_wave_velocity(from_time, to_time, shrink_range=True)
        
        def residual(wavespeed):
            wavespeed = 0.001*wavespeed
            x,y = self.collapse_wave(wavespeed=wavespeed, from_time=from_time, to_time=to_time, max_datapoints=max_datapoints)
            n = self.sim_chain.n[2]
            
            # data selection
            mask    = (y>y_cut*y.max())
            x       = x[mask]
            y       = y[mask]
            
            # model solution function
            def heaviside(x):
                return 1.0 * (x > 0.0)
            def model_sol(x, ampl, lam, x0):
                ampl=1e-8*ampl
                return ampl*np.abs(np.cos(lam*(x-x0)))**(2./(n-2.))*heaviside((x-x0)+np.pi/(2*lam))*heaviside(-(x-x0)+np.pi/(2*lam))
            
            # fit
            popt, pcov  = curve_fit(model_sol, x, y, p0=[7,10,0])
            ampl        = popt[0]
            lam         = popt[1]
            x0          = popt[2]
            fit         = model_sol(x, ampl, lam, x0)
            res         = y - fit
            MSE         = (res**2).mean()
            return 1.0e8*MSE
        
        minim = minimize(residual, tol=1e-20, x0=1000*speed0,options={'gtol': gtol, 'disp': True})
        print minim
        return 0.001*minim['x'][0]
    
    def fit_wave(self, from_time, to_time, fun='sin', max_datapoints=1000, y_cut=0.5, gtol=1e-15, save_plot=False, show_plot=False, foldername='', title=None, figureheight='0.8\\textwidth', figurewidth='\\textwidth', x_min=None, x_max=None, y_min=None, y_max=None):
        n       = self.sim_chain.n[2]
        speed   = self.measure_wave_velocity2(from_time, to_time, max_datapoints=max_datapoints, gtol=gtol)
        x,y     = self.collapse_wave(speed, from_time, to_time, max_datapoints=max_datapoints)
        
        print 'no. datapoints:', len(x)
        
        # data selection
        mask    = (y>y_cut*y.max())
        x_fit   = x[mask]
        y_fit   = y[mask]
        
        # fit function
        def heaviside(x):
            return 1.0 * (x > 0.0)
        def model_sol(x, ampl, lam, x0):
            ampl=1e-8*ampl
            return ampl*np.abs(np.cos((np.pi/lam)*(x-x0)))**(2./(n-2.))*heaviside((x-x0)+np.pi/(2*(np.pi/lam)))*heaviside(-(x-x0)+np.pi/(2*(np.pi/lam)))
        
        if fun == 'sin':
            popt, pcov  = curve_fit(model_sol, x_fit, y_fit, p0=[10,10,0], ftol=1e-10)
        elif fun == 'sech':
            def sech_sol(x, ampl, lam, x0, p):
                ampl=1e-8*ampl
                return ampl/(np.cosh( (np.pi/lam)*(x-x0) )**p)
            popt, pcov  = curve_fit(sech_sol, x_fit, y_fit, p0=[10,10,0,1.1], ftol=1e-10)
            p = popt[3]
            p_var = np.sqrt( pcov[3,3] )
        elif fun == 'gauss':
            def gauss_sol(x, ampl, lam, x0, p):
                ampl=1e-8*ampl
                return ampl*np.exp( - ((x-x0)**2)/(2*lam**2) )**p
            popt, pcov  = curve_fit(gauss_sol, x_fit, y_fit, p0=[10,10,0,1.1], ftol=1e-10)
            p = popt[3]
            p_var = np.sqrt( pcov[3,3] )
            
        ampl        = popt[0]
        lam         = popt[1]
        x0          = popt[2]
        ampl_var    = np.sqrt(pcov[0,0])
        lam_var     = np.sqrt(pcov[1,1])
        
        # residual
        if fun == 'sin':
            fit         = model_sol(x, ampl, lam, x0)
        elif fun == 'sech':
            fit         = sech_sol(x, ampl, lam, x0, p)
        elif fun == 'gauss':
            fit         = gauss_sol(x, ampl, lam, x0, p)
        res         = y - fit
        MSE         = (res**2).mean()
        
        if fun == 'sin':
            fit_fit     = model_sol(x_fit, ampl, lam, x0)
        elif fun == 'sech':
            fit_fit     = sech_sol(x_fit, ampl, lam, x0, p)
        elif fun == 'gauss':
            fit_fit     = gauss_sol(x_fit, ampl, lam, x0, p)
        res_fit     = y_fit - fit_fit
        MSE_fit     = (res_fit**2).mean()
        
        # model
        om_model    = (n-2)/(n) * np.sqrt( (6.0*n)/(n-1) )
        lam_model   = np.pi/om_model
#        model       = model_sol(x, ampl, lam_model, x0)
        model       = model_sol(x, y.max()*1e8, lam_model, x0)
        res_model   = y - model
        MSE_model   = (res_model**2).mean()
        
        
        
        if save_plot:
            # plot
            fig = plt.figure()
            ax = plt.subplot(111)
            plt.plot(x-x0,y,'.',ms=1,label='\\footnotesize sim')
            plt.plot(x-x0,fit,lw=2,label='\\footnotesize fit')
            plt.plot(x-x0,model, lw=2,label='\\footnotesize model')
            if y_max == None: y_max = np.max(y)
            if y_min == None: y_min = 0
            if x_max == None: x_max = np.max(x-x0)
            if x_min == None: x_min = np.min(x-x0)
            ax.set_ylim(ymin=y_min, ymax=y_max)
            ax.set_xlim(xmin=x_min, xmax=x_max)
            ax.set_xlabel("\\footnotesize{bead}")
            ax.set_ylabel("\\footnotesize{$v$ (c.u.)}")
            ax.legend()
            plt.tight_layout()
            if title=='' or title==None:
                pass
            else:
                ax.set_title("\\footnotesize{"+title+"}")
            tikz_save(foldername+'/wave_fit,n='+str(n)+'.tex', figureheight=figureheight, figurewidth=figurewidth)
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
        
        # return dictionary
        return_dict = {}
        return_dict['speed']    = speed
        return_dict['ampl']     = ampl
        return_dict['lam']      = lam
        return_dict['ampl_var'] = ampl_var
        return_dict['lam_var']  = lam_var
        return_dict['MSE']      = MSE
        return_dict['MSE_fit']  = MSE_fit
        return_dict['MSE_model']= MSE_model
        
        try:
            return_dict['p']        = p
            return_dict['p_var']    = p_var
        except:
            pass
        return return_dict
        
        
    def measure_max_bead_kin_energy(self, t_measure, index=False, only_pos=False):
        vel         = self.get_velocities(t_measure)
        if only_pos:
            vel[vel<=0.0] = 0.0
        m           = self.sim_chain.m
        kin_energy  = 0.5*np.multiply( m, vel**2.0 )
        if index:
            return (kin_energy.max(), kin_energy.argmax())
        else:
            return kin_energy.max()
    
    def measure_energy_transfer(self, from_time=None, to_time=None, from_bead=None, to_bead=None):
        if from_time == None: 
            from_time=0
        if to_time == None: 
            to_time=self.t1
        
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.get_N() - 1
        
        assert from_time < to_time
        from_idx    = self._find_index_for_time(from_time)
        to_idx      = self._find_index_for_time(to_time)
        N           = self.get_N()
        assert 0<=from_bead<to_bead<N
        velocities  = self.data[from_idx:to_idx+1, N+from_bead:N+to_bead+1]
        
        # calculate kinetic energy
        max_vel     = np.max(velocities, axis=0)
        masses      = self.sim_chain.m[from_bead:to_bead+1]
        max_kin     = 0.5*masses*max_vel**2.0
        
        # get exponents
        n           = self.sim_chain.n[from_bead:to_bead]
        n_left      = n[:-1]
        n_right     = n[1:]
        
        # divide max vel by max vel of left neighbour
        max_kin_left    = max_kin[:-2]
        max_kin_self    = max_kin[1:-1]
        max_kin_rel     = max_kin_self/max_kin_left
        
        return_array = np.stack([max_kin_rel, n_left, n_right], axis=1)
        return return_array
        
    
    
    def measure_multipulse_energies(self, t_measure, n, cut_per=0.001, from_bead=None, to_bead=None, rev=False):
        """
        
        Function measuring the energies carried by the first n peaks of a multipulse structure. This function assumes that the section of the chain in which the multipulse structure propagates is uniform.
        
        Returns
        -------
        
        energies : array-like
            Energies of first n peaks.
    
        Parameters
        ----------
        
        t_measure : float
            Time of measurement.
        
        n : int
            Number of peaks to detect.
        
        cut_per : float, optional
            Cut-off value for finding the right end of the largest peak relative to its size. (default=0.001)
        
        from_bead : int, optional
            First bead on the left of the strip in which to detect multipulse peaks. (default=None)
        
        to_bead : int, optional
            Last bead on the right of the strip in which to detect multipulse peaks. (default=None)
        
        rev : bool, optional
            Look for peaks with negative velocity, i.e. waves going in the reverse direction to the left. (default=False)
        
        """
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.get_N() - 1
        assert self.get_N() > to_bead > from_bead >= 0
        
        pulse_bounds            = self.find_multipulse_bounds(t_measure, n, cut_per=cut_per, from_bead=from_bead, to_bead=to_bead, rev=rev)
        kinetic_energy_distr    = self.measure_kinetic_energy_distr(t_measure)
        potential_energy_distr  = self.measure_potential_energy_distr(t_measure)
        energies                = np.zeros(len(pulse_bounds)-1)
        
#        print 'pulse bounds:', pulse_bounds
#        print 'kinetic_energy_distr:', kinetic_energy_distr
#        print 'potential_energy_distr:', potential_energy_distr
        
        for i in range( len(pulse_bounds[:-1]) ):
            idx         = pulse_bounds[i]
            idx_next    = pulse_bounds[i+1]
            kin_energy  = np.sum( kinetic_energy_distr[idx:idx_next] )
            pot_energy  = np.sum( potential_energy_distr[idx:idx_next] )
            energies[i] = kin_energy + pot_energy
        
        return energies
    
    
    def measure_multipulse_separation(self, t_measure, n, vel_per=0.001, from_bead=None, to_bead=None, rev=False):
        """
        
        Function measuring whether or not the peaks in the multipulse structure are separated enough using the minimum value of the velocity between two peaks as a measure for separation. If this velocity is smaller than vel_per times the velocity of the leading pulse, two peaks are said to be separated.
        
        Returns
        -------
        
        separation : array-like
            Boolean values for the separation between the peaks.
        
        
        Parameters
        ----------
        
        t_measure : float
            Time of measurement.
        
        n : int
            Number of peaks to detect.
        
        vel_per : float, optional
            Bound for separation between two peaks as percentage of the velocity of the leading peak. (default=0.001)
        
        from_bead : int, optional
            First bead on the left of the strip in which to detect multipulse peaks. (default=None)
        
        to_bead : int, optional
            Last bead on the right of the strip in which to detect multipulse peaks. (default=None)
        
        rev : bool, optional
            Look for peaks with negative velocity, i.e. waves going in the reverse direction to the left. (default=False)
        
        """
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.get_N() - 1
        assert self.get_N() > to_bead > from_bead >= 0
        
        pulse_peaks = self.find_multipulse_peaks(t_measure, n, from_bead=from_bead, to_bead=to_bead, rev=rev)
        velocities = self.get_velocities(t_measure)
        if rev:
            velocities = - velocities
        
        max_vel = np.max( np.abs( velocities[pulse_peaks] ) )
        pulse_bounds = self.find_multipulse_bounds(t_measure, n, from_bead=from_bead, to_bead=to_bead, rev=rev)
        separation = ( velocities[pulse_bounds]<(vel_per*max_vel) )
        return separation
        
    def find_multipulse_bounds(self, t_measure, n, cut_per=0.001, from_bead=None, to_bead=None, rev=False):
        """
        
        Function finding the boundaries of the mulitipulse peaks.
        
        Returns
        -------
        
        pulse_bounds : array-like
            Bounds of first n peaks.
        
        
        Parameters
        ----------
        
        t_measure : float
            Time of measurement.
        
        n : int
            Number of peaks to detect.
        
        cut_per : float, optional
            Cut-off value for finding the right end of the largest peak relative to its size. (default=0.001)
        
        from_bead : int, optional
            First bead on the left of the strip in which to detect multipulse peaks. (default=None)
        
        to_bead : int, optional
            Last bead on the right of the strip in which to detect multipulse peaks. (default=None)
        
        rev : bool, optional
            Look for peaks with negative velocity, i.e. waves going in the reverse direction to the left. (default=False)
        
        """
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.get_N() - 1
        assert self.get_N() > to_bead > from_bead >= 0
        
        pulse_idx = self.find_multipulse_peaks(t_measure, n, from_bead=from_bead, to_bead=to_bead, rev=rev)
        velocities = self.get_velocities(t_measure)
        if rev:
            velocities = - velocities
        
        
        # find minima around pulses to determine pulse extension
        """
        Determine the left end of the smallest peak by finding the minimum of the velocities through the change of sign in the gradient. If it does not work determine the left end of the smallest peak by finding the last bead below the cut-off value velocity.
        """
        pulse_bounds    = np.zeros(len(pulse_idx) + 1, dtype=int)
        vel_grad        = np.gradient(velocities)
        sign_change     = np.argwhere( vel_grad[:pulse_idx[0]]<0 )
        max_value       = np.max( velocities[pulse_idx[-1]] )
        if len(sign_change)!=0:
            pulse_bounds[0] = np.max( sign_change ) + 1
        else:
            pulse_bounds[0] = np.max(np.argwhere( velocities[:pulse_idx[0]]<(cut_per*max_value) ))
        # Select the minimum between adjacent peaks as the boundary
        for i in range( len(pulse_idx[:-1]) ):
            idx = pulse_idx[i]
            idx_next = pulse_idx[i+1]
            pulse_bounds[i+1] = idx + np.argmin( velocities[idx:idx_next+1] )
        
        """
        Determine the right end of the largest peak by finding the first bead below the cut-off value velocity.
        """
        pulse_bounds[-1] = pulse_idx[-1] + np.min(np.argwhere( velocities[pulse_idx[-1]:]<(cut_per*max_value) ))
        
        return pulse_bounds
        
        
    
    def find_multipulse_peaks(self, t_measure, n, cut_per=0.001, from_bead=None, to_bead=None, rev=False):
        """
        
        Function finding the first n peaks of a multipulse structure.
        
        Returns
        -------
        
        peaks : array-like
            Inidces of first n peaks.
    
        Parameters
        ----------
        
        t_measure : float
            Time of measurement.
        
        n : int
            Number of peaks to detect.
        
        cut_per : float, optional
            Cut-off value for accepting maxima relative to the size of the largest peak. (default=0.001)
        
        from_bead : int, optional
            First bead on the left of the strip in which to detect multipulse peaks. (default=None)
        
        to_bead : int, optional
            Last bead on the right of the strip in which to detect multipulse peaks. (default=None)
        
        rev : bool, optional
            Look for peaks with negative velocity, i.e. waves going in the reverse direction to the left. (default=False)
        
        """
        if from_bead == None:
            from_bead = 0
        if to_bead == None:
            to_bead = self.get_N() - 1
        assert self.get_N() > to_bead > from_bead >= 0
        
        velocities = self.get_velocities(t_measure)[from_bead:to_bead+1]
        
        if rev:
            velocities = - velocities
        
        # calculate local maxima
        peaks = local_maxima(velocities)
        
        # apply cut-off
        max_value = np.max( np.abs( velocities[peaks] ) )
        peaks = peaks[ np.abs(velocities[peaks])>(cut_per*max_value)  ]
        
        return peaks[-n:]+from_bead
        
        
    def find_exit_time(self, bead_number=None, exit_threshold=0.0, rev=False):
        """
        
        Function that finds the time for which the wave reaches the end of the chain. This only works if the wave is travelling from left to right and if there are no other disturbances.
        
        Returns
        -------
        
        exit_time : float
        
        Parameters
        ----------
        
        bead_number : int, optional
            Bead number at which to check when the wave arrives. (default=None)
        
        rev : bool, optional
            Look for peaks with negative velocity, i.e. waves going in the reverse direction to the left. (default=False)

        """
        if bead_number == None:
            bead_number = self.N - 1
        assert self.get_N() > bead_number >= 0
        
        # check when bead starts moving
        vel_bead = self.get_bead_velocity(bead_number)
        if rev:
            vel_bead = - vel_bead
        vel_bead[vel_bead <= exit_threshold] = 0.0
        nonzero = np.flatnonzero( vel_bead )
        
        if nonzero.size == 0:
            warnings.warn('The wave did not exit the chain or it could not be determined.')
            return np.nan
        else:
            exit_index = np.min( nonzero )
            time = self.get_time()
            exit_time = time[exit_index]
            return exit_time
    
    def measure_reflected_energy(self, from_bead, to_bead, threshold=0.0):
        """
        
        Function measuring the reflected energy through the energy of the first reflected solitary wave.
        
        Returns
        -------
        
        exit_time : float
        
        Parameters
        ----------
        
        from_bead : int
            Left end of the range in which to detect the reflected solitary wave.
        
        to_bead : int
            Right end of the range in which to detect the reflected solitary wave.
        
        threshold : float, optional
            Velocity threshold for detecting the reflected wave. (default=0.0)

        """
        assert self.get_N() > to_bead > from_bead >= 0
        
        # detect when the reflected wave arrives in the middle of the range
        mid_bead    = int( (to_bead + from_bead)/2 )
        t_mid       = self.find_exit_time(bead_number=mid_bead, exit_threshold=threshold, rev=True)
        print 'mid_bead =', mid_bead
        print 't_mid =', t_mid
        
        if np.isnan(t_mid):
            warnings.warn('No reflected wave could be detected')
            return np.nan
        else:
            # measure peak energy
            energy = self.measure_multipulse_energies(t_mid, n=1, from_bead=from_bead, to_bead=to_bead, rev=True)
            
            return energy
    
    
    # check if a provided time is in the time span of the simulation
    def _check_time(self, t_provided):
        assert self.data is not None
        time = self.get_time()
        if t_provided < time.min() - self.dt:
            warnings.warn('provided time %s smaller than simulation start time' %str(t_provided), stacklevel=2)
        if t_provided > time.max() + self.dt:
            warnings.warn('provided time %s larger than simulation end time' %str(t_provided), stacklevel=2)
    
    def _find_index_for_time(self, t_provided):
        self._check_time(t_provided)
        time = self.get_time()
        idx = (np.abs(time - t_provided)).argmin()
        return idx
    


# other functions 
def vel_force_function(sigma, Y, m, R, Fm):
    theta = 3*(1-sigma**2)/(4*Y)
    C = np.sqrt( (2*R**3)/(m*theta) )
    return C*(np.sqrt(0.8))*( 2*theta*Fm/(R**2) )**(1.0/6.0)


def local_maxima(a):
    a_left = a[:-2]
    a_right = a[2:]
    a_max = np.maximum(a_left, a_right)
    a = a[1:-1]
    comp = a>a_max
    maxima = np.swapaxes( np.argwhere( comp == True) + 1, 0, 1 )[0]
    return maxima


def local_maxima2(array, min_distance = 1, periodic=False, edges_allowed=True): 
    """Find all local maxima of the array, separated by at least min_distance."""
    array = np.asarray(array)
    cval = 0 
    
    if periodic: 
            mode = 'wrap' 
    elif edges_allowed: 
            mode = 'nearest' 
    else: 
            mode = 'constant' 
    cval = array.max()+1 
    max_points = array == ndimage.maximum_filter(array, 1+2*min_distance, mode=mode, cval=cval) 
    
    return [indices[max_points] for indices in np.indices(array.shape)][0]

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

SPINE_COLOR = 'gray'
def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax



    
    
    
        
                
        








