import matplotlib.pyplot as plt
import numpy as np
from time import time
from statistics import mean
import timesynth as ts
from IPython.display import clear_output
from pde_testing.generate_database import diffusion_1D, wave_1D
from scipy.interpolate import interp1d
from metrics import accuracy

def smoothen(data):
    data[0] = 0
    data[-1] = 0
    size = len(data)
    middle = int((size-1)/2)
    for steps in range(15):
        for i in range(1, middle):
            data[i] = data[i-1]*alpha + data[i]*(1-alpha)
            data[-(i+1)] = data[-i]*alpha + data[-(i+1)]*(1-alpha)
        data[middle] = (data[middle-1]+data[middle+1])/2

def format_data(u_initial, u_final, length):
    if len(u_final) < length:
        xnew = np.linspace(0, 1, length)
        f_final = interp1d(np.linspace(0,1,len(u_final)), u_final, kind='cubic')
        f_initial = interp1d(np.linspace(0,1,len(u_initial)), u_initial, kind='cubic')
        u_final = f_final(xnew)
        u_initial = f_initial(xnew)
        return np.array([u_initial, u_final])
    else:
        return np.array([u_initial, u_final])

def random_fourier_ts(X=np.linspace(0,1,101), T=2, N=10):

    C = [np.random.logistic(loc=0,scale=0.1)*1j + np.random.logistic(loc=0,scale=0.1) for index in range(21)]
    u = Fourier_serie(X, C, N, T)
    return u

def get_data(cfact=[1], T=[1], u0_param=[1]):

    c_diff = [min(1, c/(np.pi**2)) for c in cfact]
    dt_diff = [0.01 for _ in cfact]
    dx_diff = [np.sqrt(2*dt*c)*2 for c,dt in zip(c_diff, dt_diff)]

    c_wave = [min(1, 0.01*cf) for cf in cfact]
    dt_wave = [0.001 for _ in cfact]
    dx_wave = [np.sqrt(c*dt) for c,dt in zip(c_wave, dt_wave)]

    max_len = max([int(1/dt_d)+1 for dt_d in dt_diff] + [int(1/dt_w)+1 for dt_w in dt_wave])

    nbr_sol = len(u0_param)*len(cfact)*len(T)*2
    solutions = np.zeros(shape=(nbr_sol, max_len))
    labels = np.zeros(nbr_sol)
    n = 1
    index = 0

    for u0_par in u0_param:
        clear_output()
        print('{}/{}'.format(n, len(u0_param)))

        u_0 = lambda x : np.exp(-100*u0_par*(x-0.5)**2)
        v_0 = lambda x : 0

        for c_d, dt_d, dx_d, c_w, dt_w, dx_w in zip(c_diff, dt_diff, dx_diff, c_wave, dt_wave, dx_wave):
            for t in T:
                X_diff, u_final_diff = diffusion_1D(dx_d, dt_d, c_d, t, u_0)
                u_initial_diff = [u_0(x) for x in X_diff]
                X_wave, u_final_wave = wave_1D(dx_w, dt_w, c_w, t, u_0, v_0)
                u_initial_wave = [u_0(x) for x in X_wave]
                waves = format_data(u_initial_wave, u_final_wave, max_len)
                diffs = format_data(u_initial_diff, u_final_diff, max_len)

                solutions[index] = diffs[1, :]
                solutions[index+1] = waves[1, :]
                labels[index] = 0
                labels[index+1] = 1
                index += 2
        n+=1
    return solutions, labels
