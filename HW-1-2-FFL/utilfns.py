import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def hill(u, K, H, reg):
    if reg == 'act' or reg == 1:
        return (u/K)**H/(1+(u/K)**H)
    if reg == 'rep' or reg == 0:
        return 1/(1+(u/K)**H)

def hill2(u, Ku, v, Kv, H, reg):
    if reg==1 or reg=='act':
        return (u/Ku)**H/(1+ (u/Ku)**H + (v/Kv)**H)
    if reg==0 or reg=='rep':
        return 1/(1+ (u/Ku)**H + (v/Kv)**H)

def Gate(x1, y1, prm):
    if prm['gate'] == 'AND':
        AND = hill(x1, prm['Kxz'], prm['H'], prm['regXZ']) * hill(y1, prm['Kyz'], prm['H'], prm['regYZ'])
        return AND
    if prm['gate'] == 'OR':
        OR = hill2(x1, prm['Kxz'], y1, prm['Kyz'], prm['H'], prm['regXZ']) + hill2(y1, prm['Kyz'], x1, prm['Kxz'], prm['H'], prm['regYZ'])
        return OR


def pulse(total_time, pulse_width, dt=.005, start_time=.5):
    t = np.arange(0, total_time, dt)
    pulse = np.zeros_like(t)
    
    if start_time is None:
        start_time = (total_time - pulse_width) / 2  
    end_time = start_time + pulse_width
    
    pulse[(t >= start_time) & (t < end_time)] = 1
    return t, pulse


# def ODEsystem(Q, t):
#     y, z = Q
    
#     xStar2 = np.interp(t, t_grid, xStar)
#     yStar2 = np.interp(t, t_grid, yStar)
    
#     ydot = prm['By'] + prm['Cy'] * hill(xStar2, prm['Kxy'], prm['H'], prm['regX']) - prm['Ay'] * y
#     zdot = prm['Bz'] + prm['Cz'] * Gate(xStar2, yStar2, prm) - prm['Az'] * z
    
#     return [ydot, zdot]


def subplotGenr(prm, Q0, axis, show_pulse=True):  
    t_grid, yStar = pulse(prm['tTotal'], prm['yWidth'])  # Sy
    _, xStar = pulse(prm['tTotal'], prm['xWidth'])       # Sx
    
    def ODEsystem(Q, t):
        y, z = Q
        xStar2 = np.interp(t, t_grid, xStar)
        yStar2 = np.interp(t, t_grid, yStar)
        ydot = prm['By'] + prm['Cy'] * hill(xStar2, prm['Kxy'], prm['H'], prm['regXY']) - prm['Ay'] * y
        zdot = prm['Bz'] + prm['Cz'] * Gate(xStar2, yStar2, prm) - prm['Az'] * z
        return [ydot, zdot]

    Qsol = odeint(ODEsystem, Q0, t_grid)
    axis.plot(t_grid, Qsol[:, 1], '-', markersize=3)

    in_pulse = False
    start = 0
    if show_pulse==True:
        for i in range(len(xStar)):
            if xStar[i] == 1 and not in_pulse:
                start = t_grid[i]
                in_pulse = True
            elif xStar[i] == 0 and in_pulse:
                end = t_grid[i]
                axis.axvspan(start, end, color='yellow', alpha=0.1)
                in_pulse = False

    axis.set_xlabel('time (a.u)')
    axis.set_ylabel('$Z$')
    
    axis.set_title(prm['header'])

