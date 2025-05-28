import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def pulse(total_time, pulse_width, dt, start_time):
    t = np.arange(0, total_time, dt)
    pulse = np.zeros_like(t)
    
    if start_time is None:
        start_time = (total_time - pulse_width) / 2  
    end_time = start_time + pulse_width
    
    pulse[(t >= start_time) & (t < end_time)] = 1
    return t, pulse

def hill(u, K, H, reg):
    if reg == 'act':
        return ((u/K)**H)/(1+(u/K)**H)
    if reg == 'rep':
        return 1/(1+(u/K)**H)

def hill2(u, Ku, v, Kv, H, reg):
    if reg=='act':
        return ((u/Ku)**H)/(1+ (u/Ku)**H + (v/Kv)**H)
    if reg=='rep':
        return 1/(1+ (u/Ku)**H + (v/Kv)**H)

def Gate(u, Ku, regu, v, Kv, regv, prm):
    if prm['gate'] == 'AND':
        AND = hill(u, Ku, prm['H'], regu) * hill(v, Kv, prm['H'], regv)
        return AND
    if prm['gate'] == 'OR':
        OR = hill2(u, Ku, v, Kv, prm['H'], regu) + hill2(v, Kv, u, Ku, prm['H'], regv)
        return OR


def subplotGenr(prm, Q0, axis, show_pulse=True,):  
    _, yStar = pulse(prm['tTotal'], prm['yWidth'], prm['dt'], prm['start_time'])  # Sy
    t_grid, xStar = pulse(prm['tTotal'], prm['xWidth'], prm['dt'], prm['start_time'])       # Sx    
    
    def ODEsystem(Q, t):
        y, z = Q
        xStar2 = np.interp(t, t_grid, xStar)
        ydot = prm['By'] + prm['Cy'] * hill(xStar2, prm['Kxy'], prm['H'], prm['regXY']) - prm['Ay'] * y
        zdot = prm['Bz'] + prm['Cz'] * Gate(xStar2, prm['Kxz'], prm['regXZ'], y,prm['Kyz'], prm['regYZ'], prm) - prm['Az'] * z
        return [ydot, zdot]

    Qsol = odeint(ODEsystem, Q0, t_grid)
    axis.plot(t_grid, Qsol[:, 1], '-', markersize=1)
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

