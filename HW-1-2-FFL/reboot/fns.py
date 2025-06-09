import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
def f(u, K, reg, H):
    if reg == 'act':
        return (u / K) ** H / (1 + (u / K) ** H)
    elif reg == 'rep':
        return 1 / (1 + (u / K) ** H)
    else:
        raise ValueError('Regulation unspecified')

def fc(u, Ku, regU, v, Kv, H):
    if regU == 'act':
        return (u / Ku) ** H / (1 + (u / Ku) ** H + (v / Kv) ** H)
    elif regU == 'rep':
        return 1 / (1 + (u / Ku) ** H + (v / Kv) ** H)
    else:
        raise ValueError('Regulation unspecified')

def Gate(gate, u, Ku, regU, v, Kv, regV, H):
    if gate == 'AND':
        return f(u, Ku, regU, H) * f(v, Kv, regV, H)
    elif gate == 'OR':
        return fc(u, Ku, regU, v, Kv, H) + fc(v, Kv, regV, u, Ku, H)
    else:
        raise ValueError('Gate must be AND or OR')

def pulse(t, start, width):
    return 1.0 if (start <= t <= start + width) else 0.0

def impulse(Si, i):
    return i if (Si==1) else 0.0 
def sys(t, Q, prm):
    y, z = Q
    Sy = pulse(t, prm['ystart'], prm['ywidth'])
    Sx = pulse(t, prm['start'], prm['width'])
    x = impulse(Sx, Sx)
    y = impulse(Sy, y)
    dydt = prm['By'] + prm['bt_y'] * f(x, prm['Kxy'], prm['regXY'], prm['H']) - prm['alp_y'] * y
    dzdt = prm['Bz'] + prm['bt_z'] * Gate(prm['gate'], x, prm['Kxz'], prm['regXZ'], y, prm['Kyz'], prm['regYZ'], prm['H']) - prm['alp_z'] * z
    return [dydt, dzdt]

def SolveAndPlot(prm, axis):
    t_eval = np.linspace(prm['tmin'], prm['tmax'], prm['N'])
    Q0 = prm['Q0']
    sol = solve_ivp(
        lambda t, Q: sys(t, Q, prm),
        (prm['tmin'], prm['tmax']),
        Q0,
        t_eval=t_eval, method='DOP853'
    )
    x = np.zeros_like(t_eval)
    x = np.where((t_eval >= prm['start']) & (t_eval <= prm['start'] + prm['width']), 1.0, 0.0)
    axis.plot(t_eval, x, color='black', alpha=.4)
    axis.plot(sol.t, sol.y[1], label='z(t)')
    # axis.plot(sol.t, sol.y[0], label='y(t)')
    axis.set_xlabel('Time($a.u$)')
    axis.set_ylabel('Z')
    axis.set_title(prm['header'])