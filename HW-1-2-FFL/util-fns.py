import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ivp

def f(u, K, H, reg):
    if reg == act or reg == 1:
        return ((u/K)**H)/(1+(u/K)**H)
    if reg == rep or reg == 0:
        return 1/(1+(u/K)**H)

def Gate(gatename, x1, Kxz, y1, Kyz, H, regX, regY):
    if gatename == 'AND':
        return f(x1, Kxz, H, regX)*f(y1, Kyz, H, regY)
    
    if gatename == 'OR':
        def fc(u, Ku, v, Kv, reg):
            if reg==1 or reg==act:
                return (u/Ku)**H/(1+ (u/Ku)**H + (v/Kv)**H)
            if reg==0 or reg==rep:
                return 1/(1+ (u/Ku)**H + (v/Kv)**H)
            
        OR = fc(x1, Kxz, y1, Kyz, H, regX, regY) + fc(y1, Kyz, x1, Kxz, H, regX, regY)
        return OR


def system(t, q, gatename):
    x, y = q.
    ydot = By + Cy*f(x1, Kxy) - Ay*y1
    zdot = Bz + Cz*Gate(gatename,x1, Kxz, y1, Kyz, H, regX, regY) - Az*z1
 
    return [ydot, zdot]



def FFL():
    t_eval = np.linspace(t_span[0], t_end[1], timepoints)
    sol = solve_ivp(fun=lambda t, q: system(t, q, gatename), 
                    t_span=t_span,initcond, t_eval=t_eval, 
                    method='RK45')
    return sol.t, sol.y

