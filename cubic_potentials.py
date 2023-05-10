## helper functions to calcuate potentials fields in a cube analytically
import numpy as np

def V(x,y,z, a=1.0, V_0=1.0, n_max=20, m_max=20):
    """ Function to calculate the potential for a cube with two sides (in the x direction) at potential V_0
            x,y,z -- coordinates to evaluate (can be arrays)
            a -- side length
            V_0 -- potential on faces at x = +/- a/2
            n_max, m_max -- max terms to sum the series over in y and z directions

        returns an array of the size of x,y,z
    """

    prefac = 16*V_0/np.pi**2
    Vout = np.zeros_like(x)
    for n in range(1,n_max,2): ## sum over odd integers from 1 to n_max
        for m in range(1,m_max,2): ## sum over odd integers from 1 to m_max
            smn = np.sqrt(n**2 + m**2)
            Vout += 1/(n*m) * np.sin(n*np.pi*(y/a + 0.5)) * np.sin(m*np.pi*(z/a + 0.5)) * np.cosh(np.pi*smn*x/a) / np.sinh(np.pi*smn/2)
    
    return prefac*Vout

def E(x,y,z, a=1.0, V_0=1.0, n_max=20, m_max=20):
    """ Function to calculate the electric field for a cube with two sides (in the x direction) at potential V_0
            x,y,z -- coordinates to evaluate (can be arrays)
            a -- side length
            V_0 -- potential on faces at x = +/- a/2
            n_max, m_max -- max terms to sum the series over in y and z directions

        returns 3 arrays Ex,Ey,Ez of the size of x,y,z
    """

    prefac = -16*V_0/(a * np.pi)
    Ex,Ey,Ez = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    
    for n in range(1,n_max,2): ## sum over odd integers from 1 to n_max
        for m in range(1,m_max,2): ## sum over odd integers from 1 to m_max
            
            ## precalculate common factors to save time
            smn = np.sqrt(n**2 + m**2)
            sy, cy = np.sin(n*np.pi*(y/a + 0.5)), np.cos(n*np.pi*(y/a + 0.5))
            sz, cz = np.sin(m*np.pi*(z/a + 0.5)), np.cos(m*np.pi*(z/a + 0.5))
            sx, cx = np.sinh(np.pi*smn*x/a), np.cosh(np.pi*smn*x/a)
            const = np.sinh(np.pi*smn/2)

            Ex += smn/(n*m) * sy * sz * sx / const
            Ey += 1/m * cy * sz * cx / const
            Ez += 1/n * sy * cz * cx / const

    return prefac*Ex, prefac*Ey, prefac*Ez