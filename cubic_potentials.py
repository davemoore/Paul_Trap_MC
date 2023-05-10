## helper functions to calcuate potentials fields in a cube analytically
import numpy as np

epsilon_0 = 8.85e-12 ## F/m
e_to_Coul = 1.6e-19 ## Coulombs
cm_to_m = 1e-2 ## convert cm to SI
k = 1/(4*np.pi*epsilon_0) * e_to_Coul * 1/cm_to_m

def V(x,y,z, a=1.0, V_0=1.0, n_max=20, m_max=20, q_sphere = 0, sphere_rad = 1.5e-4):
    """ Function to calculate the potential for a cube with two sides (in the x direction) at potential V_0
            x,y,z -- coordinates to evaluate (can be arrays, units must be cm!)
            a -- side length (units must be cm!)
            V_0 -- potential on faces at x = +/- a/2
            n_max, m_max -- max terms to sum the series over in y and z directions
            q_sphere -- charge of sphere (in electrons)
            sphere_rad -- sphere radius (units must be_cm!)
            
        returns an array of the size of x,y,z
    """

    ## potential from sphere (if desired)
    r = np.sqrt(x**2 + y**2 + z**2) * cm_to_m
    Vsphere = 1/(4*np.pi*epsilon_0) * e_to_Coul * q_sphere/r
    Vsphere[r <= sphere_rad] = 1/(4*np.pi*epsilon_0) * q_sphere/sphere_rad

    prefac = 16*V_0/np.pi**2
    Vout = np.zeros_like(x)
    for n in range(1,n_max,2): ## sum over odd integers from 1 to n_max
        for m in range(1,m_max,2): ## sum over odd integers from 1 to m_max
            smn = np.sqrt(n**2 + m**2)
            Vout += 1/(n*m) * np.sin(n*np.pi*(y/a + 0.5)) * np.sin(m*np.pi*(z/a + 0.5)) * np.cosh(np.pi*smn*x/a) / np.sinh(np.pi*smn/2)
    
    return prefac*Vout + Vsphere

def E(x,y,z, a=1.0, V_0=1.0, n_max=20, m_max=20, q_sphere = 0, sphere_rad = 1.5e-4):
    """ Function to calculate the electric field for a cube with two sides (in the x direction) at potential V_0
            x,y,z -- coordinates to evaluate (can be arrays)
            a -- side length
            V_0 -- potential on faces at x = +/- a/2
            n_max, m_max -- max terms to sum the series over in y and z directions
            q_sphere -- charge of sphere (in electrons)
            sphere_rad -- sphere radius (units must be_cm!)

        returns 3 arrays Ex,Ey,Ez of the size of x,y,z in V/cm!
    """

    prefac = -16*V_0/(a * np.pi)
    Ex,Ey,Ez = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    
    ## field from sphere (if desired)
    r = np.sqrt(x**2 + y**2 + z**2)
    Esphere = k * q_sphere/r**2
    Esphere[r <= sphere_rad] = 0
    Es_x, Es_y, Es_z = Esphere*x/r, Esphere*y/r, Esphere*z/r 

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

    return prefac*Ex + Es_x, prefac*Ey + Es_y, prefac*Ez + Es_z