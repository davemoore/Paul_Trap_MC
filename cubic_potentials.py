## helper functions to calcuate potentials fields in a cube analytically
import numpy as np
import matplotlib.pyplot as plt

epsilon_0 = 8.85e-12 ## F/m
e_to_Coul = 1.6e-19 ## Coulombs
cm_to_m = 1e-2 ## convert cm to SI
k = 1/(4*np.pi*epsilon_0) * e_to_Coul * 1/cm_to_m

def V(x,y,z, a=1.0, V_0=1.0, n_max=20, m_max=20, q_ion = 2, q_sphere = 0, sphere_rad = 1.5e-4):
    """ Function to calculate the potential for a cube with two sides (in the x direction) at potential V_0
            x,y,z -- coordinates to evaluate (can be arrays, units must be cm!)
            a -- side length (units must be cm!)
            V_0 -- potential on faces at x = +/- a/2
            n_max, m_max -- max terms to sum the series over in y and z directions
            q_sphere -- charge of sphere (in electrons)
            q_ion -- charge of ion (in electrons)
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

def E(x,y,z, a=1.0, V_0=1.0, n_max=20, m_max=20, q_ion = 2, q_sphere = 0, sphere_rad = 1.5e-4):
    """ Function to calculate the electric field for a cube with two sides (in the x direction) at potential V_0
            x,y,z -- coordinates to evaluate (can be arrays)
            a -- side length
            V_0 -- potential on faces at x = +/- a/2
            n_max, m_max -- max terms to sum the series over in y and z directions
            q_sphere -- charge of sphere (in electrons)
            q_ion -- charge of ion (in electrons)
            sphere_rad -- sphere radius (units must be_cm!)

        returns 3 arrays Ex,Ey,Ez of the size of x,y,z in V/cm!
    """

    prefac = -16*V_0/(a * np.pi)
    Ex,Ey,Ez = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    
    ## field from sphere (if desired)
    r = np.sqrt(x**2 + y**2 + z**2)
    Esphere = k * q_sphere/r**2
    Esphere[r<=sphere_rad] = 0 ## no field inside
    Es_x, Es_y, Es_z = Esphere*x/r, Esphere*y/r, Esphere*z/r 

    for n in range(1,n_max,2): ## sum over odd integers from 1 to n_max
        for m in range(1,m_max,2): ## sum over odd integers from 1 to m_max
            
            ## precalculate common factors to save time
            smn = np.sqrt(n**2 + m**2)
            sy, cy = np.sin(n*np.pi*(y/a + 0.5)), np.cos(n*np.pi*(y/a + 0.5))
            sz, cz = np.sin(m*np.pi*(z/a + 0.5)), np.cos(m*np.pi*(z/a + 0.5))
            sx, cx = np.sinh(np.pi*smn*x/a), np.cosh(np.pi*smn*x/a)
            const = np.sinh(np.pi*smn/2)

            Ex = Ex + smn/(1.0*n*m) * sy * sz * sx / const
            Ey = Ey + 1.0/m * cy * sz * cx / const
            Ez = Ez + 1.0/n * sy * cz * cx / const

    return prefac*Ex + Es_x, prefac*Ey + Es_y, prefac*Ez + Es_z

def track_particle(x0, mu, fAC, VAC, q_sphere, sphere_rad, max_t=0.1, side_length=5):
    ''' Track a particle through the E-field
        x0 -- initial position (x,y,z) in cm
        mu -- mobility in cm^2/(s*V)
        fAC -- frequency of AC field in Hz
        max_t -- maximum time in s
        VAC -- voltage amplitude in V
        side_length -- cube side length in cm
        q_sphere -- charge of sphere (in electrons)
        sphere_rad -- sphere radius (units must be_cm!)

        returns trajectory [x,y,z,t]
    '''
    
    MAX_STEPS = int(max_t*fAC*100) ## maximum steps to take ever
    high_res_rad = 1e-2 ## cm, when in the radius step by at most 0.5 um
    MAX_STEP_SIZE = 0.5e-4 ## cm, step by at most 0.5 um

    tstep = 0.1/fAC ## time step as a fraction of sample time (10x)
    curr_t = 0
    delta = np.random.rand()*2*np.pi

    trajectory = []
    trajectory.append([x0[0], x0[1], x0[2], curr_t]) # put initial position in trajectory
    print("Working on %d points in trajectory:"%MAX_STEPS)
    for j in range(MAX_STEPS):

        if(j%10000 == 0): print(j)

        curr_volt = VAC*np.sin(2*np.pi*fAC*curr_t + delta) ## voltage at this time
        curr_pos = np.array(trajectory[-1][0:3]) ## last position

        ex, ey, ez = E(curr_pos[0:1], curr_pos[1:2], curr_pos[2:], a=side_length, 
                       V_0=curr_volt, q_sphere = q_sphere, sphere_rad = sphere_rad)
        
        if(np.sqrt(np.sum(curr_pos**2)) > high_res_rad):
            ctm = tstep*mu
            dx, dy, dz = ctm*ex[0], ctm*ey[0], ctm*ez[0]
            curr_t += tstep
        else:
            ## if close to the sphere take an adaptive step that moves by a max size
            enorm = np.sqrt(ex[0]**2 + ey[0]**2 + ez[0]**2)/MAX_STEP_SIZE
            dx, dy, dz = ex[0]/enorm, ey[0]/enorm, ez[0]/enorm
            variable_step = 1/(enorm * mu)
            curr_t += variable_step
    
        updated_pos = [curr_pos[0]+dx, curr_pos[1]+dy, curr_pos[2]+dz]
        trajectory.append([updated_pos[0], updated_pos[1], updated_pos[2], curr_t])

        ## break if we are in the sphere
        if(np.sqrt(np.sum(np.array(updated_pos)**2)) <= sphere_rad):
            break

    return np.array(trajectory)

def plot_sphere(ax, r, c, alph=0.5):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r*np.outer(np.cos(u), np.sin(v))
    y = r*np.outer(np.sin(u), np.sin(v))
    z = r*np.outer(np.ones(np.size(u)), np.cos(v))

    # plot sphere with transparency
    ax.plot_surface(x, y, z, alpha=alph, color=c)

def plot3Dtraj(traj, sphere_rad = 1.5e-4, title="", nmax = -1, zoom_range=5e-3):

    ### plot the trajoectory in 3D
    if(nmax < 0):
        nmax = len(traj[:,0])

    # Create a 3D plot
    fig = plt.figure(figsize=(12,9), facecolor='white', layout='constrained')

    subfigs = fig.subfigures(2, 1, hspace=0.07, height_ratios=[2.5, 1])

    ax3d = subfigs[0].add_subplot(1,2,1, projection='3d')
    ax3d_in = subfigs[0].add_subplot(1,2,2, projection='3d')
    ax2d = subfigs[1].subplots(1,3)

    ######## first the 3d plot zoomed out ##################
    cvec = traj[:nmax,3] #/np.max(traj[:,3])
    cmap = plt.get_cmap("jet")
    cb=ax3d.scatter3D(traj[:nmax,0], traj[:nmax,1], traj[:nmax,2], c=cvec, cmap=cmap, s=1, ls='--')
    ax3d.plot(traj[:nmax,0], traj[:nmax,1], traj[:nmax,2], 'k', lw=1, alpha=0.4)

    cbar = plt.colorbar(cb)
    cbar.set_label('Time [s]')

    # Add labels and title
    ax3d.set_xlabel('X [cm]')
    ax3d.set_ylabel('Y [cm]')
    ax3d.set_zlabel('Z [cm]')
    ax3d.set_xlim(-1,1)
    ax3d.set_ylim(-1,1)
    ax3d.set_zlim(-1,1)
    ax3d.set_title(title)

    ### now 3d plot zoomed in to end ###############

    ## find the index where we reach within 50 um of trajectory end
    rel_traj = traj[:,:3]-traj[-1,:3]
    nsamples = np.where(np.sqrt(np.sum(rel_traj**2, axis=1))<zoom_range)[0][0]

    cvec = traj[nsamples:,3]
    cb=ax3d_in.scatter3D(traj[nsamples:,0], traj[nsamples:,1], traj[nsamples:,2], c=cvec, cmap=cmap, s=1, ls='--')
    ax3d_in.plot(traj[nsamples:,0], traj[nsamples:,1], traj[nsamples:,2], 'k', lw=1, alpha=0.4)

    cbar = plt.colorbar(cb)
    cbar.set_label('Time [s]')

    plot_sphere(ax3d_in, sphere_rad, 'gray', alph=0.5)

    # Add labels and title
    ax3d_in.set_xlabel('X [cm]')
    ax3d_in.set_ylabel('Y [cm]')
    ax3d_in.set_zlabel('Z [cm]')
    ax_extent = np.max((np.ptp(traj[nsamples:,0]), np.ptp(traj[nsamples:,1]), np.ptp(traj[nsamples:,2])))
    cl,ch = np.median(traj[nsamples:,0])-ax_extent/2, np.median(traj[nsamples:,0])+ax_extent/2
    ax3d_in.set_xlim(cl,ch)
    cl,ch = np.median(traj[nsamples:,1])-ax_extent/2, np.median(traj[nsamples:,1])+ax_extent/2
    ax3d_in.set_ylim(cl,ch)
    cl,ch = np.median(traj[nsamples:,2])-ax_extent/2, np.median(traj[nsamples:,2])+ax_extent/2
    ax3d_in.set_zlim(cl,ch)
    ax3d_in.set_title("Zoom to end of trajectory")