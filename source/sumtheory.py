# Global variables for calculation and plotting
plotfig = True
figtype = '.eps'
fontsize = 18
offaxis  = True
#offaxis  = False
# Initial Lorentz factor
G0       = 10
# Initial Jet opening angle in radians
theta0   = 0.08#3.03*np.pi/180.#0.05
# Off-axis viewing angle, only used if offaxis = True
thetaobs = 0.35
Omega0   = 4.*pi*(sin(0.5*theta0)**2.)

from physparams import Physparams
params = Physparams(\
        k     = 0,
        p     = 2.1,
        z     = 1.,
        e_e   = 0.1,
        e_b   = 0.01,
        n_0   = 1.,
        Astar = 0.1,
        E52   = 1,
        t_jet = 1.0,
        A_B   = 0.0,
        zeta  = 1.0,
        applyIC = False)

params_Mooley = Physparams(\
        k     = 0,
        p     = 2.16,
        z     = 0.01,
        e_e   = 0.1,
        e_b   = 1e-4,
        n_0   = 6e-4,
        Astar = 0.1,
        E52   = 1e-2*4*np.pi/Omega0,
        t_jet = 1.0,
        A_B   = 0.0,
        zeta  = 1.0,
        applyIC = False)

params_SGRB = Physparams(\
        k     = 0,
        p     = 2.1,
        z     = 0.012,
        e_e   = 0.1,
        e_b   = 1e-2,
        n_0   = 1e-3,
        Astar = 0.1,
        E52   = 0.1,
        t_jet = 1.0,
        A_B   = 0.0,
        zeta  = 1.0,
        applyIC = False)

params_130606A = Physparams(\
        k     = 0,
        p     = 2.079,
        z     = 5.913,
        e_e   = 0.106,
        e_b   = 0.894,
        n_0   = 71.19,
        Astar = 0.1,
        E52   = 43.79,
        t_jet = 0.277,
        A_B   = 0.0,
        zeta  = 1.0,
        applyIC = False)

params = params_Mooley

# Set some variables for the hydrodynamic calculation
k      = params.k
E52    = params.E52
z      = params.z
if (k == 0):
    profile = 'ISM'
    A       = params.n_0
    rmin    = 1e-7
    rmax    = 1e3
elif (k == 2):
    profile = 'wind'
    A       = params.Astar
    rmin    = 1e-20
    rmax    = 1e15

# RUN THE HYDRO CALCULATION
execfile("anly.py")
data = calcall(k=k,E52=E52,A=A,G0=G0,theta0=theta0,thetaobs=thetaobs,rmin=rmin,rmax=rmax,z=z,N=1000,physical=True,modeltype='DL')

# Assign results to convenience variables
u     = data['upeak']
theta = data['theta']
Gamma = data['gamma']
R = data['r']
tlab = data['t']
if (offaxis):
    tobs = data['tobsoffaxis']
else:
    tobs  = data['tobs']

# Set up variables for diagnostics against Granot and Sari (2002)
execfile("diagnostics.py")
tjet, tnr, tsph, tjet_oldtheory = calcdiagnostics(k)
params.t_jet = tjet_oldtheory
params.updatejetparams()
tjet_theory = tjet
tnr_theory  = tnr
tsph_theory = tsph
