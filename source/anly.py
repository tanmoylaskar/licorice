'''Define numerical functions for Duffell & Laskar (2018) and Laskar & Duffell (2019), 
   henceforth DL18 and LD19, respectively'''

from numpy import log, pi, arange, sqrt, sin, cos, arctan

def Pk(k):
    """$P_k(k)$, derived from fitting simulations (DL18, section 2.3)"""
    if (k == 0):
        return 4.0
    if (k == 2):
        return 2.0
    else:
        raise ValueError("k must be 0 or 2")

def Qk(k):
    """"$Q_k(k)$, derived from fitting simulations (DL18, section 2.3)"""
    if (k == 0):
        return 2.5
    if (k == 2):
        return 1.6
    else:
        raise ValueError("k must be 0 or 2")

def u_from_theta(theta, th0, p, q):
    """Calculate 4-velocity from jet angle (DL18, eq 27)"""
    u1 = 1./theta/(q + p*log(theta/th0))
    u0 = 2./pi/(q + p*log(0.5*pi/th0))
    return u1-u0

def dudtheta(theta, th0, p, q):
    """Derivative of 4-velocity with $\theta$ (derived from DL18, eq 27)"""
    denom = q+p*log(theta/th0)
    return -1./theta/theta/denom*(1.+p/denom)

def get_theta(th0, u, k):
    """Newton-Raphson calculation to determine theta from u (from DL 18, eq 27)"""
    p, q = Pk(k), Qk(k)
    u0   = 2./pi/(q+p*log(0.5*pi/th0))
    if (q*(u+u0)*th0 > 1.0):
        return th0
    theta = 1./q/(u+u0)
    theta = 1./(u+u0)/(q + p*log(theta/th0))

    def iterate(u, theta, th0, p, q):
        """Newton-Raphson iteration step"""
        du        = u_from_theta(theta, th0, p, q) - u
        deriv     = dudtheta(theta, th0, p, q)
        theta_new = theta - du/deriv
        return theta_new

    for n in arange(5):
        theta = iterate(u, theta, th0, p, q)

    return theta

def get_u_from_f(f, G0):
    """Calculate 4-velocity given initial Lorentz factor (G0) and ratio of swept up mass to ejecta mass (DL18, eq 4)"""
    return G0/sqrt(1. + 2.*G0*f + f**2.)*sqrt(1.+f/G0)

def get_f(r, theta0, A, k, M0, G0, theta_rhoads, modeltype='DL'):
    """Return fraction of ejecta mass swept up (f), given:
       $r$: radius (in cm)
       $\theta0$: initial opening angle,
       $A$: density (in cm$^{-3}$ for a constant density environment or g/cm for a wind environment)
       $k$: density profile
       $M_0$: initial ejecta mass (in g)
       $G0$: initial Lorentz factor
       $\theta_{\rm Rhoads}$: the opening angle, if using the Rhoads model
       modeltype: 'DL' (for this work) or 'Rhoads' (to use the results of Rhoads 1999)"""

    if (modeltype is 'DL'):
        theta = theta0
        for n in arange(10):
            Omega = 4.*pi*(sin(0.5*theta)**2.)  # Jet solid angle
            f     = Omega*A*r**(3.-k)/(3.-k)/M0 # DL18, eq 7
            u     = get_u_from_f(f, G0)
            theta = get_theta(theta0, u, k)
        return f
    elif (modeltype is 'Rhoads'):
        theta = theta_rhoads
        Omega = 4.*pi*(sin(0.5*theta)**2.)
        f     = Omega*A*r**(3.-k)/(3.-k)/M0
        return f
    else:
        raise ValueError("Model type must be 'DL' or 'Rhoads'")

#def get_df(r, dr, theta, rho, M0):
#    Omega = 4.*pi*(sin(0.5*theta)**2.)
#    return rho*Omega*r**2*dr/M0

def get_gamma(u):
    """Compute the Lorentz factor given 4-velocity (normalized to c)"""
    return sqrt(1. + u**2)

def get_dt(c, u, dr):
    """Return the time step for a given radial expansion,
       dt = dr/$\beta$c = dr$\Gamma/(uc)$, with $u = \Gamma\beta$"""
    return dr*sqrt(1.+u**2)/(c*u)

#def get_ttheta(c, u, r, theta):
#    """"""
##    Gamma = get_gamma(u)
##    beta  = u/Gamma
#    theta_Gamma = arctan(1./u)
#    theta_min = min(theta,theta_Gamma)
#    return r*(1.-cos(theta_min))/c

def get_X_from_ufluid(ufluid, r=4.):
    """Return the shock compression given the adiabatic index ratio, $r$ (LD18, eq 16)"""
    X = r*sqrt(1.+ufluid**2) + r-1.
    return X

def get_ushock_factor(X, r):
    """"Return the ratio of the shock 4-velocity to the fluid 4-velocity, 
        given the shock compression, $X$ and the adiabatic index ratio, $r$ (LD18, eq 17)"""
    return r*X/((X+1.)*(r**2+(r-2.)*r*X))**0.5

def get_Bk_from_k(Ak, k):
    """Return the function $B_k(k)$, representing the transition time to the Sedov-Taylor phase (LD18, eq 23)"""
    from numpy import exp
    return 0.026*exp(1.056*k)
def get_Ak_from_k(k):
    """Return the function $A_k(k)$, representing the correction to the fluid 4-velocity in the Sedov-Taylor phase (LD18, eq 23)"""
    return 1.99-0.62*k+0.089*k**2
def gk(k, x):
    """Return the function $g_k(k)$ used to convert the swept-up mass fraction, $f$ to the fluid 4-velocity (LD18, eq 23)"""   
    from numpy import sqrt
    Ak = get_Ak_from_k(k)
    Bk = get_Bk_from_k(Ak,k)
    return 1.+Ak/sqrt(1.+1./(Bk*x))

def get_upeak_from_f(f, G0, k):
    """Calculate the fluid 4-velocity from the swept-up mass fraction, given the initial Lorentz factor (LD18, eq 23)"""
    upeak = ( ((17.-4*k)/(6.-2*k)) * (G0**2/(1.+2*f*G0)) * 1./(gk(k,f/G0)) )**0.5
    return upeak
    
def get_R0(k):
    """Return the radial scale for a given density profile.
       This is used to convert the radius from code units to physical (cgs) units."""
    if (k == 0):
        R0 = 1.88e18 # Radius scale in cm = (1e52 erg/(proton mass cm^{-3} * c^2)
    elif (k == 2):
        R0 = 2.23e19 # Radius scale in cm = (1e52 erg/(5e11 g/cm * c^2))
    return R0

def calcall(theta0 = 0.14, k = 2.0, A = 1.0, E52 = 1.0, z = 1.0, G0 = 1000.0, thetaobs = 0.2, 
            rmin = 1e-8, rmax = 1e8, N = 10000, filename = 'output.txt', physical = True, modeltype = 'DL'):
    """Calculate the 4-velocity of the shock as a function of blastwave radius ($r$), given
    $\theta_0$: initial opening angle of the jet
    $k$: density profile (0 for constant density and 2 for wind-like medium)
    $A$: density parameter ($n_0$ for constant density and $A_*$ for wind-like medium)
    $E_{52}$: isotropic-equivalent kinetic energy
    $z$: cosmological redshift
    $G0$: initial bulk Lorentz factor, required to calculate the deceleration time
    $r_{\rm min}$: the smallest radius at which to begin the calculation (in units of $R_0$)
    $r_{\rm max}$: the radius at which to terminate the calculation (in units of $R_0$)
    $N$: number of steps at which to calculate the 4-velocity
    filename: output file name
    physical: if set to true, write file in physical (cgs) units
    modeltype: set to "DL" for Duffell and Laskar (2018) and "Rhoads" for the Rhoads (1999) model
    Returns:
    $u_{\rm shock}$: 4-velocity of the shock
    $u_{\rm peak}$: 4-velocity of post-shock fluid
    $u$: mean 4-velocity of post-shock fluid
    $\theta$: instantaneous jet opening angle
    $m_{\rm swept}$: mass swept up by shock (in code units or in g, if physical=True)
    $t$: elapsed lab-frame time (in code units, or in days if physical=True)
    $t_{\rm obs}$: elapsed observer-frame time (in code units, or in days and including cosmological redshift if physical=True)
    $r$: radius of the blastwave shock
    """

    import numpy as np
    R0 = get_R0(k) 

    E  = E52*sin(theta0/2.)**2. # kinetic energy of the ejecta in 1e52 erg, corrected for beaming
    c  = 1.                     # speed of light in code units  
    c0 = 3e10                   # speed of light in cm/s 
    t0 = R0/c0                  # physical time scale of the problem in seconds
    M0 = E/G0/c/c               # ejecta mass in code units

    rprev = 0          
    dr = 0
    outfile = open(filename,'w')
    headerline = "r               u               upeak           ushock          gamma           theta           mswept          t                tobs          tobsoffaxis    ushockfactor\n"
    outfile.write(headerline)
    
    rlist  = logspace(log10(rmin),log10(rmax),N)
    theta_rhoads = theta0
    for i in arange(N):
        r  = rlist[i]
        dr = r - rprev
        f = get_f(r, theta0, A, k, M0, G0, theta_rhoads=theta_rhoads, modeltype=modeltype) # calculate f
        u = get_u_from_f(f, G0)                # ufluid_average (LD18, eq 19 and 20) in units of c
        upeak = get_upeak_from_f(f, G0, k)     # 4-velocity of the post-shock fluid in units of c
        X      = get_X_from_ufluid(upeak)      # post-shock density compression factor (LD18, eq 7)
        ushockfactor = get_ushock_factor(X, 4) # Taking r = 4 (relativistic equation of state for shocked fluid)
        ushock = ushockfactor*upeak            # 4-velocity of the shock
 
        gamma = get_gamma(upeak)               # Lorentz factor of the post-shock fluid
        theta = get_theta(theta0, u, k)        # jet opening angle
        gamma_sh = get_gamma(ushock)           # Lorentz factor of the shock
        if (i == 0):
            t = r*gamma_sh/(c*ushock)
        else:
            t = t + get_dt(c, ushock, dr)
        rprev = r
        mswept = f*M0 * 1.11e31                # swept-up mass in g
        theta_rhoads = min(theta_rhoads+get_dt(c, ushock, dr)/(1.732*t*gamma_sh), np.pi/2) # opening angle for Rhoads (1999)
        mu = cos(thetaobs)
        
        if (physical):
            output = "{:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}     {:8.6e}  {:8.6e}   {:8.6e}\n".format\
                      (r*R0, u, upeak, ushock, gamma, theta, mswept, t*t0, (1.+z)*t0*(t-r/c)/86400., (1.+z)*t0*(t-r*mu/c)/86400., ushockfactor)
        else:
            output = "{:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}    {:8.6e}     {:8.6e}  {:8.6e}   {:8.6e}\n".format\
                      (r, u, upeak, ushock, gamma, theta, mwept, t, t-r/c, t-r*mu/c, ushockfactor)

        outfile.write(output)

    outfile.close()
    from astropy.io import ascii as asciireader
    data = asciireader.read(filename)
    return data
