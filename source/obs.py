def readdata(filename):
    from astropy.io import ascii as asciireader
    data = asciireader.read(filename)
    return data

def betaofu(u):    
    return u/(1.+u**2.)**0.5

def Gammaofu(u):
    return (1.+u**2.)**0.5

def uofx(X,r):
    return X*((X-2*r+1.)/(r**2+(r-2.)*r*X))**0.5

def Xofr(ufluid, r):
    return r*(1.+ufluid**2)**0.5 + r - 1.

def ushockofufluid(ufluid, r):
    X = Xofr(ufluid, r)
    return ufluid*r*X/((X+1.)*(r**2+(r-2.)*r*X))**0.5

def e2_calc(ushock, rho1, X, r, c):    
    return (r-1.)*(1.-X**(-1.))*rho1*ushock**2*c**2. / (1. + r*ushock**2/X**2)

def gammaminofe2(e2, rho2, m_p, m_e):
    return gstable2.e_ebar(params)*e2*(m_p/m_e)/(rho2*c**2)

def gammacool(params,t,B,ushock,sigma_T,m_e,c,mu0):
    import numpy as np
    w = 1.0; #1.3 # Cooling coefficient
    pfactor = 1.0#1./(0.126*p**2-0.494*p+0.649)**0.5# (p-0.46)*np.exp(-1.16*p)
    return 0.75*(m_e*c/sigma_T)*(2.*mu0/B**2)*(Gammaofu(ushock)/(t*86400.))*w*pfactor

def Bofe2(params, e2, mu0):
    e_b = params.e_b
    return (2*mu0*e_b*e2)**0.5

def numaxofgamma(params, ushock, gammae, B, qe, m_e, c, useSI):
    if (k == 0):
        gsscalefactor = 3.50*(params.p-0.67)/1.82 #7.5*pi/8.0
    else:
        gsscalefactor = 1.88*(params.p-0.69)/1.82 #2e-2
    return gsscalefactor*nuofgamma(params, ushock, gammae, B, qe, m_e, c, useSI)

def nucofgamma(params, ushock, gammac, B, qe, m_e, c, useSI):
    from numpy import exp
    p = params.p
    if (k == 0):
        gsscalefactor = 4*33.*(p-0.46)*exp(-1.16*p)
    else:
        gsscalefactor = 4*4.39e-2*(3.45-p)*exp(0.45*p)
    return gsscalefactor*nuofgamma(params, ushock, gammac, B, qe, m_e, c, useSI)

def nuofgamma(params, ushock, gammae, B, qe, m_e, c, useSI):
        if (useSI):
            return Gammaofu(ushock)*(1.+betaofu(ushock))*gammae**2.*qe*B/(2.*pi*m_e*(1.+z))
        else:
            return Gammaofu(ushock)*(1.+betaofu(ushock))*gammae**2.*qe*B/(2.*pi*m_e*c*(1.+z))

def nuaofgamma(params, ushock, gammamin, rshock, rho1, B, X, qe, eps0, c, m_p, m_e, useSI):
    G    = Gammaofu(ushock)
    beta = betaofu(ushock)
    p    = params.p
    R    = rshock
    from numpy import pi    
    from scipy.special import gamma
    if (useSI):
        cpower = 1
        additional_factor = 1
    else:
        cpower = 5./3.
        additional_factor = 10
    Z = 3**(2./3.)/(2**(8./3.)*5*pi**(1./6.)*gamma(5./6.)) * \
        ((p-1.)*(p+2.)/(3.*p+2.)) * \
        qe**(8./3.)*rho1/(eps0*c**cpower* m_e**(5./3.)*m_p) * \
        additional_factor # A mysterious factor that seems to be necessary with cgs units ?!

    if (k == 0):
        gsscalefactor = 0.255/(p+2.)**0.6
    else:
        gsscalefactor = 0.879/(p+2.)**0.6
    return gsscalefactor * (1.+beta) * G**0.4*(Z*R*X)**0.6*B**0.4/gammamin / (1.+z)

def nufiveofgamma(params, ushock, gammamin, rshock, rho2, B, qe, eps0, c, m_p, m_e, useSI):
    from numpy import pi
    from scipy.special import gamma
    G    = Gammaofu(ushock)
    beta = betaofu(ushock)
    p    = params.p
    R    = rshock
    Gammafactors = gamma((3.*p+22.)/12.) * gamma((3.*p+2.)/12.) *\
                   gamma((p+6.)/4.) / gamma((p+8.)/4.)    
    K = (p-1.)*gammamin**(p-1.)*rho2/m_p
    Z = (3*pi)**0.5 * qe**3 * B**((p+2.)/2.) / (64.*pi**2*eps0*m_e**2*c) * \
        (3*qe/(2*pi*m_e))**(p/2.) * \
        Gammafactors * K
    nuprime = (Z * R / G)**(2./(p+4.))
    nu = G*(1.+beta) * nuprime / (1.+z)
    p = params.p
    if (k == 0):
        gsscalefactor = (5.55*p**2 -4.91*p +46.66)*1e-2 #1.0#0.69#0.61 * (4.03-p)*exp(2.34*p)/262.83
    else:
        gsscalefactor = (-11.15*p**2 +83.17*p -37.59)*1e-2
    return gsscalefactor * nu

def nusixofothers(params, nuseven, numax, nuc):
    # NOTE: This break frequency requires nuseven and nuten WITHOUT the GS02 p-dependent corrections
    # The fits for p work better that way.
    # The gsscalefactors below are divided by a complicated-looking quantity,
    # which is simply the appropriate power of the gsscalefactor for nuseven, 
    # and which will cancel out the p-dependence of nuseven
    p = params.p
    nuten = nutenofothers(params, nuseven, numax, nuc, gsscale = False)    
    if (k == 0):
        gsscalefactor = (20.2*p-34.5)*exp(-1.18*p)/((-2.18*p**2 + 22.96*p - 8.71)*1e-2)**(10./(3*p+15.))
    else:
        gsscalefactor = (21.6*p-36.0)*exp(-1.16*p)/ ((-2.23*p**2 + 23.49*p - 9.04)*1e-2)**(10./(3*p+15.))
    return gsscalefactor*(nuten**(10./3.) * nuc**(8./3.) * numax**(p-1.))**(1./(p+5.))

def nusevenofgamma(params, ushock, rho2, B, numax, gammam, qe, c, m_p, m_e, gsscale=True):
    G = Gammaofu(ushock)
    from numpy import pi
    from scipy.special import gamma    
    prefactor = 9*pi**2/(8*gamma(1./3.))
    ufs       = c/3.                      # Speed of post-shock fluid in shock frame        
    n2        = rho2/m_p
    #prefactor = (9./(10.*gamma(5./6.)))**0.6 * (2./3.)**0.2 * pi**1.1
    p = params.p
    if (gsscale):
        if (k == 0):
            gsscalefactor = (-2.18*p**2 + 22.96*p - 8.71)*1e-2
        else:
            gsscalefactor = (-2.23*p**2 + 23.49*p - 9.04)*1e-2 #1#0.36
    else:
        gsscalefactor = 1.0
    return gsscalefactor*(prefactor* G**2*n2* m_e*c**2*ufs / (qe*B* (2*numax*(1.+z))**(1./3.) * gammam**2))**0.6 / (1.+z)
#    return gsscalefactor*prefactor * (gammam**2*m_e/(qe*B))**0.8 * (c**2*ufs*G*n2)**0.6 * G/(1.+z)

def nueightofothers(params, nuseven, numax, nuc):
    nuten = nutenofothers(params, nuseven, numax, nuc, gsscale = False)
    p = params.p
    if (k == 0):
        gsscalefactor = (16.74*p**2 - 116.76*p + 273.38)*1e-2
    else:
        gsscalefactor = (20.89*p**2 -145.46*p + 339.05)*1e-2
    return gsscalefactor*nuten**(5./9.) * nuc**(4./9.)

def nutenofothers(params, nuseven, numax, nuc, gsscale = True):
    p = params.p
    if (gsscale):
        if (k == 0):
            gsscalefactor = (79.38*p**2 -510.4*p +869.6)*1e-2
        else:
            gsscalefactor = 2.04*p**2 -15.45*p +30.20
    else:
        gsscalefactor = 1.0
    return gsscalefactor*nuseven*(numax/nuc)**0.8

def nuelevenofnuc(params, nuc):
    p = params.p
    if (k == 0):
        gsscalefactor = 9.2e-2/((p-0.46)*exp(-1.16*p))
    else:
        gsscalefactor = 0.532/((3.45-p)*exp(0.45*p))
    return gsscalefactor*nuc
    
def Frad(gammamin, ush, B, mswept, theta, DL, params, numax, sigma_T, c, mu0, m_p, mJy):
    Ne = mswept/m_p
    Omega = 4.*pi*(sin(0.5*theta)**2.)
    zeta  = zeta_beam(ush, offaxis=offaxis)
    return mJy*Gammaofu(ush)**2*(1.+betaofu(ush))*(4./3.)*sigma_T*c*(gammamin**2)*(B**2/(2.*mu0))*Ne/(4*pi*DL**2*numax)/Omega * zeta #/ (1. + 2*pi/(Omega*Gammaofu(ush)**2))

def FmaxofFrad(params, nuseven, numax, nuc, Frad):
    # Two factors to correct Frad to Fmax
    # The first one is p-dependent, and based on correcting Frad to F2
    p = params.p 
    if (params.k == 0):
        gsfactor1 = 1.541*p**2 -0.817*p -0.145 #5.45* (params.p + 0.14) / 2.5
        gsfactor2 = 4.41*p - 2.95
    else:
        gsfactor1 = 2.106*p**2 -1.20*p -0.174    #3.0*(params.p+0.12)
        gsfactor2 = 12.10*p - 8.35
    
    # The second one is also p-dependent, and is based on corrected Frad to F11
    nuten   = nutenofothers(params,nuseven,numax,nuc)
    nueight = nueightofothers(params,nuseven,numax,nuc)
    slowcooling    = numax < nuc
    fastcooling4   = (numax >= nuc) & (nuc < nueight)
    fastcooling5   = (numax >= nuc) & (nuc >= nueight)

    coolingfactor = gsfactor1*slowcooling + gsfactor1*fastcooling4 + gsfactor2*fastcooling5#2.588*fastcooling5
    return Frad * coolingfactor

def F_b (params, b, q_break, Norm, q):
    import gstable2
    s_break = gstable2.s(params, b)
    beta1   = gstable2.gsbeta1(params, b)
    beta2   = gstable2.gsbeta2(params, b)
    if b == 4:
        from numpy import e
        phi4 = q/q_break
        F = Norm*(phi4**2*e**(-1.*s_break*phi4**(2./3.))+phi4**2.5)
    elif b in (range(1,4) + range(5,12)):
        F = Norm* ((1.0*q/q_break)**(-1.*s_break*beta1)\
                + (1.0*q/q_break)**(-1.*s_break*beta2))\
                ** (-1./s_break)
    else:
        raise ValueError('Invalid break, b = %d' % (b))
    return F

def F_b_tilde (params, b, q_break, q):
    import gstable2
    s_break = gstable2.s(params, b)
    beta1   = gstable2.gsbeta1(params, b)
    beta2   = gstable2.gsbeta2(params, b)
    return (1+(1.0*q/q_break)**(s_break*(beta1-beta2)))**(-1./s_break)

def calculatespectrum(params, spectype, nua, num, nuc, Fmax, q):
    if (spectype == 1):
        fnua = Fmax*(nua/num)**(1./3.)
        return F_b(params, 1, nua, fnua, q)*F_b_tilde(params, 2, num, q)*F_b_tilde(params, 3, nuc, q)
    elif (spectype == 2):
        fnum = Fmax*(num/nua)**(0.5*params.p + 2)
        return F_b(params, 4, num, fnum, q)*F_b_tilde(params, 5, nua, q)*F_b_tilde(params, 3, nuc, q)
    elif (spectype == 4):
        nuac    = nua # nuseven
        nuten   = nutenofothers(params,nuac,num,nuc)
        nueight = nueightofothers(params,nuac,num,nuc)
        nusa    = nueight
        fnuac   = Fmax*(nusa/nuc)**(-0.5)*(nuac/nusa)**(11./8.)
        return F_b(params, 7, nuac, fnuac, q)*F_b_tilde(params, 8, nusa, q)\
               *F_b_tilde(params, 9, num, q)        
    elif (spectype == 5):
        nueleven = nuelevenofnuc(params, nuc)
        nuac  = nua
        nusa = nutenofothers(params,nuac,num,nuc) #nusa = nuac*(num/nuc)**0.8
        fnuac = Fmax*(nusa/nueleven)**(1./3.)*(nuac/nusa)**(11./8.)
        return F_b(params, 7, nuac, fnuac, q)*F_b_tilde(params, 10, nusa, q)\
               *F_b_tilde(params, 11, nueleven, q)*F_b_tilde(params, 9, num, q)

def spectrum_all(params, nua, num, nuc, nufive, nuseven, Fmax, q):
    # Break frequencies required for distinguishing b/w spectrum 4 & 5:
    nuten   = nutenofothers(params,nuseven,num,nuc)
    nueight = nueightofothers(params,nuseven,num,nuc)
    # Determine slow / fast and optically thick / thin
    slowcooling    = num < nuc
    fastcooling4   = (num >= nuc) & (nuc < nueight)
    fastcooling5   = (num >= nuc) & (nuc >= nueight)
    opticallythin  = nua < num
    opticallythick = nua >= num
    # Calculate all the spectra
    spectrum1 = calculatespectrum(params, 1, nua,     num, nuc, Fmax, q)
    spectrum2 = calculatespectrum(params, 2, nufive,  num, nuc, Fmax, q)
    spectrum4 = calculatespectrum(params, 4, nuseven, num, nuc, Fmax, q)
    spectrum5 = calculatespectrum(params, 5, nuseven, num, nuc, Fmax, q)
    F = (spectrum4*fastcooling4 + spectrum5*fastcooling5 + spectrum1*slowcooling) * opticallythin + spectrum2 * opticallythick
    return F

def zeta_beam(ush, offaxis=False):
    '''Calculate the beaming correction'''
    # Calculate beaming correction
    w = 3
    Gammaf = Gammaofu(ush)
    betaf  = betaofu(ush)    

    if (offaxis):
        Omega = 4.*pi*(sin(0.5*theta)**2.)
        mu                = 1.-cos(theta)
        one_minus_beta    = 1./(Gammaf**2*(1.+betaf))
        one_minus_beta_mu = (1.-mu**2 + mu**2/Gammaf**2)/(1.+betaf*mu)
        zeta_beam_jet     = (Omega / (4.*np.pi)) * (one_minus_beta/one_minus_beta_mu)**4
    else:    
        inv_one_minus_beta = Gammaf**2*(1.+betaf)
        zeta_beam_jet = (inv_one_minus_beta**w - (1./(1.-betaf*cos(theta)))**w) / \
                        (inv_one_minus_beta**w - (1./(1.+betaf))**w)

        betaf  = -betaofu(ush)    
        inv_one_minus_beta = Gammaf**2*(1.+betaf)
        zeta_beam_counterjet = (inv_one_minus_beta**w - (1./(1.-betaf*cos(theta)))**w) / \
                               (inv_one_minus_beta**w - (1./(1.+betaf))**w)

    return zeta_beam_jet #+ zeta_beam_counterjet
    
def Urad(params, nu, fnu, nutilde, m_p, c, DL, E):
    k = params.k   
    prefactor = 48.*pi*m_p*c*DL**2/((17-4.*k)*E)
    sel = nu < nutilde
    integral = trapz((fnu*nu)[sel], nu[sel])
    return prefactor * integral

useSI = True
# Constants
if (useSI):
    c      = 3e8                             # Speed of light in m/s
    qe     = 1.6021e-19                      # Fundamental charge
    m_p    = 1.673e-27                       # Proton mass in kg
    m_e    = 9.11e-31                        # Electron mass in kg
    mu0    = 4*pi*1e-7                       # Vacuum permeability in Tm/A
    eps0   = 1./(mu0*c**2)                   # Vacuum permittivity in Nm^2/C^2
    r      = 4                               # gamma/(gamma-1) = 4 for relativistic plasma
    sigma_T= 6.6525e-29                      # Electron Thomson cross-section in m^2
    mJy    = 1e29                            # To convert W/m^2.Hz into mJy
    d      = params.dL28*1e26                # Luminosity distance in m
    rshock = R*1e-2                          # Shock radius in m
    rho1   = gstable2.dens(params, R)*1e3    # Pre-shock density in kg/m^3
    mswept = data['mswept']*1e-3             # Swept up mass in kg
    Eiso   = params.E52*1e52/1e7             # Shock energy in J
else:
    c      = 3e10                            # Speed of light in cm/s
    qe     = 4.80326e-10                     # Fundamental charge in statcoulomb
    m_p    = 1.673e-24                       # Proton mass in  g
    m_e    = 9.11e-28                        # Electron mass in kg
    mu0    = 4*pi                            # Vacuum permeability
    eps0   = 1./4*pi                         # Vacuum permittivity
    r      = 4                               # gamma/(gamma-1) = 4 for relativistic plasma
    sigma_T= 6.6525e-25                      # Electron Thomson cross-section in cm^2
    mJy    = 1e26                            # To convert erg/cm^2.s.Hz into mJy
    d      = params.dL28*1e28                # Luminosity distance in cm
    rshock = R                               # Shock radius in cm
    rho1   = gstable2.dens(params, R)        # Pre-shock density in g/cm^3
    mswept = data['mswept']                  # Swept up mass in g
    Eiso   = params.E52*1e52                 # Shock energy in erg
    
# Hydrodynamic quantities
ufluid = u                               # Fluid 4-velocity (dimensionless)
X      = Xofr(ufluid, r)                 # Compression (dimensionless)
ushock = ushockofufluid(ufluid, r)       # Shock 4-velocity (dimensionless)
e2     = e2_calc(ushock, rho1, X, r, c)  # Post-shock energy density (J/m^3)
rho2   = X*rho1                          # Post-shock matter density (kg/m^3)
gammam = gammaminofe2(e2, rho2, m_p,m_e) # Minimum Lorentz factor
B2     = Bofe2(params, e2, mu0)          # Post-shock magnetic field in Tesla
gammac = gammacool(params,tlab,B2,ushock,sigma_T,m_e,c,mu0) # Cooling Lorentz factor

# Compute break frequencies
nua     = nuaofgamma(params, ushock, gammam, rshock, rho1, B2, X, qe, eps0, c, m_p, m_e, useSI)
numax   = numaxofgamma(params, ushock, gammam, B2, qe, m_e, c, useSI)
nuc     = nucofgamma(params, ushock, gammac, B2, qe, m_e, c, useSI)
nufive  = nufiveofgamma(params, ushock, gammam, rshock, rho2, B2, qe, eps0, c, m_p, m_e, useSI)
nuseven = nusevenofgamma(params, ushock, rho2, B2, numax, gammam, qe, c, m_p, m_e)
# Derived break frequencies
nuten   = nutenofothers(params,nuseven,numax,nuc)
nueight = nueightofothers(params,nuseven,numax,nuc)
nusix   = nusixofothers(params,nuseven,numax,nuc)

nu1  = gstable2.f_b(params,1,tobs,False)
nu2  = gstable2.f_b(params,2,tobs,False)
nu3  = gstable2.f_b(params,3,tobs,False)
nu5  = gstable2.f_b(params,5,tobs,False)
nu6  = gstable2.f_b(params,6,tobs,False)
nu7  = gstable2.f_b(params,7,tobs,False)
nu8  = gstable2.f_b(params,8,tobs,False)
nu10 = gstable2.f_b(params,10,tobs,False)

Fmax0 = Frad(gammam, ushock, B2, mswept, theta, d, params, numax, sigma_T, c, mu0, m_p, mJy)
Fmax  = FmaxofFrad(params, nuseven, numax, nuc, Fmax0)
F2    = gstable2.F_b_ext(params, 2, tobs)
F11   = gstable2.F_b_ext(params, 11, tobs)

plotfig  = True
plotzeta = True
if plotfig:
    figure()
    subplot(211)
    loglog(tobs,numax,'g-'); loglog(tobs,nu2,'g--')
    loglog(tobs,nuc,'b-');   loglog(tobs,nu3,'b--')
    loglog(tobs,nua,'r-');   loglog(tobs,nu1,'r--')
    loglog(tobs,nufive,'C0',ls='-'); loglog(tobs, nu5, 'C0',ls='--')
    sel7 = tobs < tjet
    loglog(tobs[sel7],nuseven[sel7],'C1',ls='-'); loglog(tobs[sel7], nu7[sel7],  'C1',ls='--')
    loglog(tobs[sel7],nuten[sel7],'C2',ls='-');   loglog(tobs[sel7], nu10[sel7], 'C2',ls='--')
    loglog(tobs[sel7],nueight[sel7],'C4',ls='-'); loglog(tobs[sel7], nu8[sel7],  'C4',ls='--')
    loglog(tobs[sel7],nusix[sel7],'C5',ls='-');   loglog(tobs[sel7], nu6[sel7],  'C5',ls='--')
    
    ylabel("Frequency")
    xlabel("Observer time (days)")
    axvline(tjet,color='k',ls='-')
    axvline(tnr, color='k',ls='--')
    axvline(tsph,color='k',ls=':')

    subplot(212)
    plot(tobs,(numax/nu2-1.)*100.,'g')
    plot(tobs,(nuc/nu3-1.)*100.,'b')
    plot(tobs,(nua/nu1-1.)*100.,'r')
    plot(tobs,(nufive/nu5-1.)*100.,'C0')
    plot(tobs[sel7],(nuseven[sel7]/nu7[sel7]-1.)*100.,'C1')
    plot(tobs[sel7],(nuten[sel7]/nu10[sel7]-1.)*100.,'C2')
    plot(tobs[sel7],(nueight[sel7]/nu8[sel7]-1.)*100.,'C4')
    plot(tobs[sel7],(nusix[sel7]/nu6[sel7]-1.)*100.,'C5')
    
    xscale('log')
    ylabel("Relative error (%)")
    xlabel("Observer time (days)")
    axvline(tjet,color='k',ls='-')
    axvline(tnr, color='k',ls='--')
    axvline(tsph,color='k',ls=':')
    
    tight_layout()
    plt.ylim(-500,500)

    figure()
    loglog(tobs,Fmax)
    loglog(tobs,F2)
    loglog(tobs, Fmax/F2)
    lw = 3; fontsize = 18
    
    if plotzeta:
        fig, ax1 = plt.subplots(1)
        zetacolor = 'red'; zetals = '-'; Omegacolor = 'k'; Omegals = '--'; approxcolor = 'orange'; approxls = '-.'
        artistlist = []; namelist = [r'Beaming correction, $\zeta_{\rm beam}$',r'Jet solid angle, $\Omega/4\pi$']
        zeta       = zeta_beam(ushock, offaxis=offaxis)
        Omega      = 4.*pi*sin(theta/2)**2
        Omegaobs   = pi/Gammaofu(ushock)**2
        zetaapprox = 1-(Omegaobs/Omega)
        art = ax1.loglog(tobs, zeta, color=zetacolor, ls=zetals,lw=lw); artistlist.append(art[0])
        art = ax1.loglog(tobs, Omega/(4.*pi), color=Omegacolor,ls=Omegals,lw=lw); artistlist.append(art[0])
        art = ax1.loglog(tobs, zetaapprox, color=approxcolor, ls=approxls, lw=lw);
        smallestplotvalue = min(min(Omega/(4.*pi)),min(zeta))
        xlim(6e-2,1e7); ylim(smallestplotvalue/1.2,1.2)   
        ax1.set_xlabel("Observer time (days)", fontsize = fontsize)
        plt.legend(artistlist,namelist,loc='lower right',shadow=True)
        axvline(2.2,color='b',ls=':',lw=2)
        savefig('zeta.png')
