applyIC = False

def R(params, t):
    '''The blast wave radius in cm at time, t from the BM solution'''
    E52   = params.E52
    n0    = params.n_0
    Astar = params.Astar
    t_nr = params.t_nr
    k = params.k    
    z = params.z
    m_p = 1.67262178e-24 # Proton mass in grams
    c = 2.99792458e10 # Speed of light in cm/s
    from numpy import pi
    if (k == 0):
        dens = n0*m_p    # density in g/cm^3
    elif (k == 2):        
        dens = 5.e11*Astar # density in g/cm
    else:
        raise(ValueError("Invalid value of k, neither 0 nor 2"))    
    R_tnr = ((17.-4.*k)*(4.-k)*(E52*1e52)*(t_nr*86400.)/(4.*pi*dens*c*(1.+z)))**(1./(4.-k))
    return (t < t_nr)*R_tnr*(t/t_nr)**(1./(4.-k)) + (t >= t_nr)*R_tnr*(t/t_nr)**(2./5.) # The latter is the Sedov-Taylor solution
    #return ((17.-4.*k)*(4.-k)*(E52*1e52)*(t*86400.)/(4.*pi*dens*c*(1.+z)))**(1./(4.-k))

def Gamma(params, t):
    '''The blast wave four-velocity as a function of time, including the jet break and NR transition'''    
    E52   = params.E52
    n0    = params.n_0
    Astar = params.Astar
    t_jet = params.t_jet
    t_nr = params.t_nr
    k = params.k    
    z = params.z
    m_p = 1.67262178e-24 # Proton mass in grams
    c = 2.99792458e10 # Speed of light in cm/s
    from numpy import pi
    if (k == 0):
        dens = n0*m_p    # density in g/cm^3
    elif (k == 2):        
        dens = 5.e11*Astar # density in g/cm
    else:
        raise(ValueError("Invalid value of k, neither 0 nor 2"))
    
    GBM        = Gamma_BM(params, t)
    Gamma_tjet = Gamma_BM(params, t_jet)
    Gamma_nr   = Gamma_tjet*(t_nr/t_jet)**(-0.5)
    Gamma = (t < t_jet)*GBM + ((t >= t_jet) & (t < t_nr))*Gamma_tjet*(t/t_jet)**(-0.5) + (t > t_nr)*Gamma_nr*(t/t_nr)**(-0.6)
    
    #Gamma_tnr = 1.215#((17.-4.*k)*(E52*1e52)*(1.+z)**(3.-k)/(4.**(5.-k)*(4.-k)**(3.-k)*pi*dens*c**(5.-k)*(t_nr*86400.)**(3.-k)))**(1./(8.-2.*k))
    #beta_tnr = (1.-1./Gamma_tnr**2.)**0.5 # Beta at the NR transition
    #beta = beta_tnr*(t/t_nr)**(-0.6)*(t >= t_nr) + beta_tnr * (t < t_nr) # Beta following the NR transition: Sedov-Taylor evolution    
    #Gamma_nr = 1./(1.-beta**2)**0.5  # Gamma following the NR transition: ST evolution
    #return (t < t_nr)*Gamma_tnr*(t_nr/t)**((3.-k)/(8.-2.*k)) + (t >= t_nr)*Gamma_nr

    return Gamma
    
def Gamma_BM(params,t):
    '''The Lorentz factor of the fluid just behind the blast wave at time, t from the BM solution'''    
    E52   = params.E52
    n0    = params.n_0
    Astar = params.Astar
    t_jet = params.t_jet
    t_nr = params.t_nr
    k = params.k    
    z = params.z
    m_p = 1.67262178e-24 # Proton mass in grams
    c = 2.99792458e10 # Speed of light in cm/s
    from numpy import pi
    if (k == 0):
        dens = n0*m_p    # density in g/cm^3
    elif (k == 2):        
        dens = 5.e11*Astar # density in g/cm
    else:
        raise(ValueError("Invalid value of k, neither 0 nor 2"))
    Gamma_BM = ((17.-4.*k)*(E52*1e52)*(1.+z)**(3.-k)/(4.**(5.-k)*(4.-k)**(3.-k)*pi*dens*c**(5.-k)*(t*86400.)**(3.-k)))**(1./(8.-2.*k))
    return Gamma_BM

def B(params, t):
    '''The post-shock fluid-frame magnetic field in G as a function of observer time, t
       Only valid in the BM (pre-jet break, ultra-relativistic) phase'''
    from numpy import pi
    shock_radius = R(params, t)
    rho          = dens(params, R)
    c            = 3e10 # Speed of light in cm/s
    G            = Gamma_BM(params, t)
    return (16*pi*params.e_b*rho*c**2 * G**2)**0.5

def gamma_m(params, t):
    '''The minimum injection Lorentz factor in the blast wave'''
    return 1836.15267*e_ebar(params)*Gamma(params, t) # SPN98; 1836.xx is m_p / m_e

def gamma_self(params, t):    
    '''Calculates the Lorentz factor of the electrons, who IC scatter best off of their own synchrotron radiation;
       from Equation 7 of Nakar, Ando, and Sari (2009)'''
    B_QED = 4.4e13
    return (B_QED/B(params,t))**(1./3.)

def gamma_mhat(params, t):
    return gamma_self(params,t)**3/gamma_m(params,t)**2

def gamma_c(params, t):
    '''The cooling Lorentz factor, calculated using equation 6 in Sari, Piran & Narayan (1998)'''
    G   = Gamma_BM(params, t)
    z   = params.z
    Bsh = B(params,t)
    return 8.96e3 * (1.+z)/ (G* t* Bsh**2)

def gamma_hat(params, gamma, t):
    return gamma_self(params,t)**3/gamma**2

def gamma_chat(params, t):
    return gamma_self(params,t)**3/gamma_c(params,t)**2

def nuofgamma(params, gamma, t):
    G   = Gamma_BM(params, t)
    Bsh = B(params, t)
    z   = params.z
    return 2.8e6*G*gamma**2*Bsh/(1.+z)

def gammaofnu(params, nu, t):
    G   = Gamma_BM(params, t)
    Bsh = B(params, t)
    z   = params.z
    return (nu*(1.+z)/(2.8e6*G*Bsh))**0.5

def Y(params):
    '''The Compton Y-parameter in fast cooling'''
    if (not(applyIC)):
        return 0
    from numpy import sqrt
    eta = 1.0
    return 0.5*(sqrt(1.+4.*eta*params.e_e/params.e_b) - 1.0)

def Y_slow(params,t):
    '''The Compton Y-parameter for slow cooling'''
    if (not(applyIC)):
        return 0    
    import numpy as np
    numax   = f_b(params,2,t)#np.sqrt(f_b(params,2,t)*f_b(params,4,t)) # An average numax that is between nu2 and nu4
    nuc_syn = f_b(params,3,t,applyIC=False) # The cooling frequency in the absence of IC cooling
    fast = nuc_syn < numax
    A = (nuc_syn/numax)**(-0.5*(params.p-2.))        
    eta_min = 1e-6
    eta_max = 1.0
    niter_max = 20
    deta_max = 1e-50 # I was noticing differences in model values for slightly different parameters. This is based on trial and error, to minimize those differences
    deta = 1.0
    p = params.p
    e_e = params.e_e
    e_b = params.e_b
    
    if ((~fast).sum() == 0): # GRB never gets to slow cooling. Return the fast cooling Y-parameter
        eta = 1.
        Y = 0.5*(np.sqrt(1.+4.*eta*e_e/e_b) - 1.0)
        return Y
    
    if (np.ndim(t) == 0):
        def func(x):        
            if (x < 0):
                return 100.
            else:
                return x - A*((1.+np.sqrt(1.+4.*x*e_e/e_b))/2.)**(p-2.0)
        def dfunc(x):
            if (x < 0):
                return 0.0
            else:
                return 1. - A*(p-2.)*(0.5*(1.+np.sqrt(1.+4.*x*e_e/e_b)))**(p-3.)*(e_e/e_b)/np.sqrt(1.+4.*x*e_e/e_b)
        eta = 0.5    
        if (fast):
            eta = 1.0
        else:            
            if (func(1.0)<0):
                eta = 1.
            else:
                i = 0
                while ((deta > deta_max) and (i < niter_max)):
                #for i in np.arange(niter):                    
                    deriv = dfunc(eta)
                    eta_new = eta - func(eta)/deriv
                    if (eta_new < eta_min):
                        eta_new = eta_min
                    if (eta_new > eta_max):
                        eta_new = eta_max
                    deta = abs(eta-eta_new); i = i+1                    
                    eta = eta_new            
        Y = 0.5*(np.sqrt(1.+4.*eta*e_e/e_b) - 1.0)                        
    else:        
        def func(x,A):
            if ((x < 0).sum() > 0):
                return 100.*np.ones_like(t[~fast])
            else:
                return x - A*((1.+np.sqrt(1.+4.*x*e_e/e_b))/2.)**(p-2.0)
        def dfunc(x,A):
            if ((x < 0).sum() > 0):
                return 0.0
            else:
                return 1. - A*(p-2.)*(0.5*(1.+np.sqrt(1.+4.*x*e_e/e_b)))**(p-3.)*(e_e/e_b)/np.sqrt(1.+4.*x*e_e/e_b)
        eta = 1.0*np.ones_like(t)
        #if (~fast.any()):
        eta_slow = eta[~fast]*0.5
        i = 0
        while ((deta > deta_max) and (i < niter_max)):
        #for i in np.arange(niter):
            deriv_slow = dfunc(eta_slow,A[~fast])
            eta_new_slow = eta_slow - func(eta_slow,A[~fast])/deriv_slow
            eta_new_slow[eta_new_slow < eta_min] = eta_min
            eta_new_slow[eta_new_slow > eta_max] = eta_max
            if (i != 0):
                deta = max(abs(eta_slow-eta_new_slow))
            i = i+1
            eta_slow = eta_new_slow
        eta_slow[func(eta_slow,A[~fast]) < 0] = 1.0
        eta_slow[eta_slow > 1] = 1.0
        eta[~fast] = eta_slow
            
        Y = 0.5*(np.sqrt(1.+4.*eta*e_e/e_b) - 1.0)
        
    return Y

def s(params, b):
    'The smoothing function'
    k = params.k
    p = params.p
    if k == 0:
        if b == 1:
            x = 1.64
        elif b == 2:
            x = 1.84 - 0.40*p
        elif b == 3:
            x = 1.15 - 0.06*p
        elif b == 4:
            x = 3.44*p - 1.41
        elif b == 5:
            x = 1.47 - 0.21*p
        elif b == 6:
            x = 0.94 - 0.14*p
        elif b == 7:
            x = 1.99 - 0.04*p
        elif b == 8:
            x = 0.907
        elif b == 9:
            x = 3.34 - 0.82*p
        elif b == 10:
            x = 1.213
        elif b == 11:
            x = 0.597
        else:
            raise ValueError('Invalid break, b = %d' % (b))
    elif k == 2:
        if b == 1:
            x = 1.06
        elif b == 2:
            x = 1.76 - 0.38*p
        elif b == 3:
            x = 0.80 - 0.03*p
        elif b == 4:
            x = 3.63*p - 1.60
        elif b == 5:
            x = 1.25 - 0.18*p
        elif b == 6:
            x = 1.04 - 0.16*p
        elif b == 7:
            x = 1.97 - 0.04*p
        elif b == 8:
            x = 0.893
        elif b == 9:
            x = 3.68 - 0.89*p
        elif b == 10:
            x = 1.213 # Assuming the same as the ISM case            
        elif b == 11:
            x = 0.597 # Assuming the same as the ISM case
        else:
            raise ValueError('Invalid break, b = %d' % (b))
    else:
        raise ValueError('Invalid index k = %d' % (k))
    return x

def gsbeta1(params, b):
    p = params.p
    gsbeta = {1: 2.0, 2: 1./3., 3: (1.-p)/2., 4: 2.0, 5: 2.5, 6: 2.5, 7: 2.0, 8: 11./8., 9: -0.5, 10: 11./8., 11: 1./3.}
    try:
        b in gsbeta
        return gsbeta[b]
    except KeyError:
        print 'Invalid break, b = %d' % (b)    

def gsbeta2(params, b):
    p = params.p
    gsbeta = {1: 1./3., 2: (1.-p)/2., 3: -1.*p/2., 4: 2.5, 5: (1.-p)/2., 6: -1.*p/2., 7: 11./8., 8: -0.5, 9: -1.*p/2., 10: 1./3., 11: -0.5}
    #gsbeta = {1: 1./3., 2: (1.-p)/2., 3: -3*(p-1.)/4., 4: 2.5, 5: (1.-p)/2., 6: -1.*p/2., 7: 11./8., 8: -0.5, 9: -1.*p/2., 10: 1./3., 11: -0.5} # with KN correction (see 161219B paper, appendix A)
    try:
        b in gsbeta
        return gsbeta[b]
    except KeyError:
        print 'Invalid break, b = %d' % (b)    

def freqev1(params, b):
    '''Time evolution of the break frequency, b, before the jet break'''    
    k = params.k
    p = params.p    
    if k == 0:
        freqev = {1: 0., 2: -1.5, 3: -0.5, 4: -1.5, 5: -(3.*p+2.)/(2.*p+8.), 6: -3*(p+1.)/(2.*p+10.), 7: 0.3, 8: -0.5, 9: -1.5, 10: -0.5, 11: -0.5}
        #freqev = {1: 0., 2: -1.5, 3: -(8.-3*p)/(8.-2*p), 4: -1.5, 5: -(3.*p+2.)/(2.*p+8.), 6: -3*(p+1.)/(2.*p+10.), 7: 0.3, 8: -0.5, 9: -1.5, 10: -0.5, 11: -0.5} # with KN correction (see 161219B paper, appendix A)
        try:
            b in freqev            
            return freqev[b]
        except KeyError:
            print 'Invalid break, b = %d in freqev1(params, b) for jet break calculation' % (b)
    elif k == 2:
        freqev = {1: -0.6, 2: -1.5, 3: 0.5, 4: -1.5, 5: -3.*(p+2.)/(2.*p+8.), 6: -1.*(3.*p+5.)/(2.*p+10.), 7: 0., 8: -2./3., 9: -1.5, 10: -1.6, 11: 0.5}
        try:
            b in freqev
            return freqev[b]
        except KeyError:
            print 'Invalid break, b = %d in freqev1(params, b) for jet break calculation' % (b)
    else:
        raise ValueError('Invalid index k = %d' % (k))    

def freqev2(params, b):
    '''Time evolution of the break frequency, b, after the jet break'''
    p = params.p
    freqev = {1: -0.2, 2: -2.0, 3: 0., 4: -2., 5: -2*(p+1.)/(p+4.), 6: -2*(p+1.)/(p+5.), 7: 0.4, 8: -2./3., 9: -2.0, 10: -1.2, 11: 0.0}
    try:
        b in freqev
        return freqev[b]
    except KeyError:
        print 'Invalid break, b = %d in freqev2(params, b) for jet break calculation' % (b)

def freqev3(params, b):
    '''Time evolution of the break frequency, b, after the non-relativistic transition'''    
    p = params.p
    freqev = {1: 1.2, 2: -3.0, 3: -0.2, 4: -3.0, 5: -1*(3*p-2)/(p+4.), 9: -3.0, 11: -0.2}
    try:
        b in freqev
        return freqev[b]
    except KeyError:
        print 'Invalid break, b = %d in freqev3(params, b) for non-relativistic transition calculation' % (b)    
    
def fluxev1(params, b):
    '''Evolution of the flux density at the break frequency, b, before the jet break'''    
    k = params.k
    p = params.p    
    if k == 0:
        fluxev = {1: 0.5, 2: 0.0, 3: (1.-p)/2.0, 4: -2.5, 5: -5*(p-1.)/(2.*p+8.), 6: -5*(p-1.)/(2*p+10.), 7: 1.1, 8: 0., 9: 0.5, 10: 0., 11: 0.}
        try:
            b in fluxev            
            return fluxev[b]
        except KeyError:
            print 'Invalid break, b = %d in freqev1(params, b) for jet break calculation' % (b)
    elif k == 2:
        fluxev = {1: -0.2, 2: -0.5, 3: 0.5-p, 4: -2.0, 5: -(4*p+1.)/(2.*p+8.), 6: -1.*(4*p-5.)/(2*p+10.), 7: 1., 8: 1./12., 9: 0.5, 10: -1.2, 11: -0.5}
        try:
            b in fluxev            
            return fluxev[b]
        except KeyError:
            print 'Invalid break, b = %d in freqev1(params, b) for jet break calculation' % (b)
    else:
        raise ValueError('Invalid index k = %d' % (k))    

def fluxev2(params, b):
    '''Evolution of the flux density at the break frequency, b, after the jet break'''
    p = params.p    
    fluxev = {1: -0.4, 2: -1.0, 3: -1.*p, 4: -4., 5: -(4*p+1.)/(p+4.), 6: -4.*p/(p+5.), 7: 0.8, 8: -2./3., 9: 0., 10: -1.4, 11: -1.}
    try:
        b in fluxev
        return fluxev[b]
    except KeyError:
        print 'Invalid break, b = %d in freqev2(params, b) for jet break calculation' % (b)

def fluxev3(params, b):
    '''Evolution of the flux density at the break frequency, b, after the non-relativistic transition'''
    p = params.p
    fluxev = {1: 2., 2: 0.6, 3: 2 - 1.4*p, 4: -6.4, 5: 0.6 - 7*(p-1.)/(p+4.)}
    try:
        b in fluxev
        return fluxev[b]
    except KeyError:
        print ValueError('Invalid break, b = %d in freqev3(params, b) for non-relativistic transition calculation' % (b))    
    
def s_jet(params, b):
    '''The steepness of the jet break for the time evolution of the break frequencies'''
    from numpy import sign
    s0 = 5
    a1 = freqev1(params, b)
    a2 = freqev2(params, b)
    return s0*sign(a1-a2)

def s_jetf(params, b):
    '''The steepness of the jet break for the time evolution of the flux at the break frequencies'''
    from numpy import sign
    s0 = 5
    a1 = fluxev1(params, b)
    a2 = fluxev2(params, b)
    return s0*sign(a1-a2)

def s_nr(params, b):
    '''The steepness of the non-relativistic transition for the time evolution of the break frequencies'''
    from numpy import sign
    s0 = 15
    a1 = freqev2(params, b)
    a2 = freqev3(params, b)
    return s0*sign(a1-a2)

def s_nrf(params, b):
    '''The steepness of the non-relativistic transition for the time evolution of the flux at the break frequencies'''
    from numpy import sign
    s0 = 5.0#0.45
    a1 = fluxev2(params, b)
    a2 = fluxev3(params, b)
    return s0*sign(a1-a2)
    
def e_ebar(params):
    e_e = params.e_e / params.zeta
    p   = params.p
    x = e_e*(p-2)/(p-1.)
    return x

def f_b(params, b = 1, t = 1.0, applyIC = True):
    """Break frequency for parameters 'params', break number, 'b' and at time, t (days) """
    k = params.k
    p = params.p
    z = params.z
    zeta = params.zeta
    e_e = params.e_e / zeta
    e_b = params.e_b / zeta
    n_0 = params.n_0 * zeta
    Astar = params.Astar * zeta
    E52 = params.E52 * zeta
    t_jet = params.t_jet
    t_nr  = params.t_nr

    from numpy import e                 
    if k == 0:
        sj = s_jet(params,b)    # This restriction is needed since s_jet is not defined otherwise in this code
        jeteffect = ((t / t_jet)**(-1*sj*freqev1(params,b)) + \
                        (t / t_jet)**(-1*sj*freqev2(params,b)))**(-1/sj)                    
        if b in (1,2,3,4,5,9,11):   # If we are in slow cooling, or if b=9 or b=11 define the nr transition sharpness            
            s_sr = s_nr(params, b)  # non-relativistic transition sharpness            
            nreffect = (1+(t/t_nr)**(s_sr*(freqev2(params,b)-freqev3(params,b))))**(-1./s_sr)            
            
        if b == 1:
            x = 1.24*(((p-1.)/(3.*p+2.))** 0.6) *1E9 \
                    * (1.+z)** (-1) * e_ebar(params)** (-1) \
                    * (e_b * n_0** 3. * E52)** (0.2) \
                    * jeteffect * nreffect
        elif b == 2:
            x = 3.73*(p-0.67)*1E15\
                    *((1.+z) * E52 * e_b)** 0.5 \
                    * e_ebar(params)** 2. * t_jet**(-1.5)\
                    * jeteffect * nreffect
        elif b == 3:
            if (applyIC):
                x = 6.37*(p-0.46)*1E13\
                        * e ** (-1.16*p) \
                        *((1.+z) * E52 * e_b ** 3.)** (-0.5) \
                        * n_0** (-1.) *t_jet**(-0.5)\
                        * jeteffect * nreffect * (1.+Y_slow(params,t))**(-2.)
            else:
                x = 6.37*(p-0.46)*1E13\
                        * e ** (-1.16*p) \
                        *((1.+z) * E52 * e_b ** 3.)** (-0.5) \
                        * n_0** (-1.) *t_jet**(-0.5)\
                        * jeteffect * nreffect
        elif b == 4:
            x = 5.04*(p-1.22)*1E16\
                    *((1.+z) * E52 * e_b * t_jet ** (-3.))** 0.5 \
                    * e_ebar(params)** 2.\
                    * jeteffect * nreffect
        elif b == 5:
            x = 3.59*(4.03-p)*1E9\
                    *e** (2.34*p) \
                    * (e_ebar(params)** (4.*(p-1.)) * e_b** (p+2.)\
                    * n_0** 4. * E52** (p+2.) \
                    / ((1.+z)** (6.-p) * t_jet** (3.*p+2.)))**(1./(2.*(p+4.)))\
                    * jeteffect * nreffect
        elif b == 6:
            if (applyIC):
                x = 3.23*(p-1.76)*1E12\
                    * (e_ebar(params)** (4.*(p-1.)) * e_b** (p-1.)\
                    * n_0** 2. * E52** (p+1.) \
                    / ((1.+z)** (7.-p) * t_jet**(3.*p+3.)))\
                    **(1./(2.*(p+5.))) * jeteffect * (1.+Y(params))**(-2./(params.p+5.))
            else:
                x = 3.23*(p-1.76)*1E12\
                    * (e_ebar(params)** (4.*(p-1.)) * e_b** (p-1.)\
                    * n_0** 2. * E52** (p+1.) \
                    / ((1.+z)** (7.-p) * t_jet**(3.*p+3.)))\
                    **(1./(2.*(p+5.))) * jeteffect
        elif b == 7:
            if (applyIC):
                x = 1.12*((3.*p-1.)/(3.*p+2.))**1.6 *1E08\
                    * (1.+z)**(-1.3) * e_ebar(params)** (-1.6)\
                    * e_b** (-0.4) * n_0** 0.3 * E52** (-0.1)\
                    * t_jet** (0.3) * jeteffect * (1.+Y(params))**(-3./5.)
            else:
                x = 1.12*((3.*p-1.)/(3.*p+2.))**1.6 *1E08\
                    * (1.+z)**(-1.3) * e_ebar(params)** (-1.6)\
                    * e_b** (-0.4) * n_0** 0.3 * E52** (-0.1)\
                    * t_jet** (0.3) * jeteffect
        elif b == 8:
            if (applyIC):
                x = 1.98E11 *(1.+z)**(-0.5) * (n_0 * E52)**(1./6.) * t_jet** (-0.5) * jeteffect * (1.+Y(params))**(-1./3.)
            else:
                x = 1.98E11 *(1.+z)**(-0.5) * (n_0 * E52)**(1./6.) * t_jet** (-0.5) * jeteffect                
        elif b == 9:
            x = 3.94*(p-0.74)*1E15* (1+z)** 0.5 * e_ebar(params)** 2.\
                    * (e_b * E52 * t_jet**(-3.0)) ** 0.5 \
                    * jeteffect * nreffect
        elif b == 10:
            if (applyIC):
                x = 1.32E10 * (1.+z)** (-0.5) * e_b** 1.2 * n_0** 1.1\
                    * E52** 0.7 * t_jet** (-0.5) * jeteffect * (1.+Y(params))
            else:
                x = 1.32E10 * (1.+z)** (-0.5) * e_b** 1.2 * n_0** 1.1\
                    * E52** 0.7 * t_jet** (-0.5) * jeteffect
        elif b == 11:
            if (applyIC):
                x = 5.86E12 * ( (1.+z) * e_b**3. * n_0**2. * E52 * t_jet)** (-0.5)\
                    * jeteffect * nreffect * (1.+Y(params))**(-2.)
            else:
                x = 5.86E12 * ( (1.+z) * e_b**3. * n_0**2. * E52 * t_jet)** (-0.5)\
                    * jeteffect * nreffect
        else:
            raise ValueError('Invalid break, b = %d' % (b))
        
    elif k == 2:
        if (b != 8): # NOTE: nu_8 is NOT included here, since there is no change in slope across t_jet
            sj = s_jet(params,b)   
            jeteffect = ((t / t_jet)**(-1*sj*freqev1(params,b)) + \
                        (t / t_jet)**(-1*sj*freqev2(params,b)))**(-1/sj)        
        if b in (1,2,3,4,5,9,11):   # If we are in slow cooling, or if b=9 or b=11 define the nr transition sharpness            
            s_sr = s_nr(params, b)  # non-relativistic transition sharpness            
            nreffect = (1+(t/t_nr)**(s_sr*(freqev2(params,b)-freqev3(params,b))))**(-1./s_sr)
            
        if b == 1:
            x = 8.31*(((p-1.)/(3.*p+2.))** 0.6) *1E9 \
                    * (1.+z)** (-0.4) * e_ebar(params)** (-1.) \
                    * (e_b * Astar** 6. * E52**(-2.) * t_jet** (-3.))** (0.2)\
                    * jeteffect * nreffect
        elif b == 2:
            x = 4.02*(p-0.69)*1E15\
                    *((1.+z) * E52 * e_b * t_jet** (-3.))** 0.5 \
                    * e_ebar(params)** 2.\
                    * jeteffect * nreffect
        elif b == 3:
            if (applyIC):
                x = 4.40*(3.45-p)*1E10\
                    * e** (0.45*p) \
                    *(1.+z)** (-1.5) * e_b** (-1.5) \
                    * Astar** (-2) * E52** (0.5) * t_jet** (0.5)\
                    * jeteffect * nreffect * (1.+Y(params))**(-2.)
            else:
                x = 4.40*(3.45-p)*1E10\
                    * e** (0.45*p) \
                    *(1.+z)** (-1.5) * e_b** (-1.5) \
                    * Astar** (-2) * E52** (0.5) * t_jet** (0.5)\
                    * jeteffect * nreffect
        elif b == 4:
            x= 8.08*(p-1.22)*1E16\
                    *((1.+z) * E52 * e_b * t_jet** (-3.))** 0.5 \
                    * e_ebar(params)** 2.\
                    * jeteffect * nreffect
        elif b == 5:
            x = 1.58*(4.10-p)*1E10\
                    *e** (2.16*p) \
                    * (e_ebar(params)** (4.*(p-1.)) * e_b** (p+2.)\
                    * Astar** 8. \
                    / ((1.+z)** (2.-p) * E52** (2.-p) * t_jet** (3*p+6.)))\
                    **(1./(2.*(p+4.))) \
                    * jeteffect * nreffect
        elif b == 6:
            if (applyIC):
                x = 4.51*(p-1.73)*1E12\
                    * (e_ebar(params)** (4.*(p-1.)) * e_b** (p-1.)\
                    * Astar** 4. * E52** (p-1.)  \
                    / ((1.+z)** (5.-p) * t_jet**(3.*p+5.)))\
                    **(1./(2.*(p+5.))) * jeteffect * (1.+Y(params))**(-2./(params.p+5.))
            else:
                x = 4.51*(p-1.73)*1E12\
                    * (e_ebar(params)** (4.*(p-1.)) * e_b** (p-1.)\
                    * Astar** 4. * E52** (p-1.)  \
                    / ((1.+z)** (5.-p) * t_jet**(3.*p+5.)))\
                    **(1./(2.*(p+5.))) * jeteffect
        elif b == 7:
            if (applyIC):
                x = 1.68*((3.*p-1.)/(3.*p+2.))**1.6 *1E08\
                    * (1.+z)**(-1.) * e_ebar(params)** (-1.6)\
                    * e_b** (-0.4) * Astar** 0.6 * E52** (-0.4) * jeteffect * (1.+Y(params))**(-3./5.)
            else:
                x = 1.68*((3.*p-1.)/(3.*p+2.))**1.6 *1E08\
                    * (1.+z)**(-1.) * e_ebar(params)** (-1.6)\
                    * e_b** (-0.4) * Astar** 0.6 * E52** (-0.4) * jeteffect
        elif b == 8:
            if (applyIC):
                x = 3.15E11 * (1.+z)** (-1./3.) * Astar** (1./3.) \
                    * t ** (-2./3.) * (1.+Y(params))**(-1./3.) # Note - nu_8 (k = 2) does not change across the jet break, so "jeteffect = 0"
            else:
                x = 3.15E11 * (1.+z)** (-1./3.) * Astar** (1./3.) \
                    * t ** (-2./3.) # Note - nu_8 (k = 2) does not change across the jet break, so "jeteffect = 0"
        elif b == 9:            
            x = 3.52*(p-0.31)*1E15* (1+z)** 0.5 * e_ebar(params)** 2.\
                    * (e_b * E52 * t_jet**(-3.0)) ** 0.5 \
                    * jeteffect * nreffect
        elif b == 10:
            if (applyIC):
                x = 2.52E12* (1+z)**0.6 * Astar**2.2 * e_b**1.2 * E52**(-0.4) * t_jet**(-1.6) * jeteffect * (1.+Y(params))
            else:
                x = 2.52E12* (1+z)**0.6 * Astar**2.2 * e_b**1.2 * E52**(-0.4) * t_jet**(-1.6) * jeteffect
        elif b == 11:            
            if (applyIC):
                x = 2.34E10 * (1+z)** (-1.5) * Astar** (-2.) * e_b** (-1.5) * E52** (0.5) * t_jet**(0.5) \
                    * jeteffect * nreffect * (1.+Y(params))**(-2.)
            else:
                x = 2.34E10 * (1+z)** (-1.5) * Astar** (-2.) * e_b** (-1.5) * E52** (0.5) * t_jet**(0.5) \
                    * jeteffect * nreffect
            
        else:
            raise ValueError('Invalid break, b = %d' % (b))
    else:
        raise ValueError('Invalid index k = %d' % (k))
    return x

def F_b_ext(params, b = 1, t = 1.0):
    """Extrapolated normalising flux density, aka power law normalisation, for parameters 'params', break number, 'b' and at time, t (days) """
    
    k = params.k
    p = params.p
    z = params.z
    zeta = params.zeta
    e_e = params.e_e / zeta
    e_b = params.e_b / zeta
    n_0 = params.n_0 * zeta
    Astar = params.Astar * zeta
    E52 = params.E52 * zeta
    t_jet = params.t_jet
    t_nr  = params.t_nr
    dL28  = params.dL28
    
    from numpy import e
    
    sjf = s_jetf(params,b)    # This restriction is needed since s_jet is not defined otherwise in this code
    jeteffect = ((t / t_jet)**(-1*sjf*fluxev1(params,b)) + \
                (t / t_jet)**(-1*sjf*fluxev2(params,b)))**(-1/sjf) # jet break        
            
    if b in (1,2,3,4,5):          
        s_srf = s_nrf(params, b) # non-relativistic transition sharpness        
        nreffect  = (1+(t/t_nr)**(s_srf*(fluxev2(params,b)-fluxev3(params,b))))**(-1./s_srf) # non-relativistic transition
        
    if k == 0:
        if b == 1:
            x = 0.647* (p-1.)** 1.2 /((3.*p-1.)*(3.*p+2.)** 0.2) \
                    * (1.+z)** (0.5) * e_ebar(params)** (-1) \
                    * (e_b** 4. * n_0** 7. * E52** 9.)** (0.1) \
                    * t_jet** 0.5 * dL28** (-2.)\
                    * jeteffect * nreffect
        elif b == 2:
            x = 9.93*(p+0.14)\
                    *(1.+z) * (e_b * n_0)** 0.5 * E52 * dL28** (-2.)\
                    * jeteffect * nreffect
        elif b == 3:
            x = 4.68*e** (4.82*(p-2.5)) *1E3\
                    *(1.+z)** ((p+1.)/2.) * e_ebar(params)** (p-1.) \
                    * e_b** (p-0.5) * n_0** (p/2.) * E52** ((p+1.)/2.)\
                    * t_jet** ((1.-p)/2.) * dL28** (-2) \
                    * jeteffect * nreffect * (1.+Y(params))**(params.p-1.)
        elif b == 4:
            x = 3.72*(p-1.79)*1E15\
                    *((1.+z)**7. * n_0** (-1.) * E52** 3. * t_jet** (-5.))**0.5\
                    * e_ebar(params)** (5.) * e_b * dL28** (-2)\
                    * jeteffect * nreffect
        elif b == 5:
            x = 20.8*(p-1.53)\
                    *e** (2.56*p) \
                    * ((1.+z)** (7.*p+3.) *e_b** (2.*p+3.) * E52** (3.*p+7.) * n_0**(6.-p)\
                    / (e_ebar(params)** (10.*(1.-p)) * t_jet** (5.*(p-1.)))) \
                    **(1./(2.*(p+4.))) \
                    *dL28** (-2)\
                    * jeteffect * nreffect
        elif b == 6:
            x = 76.9*(p-1.08)\
                    *e** (2.06*p) \
                    * ((1.+z)** (7.*p+5.) *e_b** (2.*p-5.) * E52** (3.*p+5.)
                    / (e_ebar(params)** (10.*(1.-p)) *n_0**p * t_jet**(5.*(p-1.)))) \
                    **(1./(2.*(p+5.))) \
                    *dL28** (-2) * jeteffect * (1.+Y(params))**(-5./(params.p+5.))
        elif b == 7:
            x = 5.27E-3 * ((3.*p-1.)/(3.*p+2))**2.2\
                    *(1.+z)**(-0.1) * e_ebar(params)** (-2.2) * e_b** (-0.8)\
                    *n_0** 0.1 * E52** 0.3 * t_jet**1.1\
                    *dL28** (-2) \
                    * jeteffect * (1.+Y(params))**(-6./5.)
        elif b == 8:
            x = 154 * (1.+z) * e_b** (-0.25) * n_0** (-1./12.) * E52** (2./3.)\
                    *dL28** (-2) * jeteffect * (1.+Y(params))**(-5./6.)
        elif b == 9:
            x = 0.221 * (6.27-p) * (1+z)** (0.5) * e_ebar(params)** (-1.)\
                    * e_b** (-0.5) * (E52*t_jet)** 0.5\
                    *dL28** (-2) * jeteffect * (1.+Y(params))**(-1.)
        elif b == 10:
            x = 3.72 * (1.+z) * e_b** 1.4 * n_0** 1.2 * E52** 1.4\
                    *dL28** (-2)\
                    * jeteffect * (1.+Y(params))
        elif b == 11:
            x = 28.4 * (1.+z) * e_b** 0.5 * n_0** 0.5 * E52\
                    *dL28** (-2) * jeteffect
        else:
            raise ValueError('Invalid break, b = %d' % (b))
    elif k == 2:
        if b == 1:
            x = 9.19* (p-1.)** 1.2 /((3.*p-1.)*(3.*p+2.)** 0.2) \
                * (1.+z)** (1.2) * e_ebar(params)** (-1) \
                * (e_b** 2. * Astar** 7. * E52 * t_jet** (-1.))** (0.2)\
                * dL28** (-2.)\
                * jeteffect * nreffect
        elif b == 2:
            x = 76.9*(p+0.12)\
                    *((1.+z)** 3. * e_b * E52 * t_jet** (-1.))** 0.5 \
                    * Astar * dL28** (-2.)\
                    * jeteffect * nreffect
        elif b == 3:
            x = 8.02*1E5* e** (7.02*(p-2.5)) \
                    *(1.+z)** (p+0.5) * e_b** (p-0.5) \
                    * Astar** (p) * E52** (0.5) * t_jet** (0.5-p)\
                    * e_ebar(params)** (p-1.) * dL28** (-2)\
                    * jeteffect * nreffect * (1.+Y(params))**(params.p-1.)
        elif b == 4:
            x= 3.04*(p-1.79)*1E15\
                    *(1.+z)** 3. * e_ebar(params)** 5. \
                    * e_b * Astar** (-1.) * E52** 2.\
                    * t_jet** (-2.) * dL28** (-2.)\
                    * jeteffect * nreffect
        elif b == 5:
            x = 158*(p-1.48)\
                    *e** (2.24*p) \
                    * ((1.+z)** (6.*p+9.) *e_b** (2.*p+3.) * E52** (4.*p+1.) \
                    / (e_ebar(params)** (10.*(1.-p)) *Astar**(2.*(p-6.)) \
                    * t_jet** (4.*p+1.))) \
                    **(1./(2.*(p+4.)))\
                    *dL28** (-2)\
                    * jeteffect * nreffect
        elif b == 6:
            x = 78.6*(p-1.12)\
                    *e** (1.89*p) \
                    * ((1.+z)** (6.*p+5.) *e_b** (2.*p-5.) * E52** (4.*p+5.) \
                    / (e_ebar(params)** (10.*(1.-p)) *Astar**(2.*p) * t_jet**(4.*p-5.))) \
                    **(1./(2.*(p+5.)))\
                    *dL28** (-2) * jeteffect * (1.+Y(params))**(-5./(params.p+5.))
        elif b == 7:
            x = 3.76E-3 * ((3.*p-1.)/(3.*p+2))**2.2\
                    * e_ebar(params)** (-2.2) * e_b** (-0.8)\
                    *Astar** 0.2 * E52** 0.2 * t_jet\
                    *dL28** (-2) * jeteffect * (1.+Y(params))**(-6./5.)
        elif b == 8:
            x = 119 * (1.+z)**(11./12.) * e_b** (-0.25) * Astar** (-1./6.) * E52** 0.75\
                    *dL28** (-2) * t_jet** (1./12.) * jeteffect * (1.+Y(params))**(-5./6.)
        elif b == 9:
            x = 0.165 * (7.14-p) * (1+z)** (0.5) * e_ebar(params)** (-1.)\
                    * e_b** (-0.5) * (E52*t_jet)** 0.5\
                    *dL28** (-2)\
                    * jeteffect * (1.+Y(params))**(-1.)
        elif b == 10:
            x = 2.08E3 * (1.+z)**2.2 * e_b**1.4 * Astar**2.4 * E52**0.2 * dL28**(-2) * t_jet**(-1.2) * jeteffect * (1.+Y(params))
        elif b == 11:
            x = 4.37E2 * (1.+z)**1.5 * e_b**0.5 * Astar * E52**0.5 * dL28**(-2) * t_jet**(-0.5)\
                    * jeteffect
        else:
            raise ValueError('Invalid break, b = %d' % (b))
    else:
        raise ValueError('Invalid index k = %d' % (k))
    return x

def F_b (f, params, b = 1, t = 1.0):
    """Smoothly-connecting flux density at intersection of two powers laws at break number, 'b', for parameters 'params', and at time, t (days) """
    
    k = params.k
    p = params.p
    z = params.z
    e_e = params.e_e / params.zeta
    e_b = params.e_b / params.zeta
    n_0 = params.n_0 * params.zeta
    Astar = params.Astar * params.zeta
    E52 = params.E52 * params.zeta
    t_jet = params.t_jet
    t_nr  = params.t_nr

    from numpy import e
    f_break = f_b(params, b, t)
    s_break = s(params, b)
    beta1   = gsbeta1(params, b)
    beta2   = gsbeta2(params, b)
    Norm    = F_b_ext(params, b, t)
    if b == 4:
        phi4 = f/f_b(params,4,t)
        F = Norm*(phi4**2*e**(-1.*s_break*phi4**(2./3.))+phi4**2.5)
    elif b in (range(1,4) + range(5,12)):
        F = Norm* ((1.0*f/f_break)**(-1.*s_break*beta1)\
                + (1.0*f/f_break)**(-1.*s_break*beta2))\
                ** (-1./s_break)
    else:
        raise ValueError('Invalid break, b = %d' % (b))
    return F

def F_b_tilde(f, params, b = 1, t = 1.0):
    """Auxiliary function for spectrum-building; defined at intersection of two powers laws at break number, 'b', for parameters 'params', and at time, t (days) """
    
    k = params.k
    p = params.p
    z = params.z
    e_e = params.e_e / params.zeta
    e_b = params.e_b / params.zeta
    n_0 = params.n_0 * params.zeta
    Astar = params.Astar * params.zeta
    E52 = params.E52 * params.zeta
    t_jet = params.t_jet
    t_nr  = params.t_nr

    f_break = f_b(params, b, t)
    s_break = s(params, b)
    beta1   = gsbeta1(params, b)
    beta2   = gsbeta2(params, b)
    return (1+(1.0*f/f_break)**(s_break*(beta1-beta2)))**(-1./s_break)

def spect(f, params, t = 1.0):
    """Returns a full spectrum, given frequency list aka x-axis values, 'f', parameters 'params', and at time, t (days) """

    i = specnum(params,t)

    if i == 1:
        return F_b(f, params, 1, t) * F_b_tilde(f, params, 2, t) * F_b_tilde(f, params, 3, t)
    elif i == 2:
        return F_b(f, params, 4, t) * F_b_tilde(f, params, 5, t) * F_b_tilde(f, params, 3, t)
    elif i == 3:
        return F_b(f, params, 4, t) * F_b_tilde(f, params, 6, t)
    elif i == 4:
        return F_b(f, params, 7, t) * F_b_tilde(f, params, 8, t) * F_b_tilde(f, params, 9, t)
    elif i == 5:
        return F_b(f, params, 7, t) * F_b_tilde(f, params, 10, t)\
            * F_b_tilde(f, params, 11, t) * F_b_tilde(f, params, 9, t)
    else:
        raise ValueError('Invalid spectrum, i = %d' % (i))

def lightcurve(f, params, t):
    """Returns a light curve, given frequency, 'f' and parameters 'params', as a function of time, t (days) """
    import numpy as np

    # Create variable to store the flux densities in
    lc = np.ones(len(t))
    for i in range(len(t)):
        specnumber = specnum(params, t[i])
        print specnumber
        lc[i] = spect(f, params, t[i])
    return lc

def specnum_pure(params, t = 1.0):
    """Predicts the correct spectrum number from Fig. 1 of Granot and Sari, given parameters 'params', and at time, t (days)"""

    k = params.k
    p = params.p
    z = params.z
    e_e = params.e_e / params.zeta
    e_b = params.e_b / params.zeta
    n_0 = params.n_0 * params.zeta
    Astar = params.Astar * params.zeta
    E52 = params.E52 * params.zeta
    t_jet = params.t_jet
    t_nr  = params.t_nr

    import numpy as np

    # Create array of break frequencies
    nu_b = np.ones(11)
    for i in range(11):
        nu_b[i] = f_b(params, i+1, t)
    
    # Determine the order of the break frequencies
    nu_order=nu_b.argsort()+1
    
    # Figure out which spectrum this corresponds to
    if (k == 0):
        decider = n_0 * E52** (4./7.) * e_b** (9./7.) * (1.+Y(params))**(10./7.)
        if (decider <= 18.0):
            if (np.where(nu_order == 1) < np.where(nu_order == 2) \
                    < np.where(nu_order == 3)):
                spnum = 1
            elif (np.where(nu_order == 4) < np.where(nu_order == 5) \
                    < np.where(nu_order == 3)):
                spnum = 2
            elif ( np.where(nu_order == 7) < np.where(nu_order == 10) \
                     < np.where(nu_order == 11) < np.where(nu_order == 9)):
                spnum = 5
            else:
                spnum = 0
                raise RuntimeError('Ordering of break frequencies \
                            does not match theory')

        if (decider > 18.0):
            if (np.where(nu_order == 4) < np.where(nu_order == 5) \
                    < np.where(nu_order == 3)):
                spnum = 2
            elif (np.where(nu_order == 4) < np.where(nu_order == 6)):
                spnum = 3
            elif (np.where(nu_order == 7) < np.where(nu_order == 8) \
                    < np.where(nu_order == 9)):
                spnum = 4
            else:
                spnum = 0
                raise RuntimeError('Ordering of break frequencies \
                            does not match theory')

    elif (k == 2):
        decider = Astar*e_ebar(params)** (-1.) * E52** (-3./7.) * e_b** (2./7.) * (1.+Y(params))**(3./7.)
        if (decider < 100):
            if (np.where(nu_order == 1) < np.where(nu_order == 2) \
                    < np.where(nu_order == 3)):
                spnum = 1
            elif (np.where(nu_order == 4) < np.where(nu_order == 5) \
                    < np.where(nu_order == 3)):
                spnum = 2
            elif (np.where(nu_order == 7) < np.where(nu_order == 8) \
                    < np.where(nu_order == 9)):
                spnum = 4
            elif ( np.where(nu_order == 7) < np.where(nu_order == 10) \
                     < np.where(nu_order == 11) < np.where(nu_order == 9)):
                spnum = 5
            else:
                spnum = 0
                raise RuntimeError('Ordering of break frequencies \
                            does not match theory')
        elif (decider >= 100):
            if (np.where(nu_order == 4) < np.where(nu_order == 5) \
                    < np.where(nu_order == 3)):
                spnum = 2
            elif (np.where(nu_order == 4) < np.where(nu_order == 6)):
                spnum = 3
            elif (np.where(nu_order == 7) < np.where(nu_order == 8) \
                    < np.where(nu_order == 9)):
                spnum = 4
            else:
                spnum = 0
                raise RuntimeError('Ordering of break frequencies \
                            does not match theory')
    else:
        spnum = 0
        raise ValueError('Invalid Value, k = %d' %((k)))
    return spnum

def specnum(params, t = 1.0):
    """Predicts the correct spectrum number from Fig. 1 of Granot and Sari, given parameters 'params', and at time, t (days)"""

    k = params.k
    p = params.p
    z = params.z
    e_e = params.e_e / params.zeta
    e_b = params.e_b / params.zeta
    n_0 = params.n_0 * params.zeta
    Astar = params.Astar * params.zeta
    E52 = params.E52 * params.zeta
    t_jet = params.t_jet
    t_nr  = params.t_nr

    import numpy as np
    import gs_equalities as gseq
    # Figure out which spectrum this corresponds to
    if (k == 0):
        decider = n_0 * E52** (4./7.) * e_b** (9./7.) * (1.+Y(params))**(10./7.)
        if (decider <= 18.0):
            crosspoints5to1=[gseq.fcross(params, 7, 10,-10,10), gseq.fcross(params, 2, 3,-10,10), gseq.fcross(params, 9, 11,-10,10)]
            s5_end = min(crosspoints5to1)
            s1_beg = max(crosspoints5to1)
            crosspoints1to2=[gseq.fcross(params, 1, 2,-10,10), gseq.fcross(params, 4, 5,-10,10)]
            s1_end = min(crosspoints1to2)
            s2_beg = max(crosspoints1to2)

            if   (t < s5_end):
                spnum = 5
            elif (t < s1_beg):
                spnum = 501
            elif ((t >= s1_beg) and (t <= s1_end)):
                spnum = 1
            elif (t < s2_beg):
                spnum = 102
            elif (t >= s2_beg):
                spnum = 2
            else:
                spnum = 0
                raise RuntimeError('Cannot figure out which spectrum we are in')

        if (decider > 18.0):
            crosspoints4to3=[gseq.fcross(params, 7, 8,-10,10), gseq.fcross(params, 7, 9,-10,10), gseq.fcross(params, 4, 6,-10,10)]
            s4_end = min(crosspoints4to3)
            s3_beg = max(crosspoints4to3)
            s3_end = gseq.fcross(params,3,5,-10,10)
    
            if   (t < s4_end):
                spnum = 4
            elif (t < s3_beg):
                spnum = 403
            elif ((t >= s3_beg) and (t <= s3_end)):
                spnum = 3
            elif (t > s3_end):
                spnum = 2
            else:
                spnum = 0
                raise RuntimeError('Cannot figure out which spectrum we are in')

    elif (k == 2):
        
        decider = Astar*e_ebar(params)** (-1.) * E52** (-3./7.) * e_b** (2./7.) * (1.+Y(params))**(3./7.)
        if (decider > 100):
            s4_end=9.3*(1.+z)* e_b** (9./7) * Astar**2. * E52** (-3./7)
            s5_beg=s4_end            
            crosspoints5to1=[gseq.fcross(params, 7, 10,-10,10), gseq.fcross(params, 2, 3,-10,10), gseq.fcross(params, 9, 11,-10,10)]
            s5_end = min(crosspoints5to1)
            s1_beg = max(crosspoints5to1)
            crosspoints1to2=[gseq.fcross(params, 1, 2,-10,10), gseq.fcross(params, 4, 5,-10,10)]
            s1_end = min(crosspoints1to2)
            s2_beg = max(crosspoints1to2)

            if   (t < s4_end):
                spnum = 4
            elif (t < s5_beg):
                spnum = 405 # This will never happen, because s4_end = s5_beg
            elif ((t >= s5_beg) and (t <= s5_end)):
                spnum = 5
            elif ((t >  s5_end) and (t <  s1_beg)):
                spnum = 501
            elif ((t >= s1_beg) and (t <= s1_end)):
                spnum = 1
            elif ((t >  s1_end) and (t <  s2_beg)):
                spnum = 102
            elif (t >= s2_beg):
                spnum = 2
            else:
                spnum = 0
                raise RuntimeError('Cannot figure out which spectrum we are in')
                            
        elif (decider <= 100):
            crosspoints4to3=[gseq.fcross(params, 7, 8,-10,10), gseq.fcross(params, 7, 9,-10,10), gseq.fcross(params, 4, 6,-10,10)]
            s4_end = min(crosspoints4to3)
            s3_beg = max(crosspoints4to3)
            s3_end = gseq.fcross(params,3,5,-10,10)

            if   (t < s4_end):
                spnum = 4
            elif (t < s3_beg):
                spnum = 403
            elif ((t >= s3_beg) and (t <= s3_end)):
                spnum = 3
            elif (t > s3_end):
                spnum = 2
            else:
                spnum = 0
                raise RuntimeError('Cannot figure out which spectrum we are in')
    else:
        spnum = 0
        raise ValueError('Invalid Value, k = %d' %((k)))
    return spnum

def dens(params, R):
    '''Returns the density in g/cm^3 at distance R (in cm) in the burster frame'''
    k = params.k    
    import numpy as np
    if (k == 0):
        m_p = 1.673e-24 # Proton mass in g
        return params.n_0*m_p*np.ones_like(R)
    elif (k == 2):
        return params.Astar*5e11*R**(-2.)
    else:
        raise ValueError('k must be 0 or 2 for computation of density')
    
def h(params):
    '''Break frequencies number 7, 9, 10, and 11 are related through
       h(p) = (nu10 / nu7) * (nu11 / nu9)**(4/5). 
       This function returns h(p)'''
    p = params.p
    k = params.k
    if (k == 0):
        raise NotImplemented('k = 0 not implemented yet')
    if (k == 2):
        return 1.13*(3*p+2.)**1.6/((3*p-1)**1.6 * (p-0.31)**0.8)
