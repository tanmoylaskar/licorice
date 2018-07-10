def Gammabeta_fit(k, t, t_jet, t_nr, t_sph, Gammajet, sj1=50., sj2=20., s_nr=20., s_sph=10., 
                  G2=1.,G3=1.,kinkcor=1.):
    #sj1 = 12.5*k+25.
    pk = Pk(k)
    qk = Qk(k)
    G2 = 1.
    Gamma_ev1 = -0.5*(3.-k)/(4.-k)                  # Blandford-McKee
    Gamma_ev2 = -0.5*(3.-k)/(4.-k)*(1.+qk/pk)*G2    # post-jet break steep decline
    Gamma_ev3 = -0.5/(1.+1./((3.-k)*(1.+qk/pk)))*G3 # post-jet break decline rate  
    Gamma_ev4 = -1.0/(1.+2./((3.-k)*(1.+qk/pk)))    # post NR-transition decline rate
    Gamma_ev5 = -(3.-k)/(5.-k)                      # spherical expansion
    
    # Calculate the kinkscale
#    if (k == 0):   
#        kinkscale = 2.3
#    elif (k == 2):
#        kinkscale = 2.3
    Gl = -Gamma_ev1; Gr = -Gamma_ev3
    kinkscale = ((2*Gr+1.)/(2*Gl+1.))**((2*Gl+1.)/(2*Gr-2*Gl)) * kinkcor
    t_jet2    = t_jet * kinkscale

    # Jet break
    jeteffect   = ((t / t_jet)**(-1*sj1*Gamma_ev1) + \
                   (t / t_jet)**(-1*sj1*Gamma_ev2))**(-1/sj1)
    sign_jet    = sign(Gamma_ev2 - Gamma_ev3)
    jeteffect2  = (1.+(t/t_jet2)**(sj2*sign_jet*(Gamma_ev2-Gamma_ev3)))**(-1./(sj2*sign_jet))

    # Non-relativistic transition
    sign_nr     = sign(Gamma_ev3 - Gamma_ev4)
    nreffect    = (1.+(t/t_nr)**(s_nr*sign_nr*(Gamma_ev3-Gamma_ev4)))**(-1./(s_nr*sign_nr))
    
    # Spherical expansion
    sign_sph    = sign(Gamma_ev4 - Gamma_ev5)
    sph_effect  = (1.+(t/t_sph)**(s_sph*sign_sph*(Gamma_ev4-Gamma_ev5)))**(-1./(s_sph*sign_sph))
    return Gammajet*jeteffect*jeteffect2*nreffect*sph_effect

def Gammabeta_fit_k(t, tjet, tnr, tsph, Gammajet, sj1, sj2, s_nr, s_sph, G2, G3, kinkcor):
    return Gammabeta_fit(k, t, tjet, tnr, tsph, Gammajet, sj1, sj2, s_nr, s_sph, G2, G3, kinkcor)

def runGammabetafit():
    if (k == 0):
        sel = (u < G0/1e4)
    elif (k == 2):
        sel = (u < G0/1e5)# & (u > 3e-6)

    x = tobs[sel]; y = u[sel]

    par0 = [tjet_theory,max(tnr_theory/100.,tjet_theory*1.05),
            tsph_theory,1./(Qk(k)*theta0),50.,1.0,0.2, 1.,1.0,1.0, 1.0]
    lbounds = (0.8*tjet_theory,tjet_theory*1.05,tjet_theory*1e2,1e-6, 10.,0.1,0.05,0.1,0.6,0.6,0.7)
    ubounds = (1.2*tjet_theory,max(tjet_theory*1e3,tnr_theory), 
               max(tjet_theory*1e5,tsph_theory),1e+6,100.,40.,40., 40.,1.2,1.2, 1.2)

    if (theta0 < 0.3):
        ''' Cant seem to be able to fit for larger angles'''
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(Gammabeta_fit_k, x, y, sigma=y*0.001, p0=par0, bounds=(lbounds,ubounds))
        tjet, tnr, tsph, Gtjet, sj1, sj2, s_nr, s_sph, G2, G3, kinkcor = popt
        #print popt
        yf =  Gammabeta_fit(k, x, *popt)

    if (k == 0):
        tmin = 1e-4
    elif (k == 2):
        tmin = 1e-6
    tmax = 1e6
    mask = (x >= tmin) & (x <= tmax)

    if (plotfig and theta0 < 0.3):
        from matplotlib import gridspec
        lw = 3
        figure()
        gs = gridspec.GridSpec(2,1,height_ratios=[4,3])
        ax0 = plt.subplot(gs[0])
        #subplot(211)
        loglog(x[mask],y[mask],lw=lw,color='k')
        loglog(x[mask],yf[mask],lw=lw*0.5,ls='--',color='orange')
        ylabel("Fluid 4-velocity", fontsize=fontsize)
        #xlabel("Observer time (days)")
        axvline(tjet,color='k',ls='-')
        axhline(1.0,color='grey',ls=':',lw=2)
        #axvline(tjet*2.0,color='k',ls='-.')
        #axvline(tnr, color='k',ls='--')
        #axvline(tsph,color='k',ls=':')
        
        axvline(tjet, color='C3',ls='-')
        axvline(tnr, color='C2',ls='-')
        axvline(tsph, color='C4',ls='-')
        #axvline(tjet*kinkscale,color='C5',ls='-')
            
        ax1 = plt.subplot(gs[1],sharex = ax0)
        #subplot(212)
        plot(x[mask],(1.-yf[mask]/y[mask])*100.)
        xscale('log')
        ylabel("Fractional error (%)",fontsize=0.7*fontsize)
        xlabel("Observer time (days)",fontsize=fontsize)
        axvline(tjet_theory, color='C3',ls='-')
        axvline(tnr_theory, color='C2',ls='-')
        axvline(tsph_theory, color='C4',ls='-')
        #axvline(tjet*2.0,color='k',ls='-.')
        #axvline(tnr, color='k',ls='--')
        #axvline(tsph,color='k',ls=':')
        plt.tight_layout(True)
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        plt.show()
        savefig("uavg_fit_"+profile+figtype)

def calcdiagnostics(k):
    if (k == 0):
        AA     = A*1.67e-24         
        tjet_oldtheory = 120.*(E52/A)**(1./3.)\
                             *theta0**(8./3.)*(1.+z) # Sari, Piran, Halpern calculation of tjet
    elif (k == 2):
        AA     = A*5e11
        tjet_oldtheory = 625.*(E52/A)*theta0**4.*(1.+z) # Chevalier and Li (2000) calculation of tjet    

    # The following is a factor of ~ 1.23 too small
    tjet = (1.+z)*(  (3.-k)**(4.-k)*E52*1e52*(Qk(k)*theta0)**(8.-2*k) \
                   / (2**(6.-k)*(4.-k)**(3.-k)*(17.-4*k)**(3.-k)*pi*AA*3e10**(5.-k) ) )**(1./(3.-k)) / 86400.

    delta_s = 0.5*(3.-k)*(1.+Qk(k)/Pk(k))
    delta_L = 0.5*(3.-k)/(4.-k)
    delta_R = 0.5/(1.+1./((3.-k)*(1.+Qk(k)/Pk(k))))
    f_kink  = ((2*delta_L+1.)/(2*delta_R+1))**(delta_R/(2*delta_R+1.))
    f_k     = ((17.-4*k)/(3.-k))**0.5    
    unr     = (1.+delta_s)/(2.+4*delta_s)
    tnr     = tjet*(unr*Qk(k)*theta0/(f_kink*f_k))**(-2.0*(1.+ 1./((3.-k)*(1.+Qk(k)/Pk(k)))))
    #tnr    = tjet*(Qk(k)*theta0)**(-2.0*(1.+ 1./((3.-k)*(1.+Qk(k)/Pk(k)))))
    delta_theta = Qk(k)*(3.-k)/Pk(k)
    tsph    = tnr*theta0**(-1./delta_theta)*(tjet/tnr)**(1./(1.+2*delta_theta))

    # Calculation of tsph
    Wsquare = ((17.-k)/(16.*pi))*(7./6.)**2/(1.+get_Ak_from_k(k))
    tsedov = (1.+z)*(Wsquare*E52*1e52/(AA*(3e10)**2))**(1./(3.-k))/3e10 / 86400. # Sedov time in days
    tsph = tsedov 
    
    #print tjet, tnr, tsph
    return tjet, tnr, tsph, tjet_oldtheory

def calcGS02spectrum_f_b(params,tobs):
    import gstable2
    nu1  = gstable2.f_b(params,1,tobs)
    nu2  = gstable2.f_b(params,2,tobs)
    nu3  = gstable2.f_b(params,3,tobs)
    nu5  = gstable2.f_b(params,5,tobs)
    nu6  = gstable2.f_b(params,6,tobs)
    nu7  = gstable2.f_b(params,7,tobs)
    nu8  = gstable2.f_b(params,8,tobs)
    nu10 = gstable2.f_b(params,10,tobs)
    return nu1, nu2, nu3, nu5, nu6, nu7, nu8, nu10

def calcGS02spectrum_F_b_ext(params,tobs):
    import gstable2
    F2    = gstable2.F_b_ext(params, 2, tobs)
    F11   = gstable2.F_b_ext(params, 11, tobs)
    return F2, F11
