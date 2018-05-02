import sys
sys.path.append('../../GAMMA/')
import gstable2
from matplotlib import gridspec

gstable2.applyIC = False

plotfig = True
figtype = '.eps'
fontsize = 18
execfile("anly.py")

params = gstable2.Physparams(\
        k     = 0,
        p     = 2.1,
        z     = 1.,
        e_e   = 0.1,
        e_b   = 0.01,
        n_0   = 1.,
        Astar = 0.1,
        E52   = 1e+3,
        t_jet = 1.0,
        A_B   = 0.0,
        zeta  = 1.0)

params_161219B = gstable2.Physparams(\
        k     = 0,
        p     = 2.1,
        z     = 0.1475,
        e_e   = 0.925,
        e_b   = 0.05139,
        n_0   = 3.6e-4,
        Astar = 0.1,
        E52   = 0.47,
        t_jet = 31.49,
        A_B   = 0.036,
        zeta  = 1.0)

params = params_161219B
k      = params.k
G0     = 1e7
theta0 = 0.26 #10.*np.pi/180.#0.05
E52    = params.E52
z      = params.z

if (k==0):
    profile = 'ISM'
    rmin   = 1e-7
else:
    profile = 'wind'
    rmin   = 1e-20

def calcparams(k):
    if (k == 0):
        A      = params.n_0
        AA     = A*1.67e-24 
        rmax   = 1e3
        #tjet   = 7.9*(1.+z)*(E52/A)**(1./3.)*(Qk(k)*theta0)**(8./3.)
        tjet_oldtheory = 120.*(E52/A)**(1./3.)\
                             *theta0**(8./3.)*(1.+z) # Sari, Piran, Halpern calculation of tjet
    elif (k == 2):
        A      = params.Astar
        AA     = A*5e11
        rmax   = 1e15
        #tjet   = 9.5*(1.+z)*(E52/A)*(Qk(k)*theta0)**4
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
    return A, rmax, tjet, tnr, tsph, tjet_oldtheory

A, rmax, tjet, tnr, tsph, tjet_oldtheory = calcparams(k)
tjet_theory = tjet
tnr_theory  = tnr
tsph_theory = tsph

# Fit u(tobs) with power laws
data = calcall(k=k,E52=E52,A=A,G0=G0,theta0=theta0,rmin=rmin,rmax=rmax,z=z,N=1000,physical=True,modeltype='DL')
u     = data['upeak']
theta = data['theta']
if (k == 0):
    sel = (u < G0/1e4)
elif (k == 2):
    sel = (u < G0/1e5)# & (u > 3e-6)
x = data['tobs'][sel]; y = u[sel]

def Gammabeta_fit_k(t, tjet, tnr, tsph, Gammajet, sj1, sj2, s_nr, s_sph, G2, G3, kinkcor):
    return Gammabeta_fit(k, t, tjet, tnr, tsph, Gammajet, sj1, sj2, s_nr, s_sph, G2, G3, kinkcor)

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
    lw = 3
    figure()
    gs = gridspec.GridSpec(2,1,height_ratios=[4,3])
    ax0 = plt.subplot(gs[0])
#    subplot(211)
    loglog(x[mask],y[mask],lw=lw,color='k')
    loglog(x[mask],yf[mask],lw=lw*0.5,ls='--',color='orange')
    ylabel("Fluid 4-velocity", fontsize=fontsize)
#    xlabel("Observer time (days)")
    axvline(tjet,color='k',ls='-')
    axhline(1.0,color='grey',ls=':',lw=2)
#    axvline(tjet*2.0,color='k',ls='-.')
#    axvline(tnr, color='k',ls='--')
#    axvline(tsph,color='k',ls=':')
    
    axvline(tjet, color='C3',ls='-')
    axvline(tnr, color='C2',ls='-')
    axvline(tsph, color='C4',ls='-')
    #axvline(tjet*kinkscale,color='C5',ls='-')
        
    ax1 = plt.subplot(gs[1],sharex = ax0)
#    subplot(212)
    plot(x[mask],(1.-yf[mask]/y[mask])*100.)
    xscale('log')
    ylabel("Fractional error (%)",fontsize=0.7*fontsize)
    xlabel("Observer time (days)",fontsize=fontsize)
    axvline(tjet_theory, color='C3',ls='-')
    axvline(tnr_theory, color='C2',ls='-')
    axvline(tsph_theory, color='C4',ls='-')
#    axvline(tjet*2.0,color='k',ls='-.')
#    axvline(tnr, color='k',ls='--')
#    axvline(tsph,color='k',ls=':')
    plt.tight_layout(True)
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=.0)
    plt.show()
    savefig("uavg_fit_"+profile+figtype)

tobs = data['tobs']; tlab = data['t']/(1.+params.z); Gamma = data['gamma']; R = data['r']
params.t_jet = tjet_oldtheory
params.updatejetparams()
#params.t_nr = tnr_theory
