from gstable2 import *
import numpy as np

pl = 2.

def checkvalid(params):
    k = params.k
    p = params.p
    z = params.z
    e_e = params.e_e
    e_b = params.e_b
    n_0 = params.n_0
    Astar = params.Astar
    E52 = params.E52
    t_jet = params.t_jet
    t_nr = params.t_nr
    A_B  = params.A_B

    if ((k not in (0,2))\
     or (p <= 2) or (p > 3.45)\
     or (z <= 1E-8)\
     or (e_e <= 0) or (e_b <= 0)\
     or (n_0 <= 0) or (Astar <= 0)\
     or (E52 <= 0)\
     or (t_jet <= 0)\
     or (t_nr <= 0)\
     or (A_B < 0)\
     or (np.isnan(np.array([k,p,z,e_e,e_b,n_0,Astar,E52,t_jet,t_nr,A_B])).sum() > 0)):
         return False
    else:
        return True             

gsspect = {
    1: (lambda f, params, t: F_b(f, params, 1, t) * F_b_tilde(f, params, 2, t) * F_b_tilde(f, params, 3, t)),
    2: (lambda f, params, t: F_b(f, params, 4, t) * F_b_tilde(f, params, 5, t) * F_b_tilde(f, params, 3, t)),
    3: (lambda f, params, t: F_b(f, params, 4, t) * F_b_tilde(f, params, 6, t)),
    4: (lambda f, params, t: F_b(f, params, 7, t) * F_b_tilde(f, params, 8, t) * F_b_tilde(f, params, 9, t)),
    5: (lambda f, params, t: F_b(f, params, 7, t) * F_b_tilde(f, params, 10, t) \
                * F_b_tilde(f, params, 11, t) * F_b_tilde(f, params, 9, t))
}    
             
def fluxes(f, params, hostfluxes, t = 1, extcor_g = 1.0, extcor_h = 1.0, pabscor_g = 1.0, pabscor_h = 1.0, igmtrans = 1.0, galtrans=1.0, filtergrid = None, printweights = False):
    
    import numpy as np
    modelgrid = spectrum(f, params, t, printweights)
    
    if (filtergrid is None):        
        if (np.ndim(f) == 0):
            nfreqs = 1        
        if (np.ndim(t) == 0):
            ntimes = 1
        if (np.ndim(f) == 1):
            nfreqs = len(f)
        if (np.ndim(t) == 1):
            ntimes = len(t)
        if ((np.ndim(f) == 2) & (np.ndim(t) == 2)):
            if (np.shape(f) != np.shape(t)):
                raise ValueError('Shape mismatch: your frequency and time arrays have 2 dimensions each. Therefore they must also have the same shape.')
            ntimes = np.shape(f)[0]
            nfreqs = np.shape(f)[1] # both f and t have the same shape, so can use either
        
        # Set up host flux grid
        if ((ntimes == 1) or (nfreqs == 1)):
            hfg = hostfluxes
        else:
            hfg = hostfluxes[np.newaxis,:].repeat(ntimes,axis=0) # Host flux grid
        return extcor_g*pabscor_g*igmtrans*galtrans*(pabscor_h*extcor_h*modelgrid + hfg)
    
    else:
        import pickle
        import igm
        
        sel        = (filtergrid != '')
        
        if (np.ndim(modelgrid) == 2):
            if ((np.shape(modelgrid)[0]) == 1):
                modelgrid = modelgrid[0]
        
        # Correct model data points without explicit filters, using band-averaged corrections supplied from outside:
        modelgrid[~sel] = extcor_g[~sel]*extcor_h[~sel]*pabscor_h[~sel]*igmtrans[~sel]*modelgrid[~sel] 

        # Calculate the corrections for the others:
        if (np.sum(sel) == 0):
            return modelgrid
        
        if (np.ndim(f) == 0):
            nfreqs = 1
            freqs  = f
        if (np.ndim(t) == 0):
            ntimes = 1
            times  = t
        if (np.ndim(f) == 1):
            nfreqs = len(f)
            freqs  = f[sel]
        if (np.ndim(t) == 1):
            ntimes = len(t)
            times  = t[sel]
        if (np.ndim(f) == 2):
            if (np.shape(f) != np.shape(t)):
                raise ValueError('Shape mismatch: your frequency and time arrays have 2 dimensions each. Therefore they must also have the same shape.')
            ntimes = np.shape(f)[0]
            nfreqs = np.shape(f)[1] # both f and t have the same shape, so can use either
            freqs  = f[sel]
            times  = t[sel]

        newfluxes = np.zeros(np.sum(sel))
        
        if ((nfreqs == 1) and (ntimes == 1)): # Only one frequency        
            f1 = open(filtergrid,'r')
            filt = pickle.load(f1)
            freq = np.array(filt['q'].tolist())
            trans = np.array(filt['trans'].tolist())
            Flux = np.trapz(freq, trans*extcor_g*pabscor_g*igmtrans*galtrans*(pabscor_h*extcor_h*spectrum(freq,params,t,printweights=False)+hostfluxes))/np.trapz(freq, trans) # Returns a single data point
            return Flux
        else:        
            filters    = filtergrid[sel]
            galextcor  = extcor_g[sel]
            hostextcor = extcor_h[sel]
            pabscor_g  = pabscor_g[sel]
            pabscor_h  = pabscor_h[sel]
            
            
            # Set up host flux grid
            if (ntimes == 1):
                hfg = hostfluxes[sel]
            else:
                hfg = hostfluxes[np.newaxis,:].repeat(ntimes,axis=0)[sel] # Host flux grid

            filterfreqs = np.array([])
            filtertrans = np.array([])
            filtertimes = np.array([])
            filterhostf = np.array([])
            filterlen   = np.zeros(len(filters))
            lowerindex  = np.zeros(len(filters))
            upperindex  = np.zeros(len(filters))
            
            for i in np.arange(len(filters)):
                f1 = open(filters[i],'r')
                filt = pickle.load(f1)
                freq = np.array(filt['q'].tolist())
                trans = np.array(filt['trans'].tolist())
                igmtrans = np.e**(-1*igm.tau(2.99792458e8/freq, params.z))
                if (ntimes > 1):
                    Flux = np.trapz(freq, trans*igmtrans*galextcor[i]*pabscor_g[i]*(pabscor_h[i]*hostextcor[i]*spectrum(freq,params,times[i],printweights=False)+hfg[i])) / np.trapz(freq, trans)
                else:
                    Flux = np.trapz(freq, trans*igmtrans*galextcor[i]*pabscor_g[i]*(pabscor_h[i]*hostextcor[i]*spectrum(freq,params,times,printweights=False)+hfg[i])) / np.trapz(freq, trans)
                newfluxes[i] = Flux
            modelgrid[sel] = newfluxes

            # Attempt at speeding it up...
                #filterlen[i] = len(filt['q'])
                #if (i == 0):
                    #lowerindex[i] = 0
                    #upperindex[i] = filterlen[i] - 1
                #else:
                    #lowerindex[i] = upperindex[i-1] + 1
                    #upperindex[i] = lowerindex[i]   + filterlen[i] - 1
                #filterfreqs = np.concatenate((filterfreqs, np.array(filt['q'])))
                #filtertrans = np.concatenate((filtertrans, np.array(filt['trans'])))
                #filtertimes = np.concatenate((filtertimes, np.repeat(times[i],len(filt['q']))))
                #filterhostf = np.concatenate((filterhostf, np.repeat(hfg[i], len(filt['q']))))

            #filterigmtrans = np.e**(-1*igm.tau(2.99792458e8/filterfreqs, params.z))
            
            #fullfilterfreqs = filterfreqs[np.newaxis,:].repeat(ntimes, axis=0)
            #fullfiltertrans = filtertrans[np.newaxis,:].repeat(ntimes, axis=0)
            #fullfiltertimes = filtertimes[np.newaxis,:].repeat(ntimes, axis=0)
            #fullfilterhostf = filterhostf[np.newaxis,:].repeat(ntimes, axis=0)
            #fullfilterigmtrans= filterigmtrans[np.newaxis,:].repeat(ntimes, axis=0)            
            
            #fullspectrum = fullfiltertrans*fullfilterigmtrans*spectrum(fullfilterfreqs, params, filterhostf, fullfiltertimes, printweights = False)

            #for i in np.arange(len(filters)):
                #l = lowerindex[i]
                #u = upperindex[i] + 1 # Otherwise slicing will not return the last element
                ##q = filterfreqs[l:u]
                ##f = fullspectrum[l:u]*galextcor[i]*hostextcor[i]
                ##newfluxes[i] = np.trapz(q, f)
                
        #modelgrid[sel] = newfluxes
        
        return modelgrid
    
def spectrum(f, params, t, printweights = False):
    """Returns a full spectrum, given frequency list aka x-axis values, 'f', parameters 'params', and at time, t (days) """
    #if (np.ndim(f) != np.ndim(t)):
        #raise ValueError('Shape mismatch: frequency and time arrays must have the same number of dimensions')
        
    if (np.ndim(f) == 0):
        nfreqs = 1
    if (np.ndim(t) == 0):
        ntimes = 1
    if (np.ndim(f) == 1):
        nfreqs = len(f)
    if (np.ndim(t) == 1):
        ntimes = len(t)
    if ((np.ndim(f) == 2) & (np.ndim(t) == 2)):
        if (np.shape(f) != np.shape(t)):
            raise ValueError('Shape mismatch: your frequency and time arrays have 2 dimensions each. Therefore they must also have the same shape.')
        ntimes = np.shape(f)[0]
        nfreqs = np.shape(f)[1] # both f and t have the same shape, so can use either

    if (checkvalid(params) == False):        
        grid = np.zeros([ntimes,nfreqs])
        return grid

    edges_calc = edges(params)
    specorder = edges_calc[0]
    decider = edges_calc[1]        
    
    spect1 = gsspect[1](f, params, t)
    spect2 = gsspect[2](f, params, t) 
    spect3 = gsspect[3](f, params, t) 
    spect4 = gsspect[4](f, params, t) 
    spect5 = gsspect[5](f, params, t) 
    
    if (specorder == 0):
        [s5_end, s1_beg, s1_end, s2_beg, t51,t12] = edges_calc[2:]
        w5     = 1./(1.+(t/t51)**pl)
        w1     = (1./(1.+(t/t51)**(-1.*pl))) * (1./(1.+(t/t12)**pl))
        w2     = 1./(1.+(t/t12)**(-1.*pl))        
     
        if (printweights == True):
            print 'k = 0, decider = %f <= 18.0' %(decider)
            print 'ISM Case. Evolution is 5 -> 1 -> 2'
            print 'Spectrum weights at time, %f days:' %(t)
            print 'Spectrum 5: %f\nSpectrum 1: %f\nSpectrum 2: %f' %(w5,w1,w2)

        grbspect = (w5*spect5 + w1*spect1 + w2*spect2)/(w5+w1+w2)

    elif (specorder == 1):
        [s4_end, s3_beg, s3_end, s2_beg, t43, t32] = edges_calc[2:]
        w4     = 1./(1.+(t/t43)**pl)
        w3     = (1./(1.+(t/t43)**(-1.*pl))) * (1./(1.+(t/t32)**pl))
        w2     = 1./(1.+(t/t32)**(-1.*pl))        
     
        if (printweights == True):
            print 'k = 0, decider = %f > 18.0' %(decider)
            print 'ISM Case. Evolution is 4 -> 3 -> 2'
            print 'Spectrum weights at time, %f days:' %(t)
            print 'Spectrum 4: %f\nSpectrum 3: %f\nSpectrum 2: %f' %(w4,w3,w2)

        grbspect = (w4*spect4 + w3*spect3 + w2*spect2)/(w4+w3+w2)

    elif (specorder == 2):
        [s4_end, s5_beg, s5_end, s1_beg, s1_end, s2_beg, t45, t51, t12] = edges_calc[2:]
        w4     = 1./(1.+(t/t45)**pl)
        w5     = (1./(1.+(t/t45)**(-1.*pl))) * (1./(1.+(t/t51)**pl))
        w1     = (1./(1.+(t/t51)**(-1.*pl))) * (1./(1.+(t/t12)**pl))
        w2     = 1./(1.+(t/t12)**(-1.*pl))        
     
        if (printweights == True):
            print 'k = 2, decider = %f < 100.0' %(decider)
            print 'Wind Case. Evolution is 4 -> 5 -> 1 -> 2'
            print 'Spectrum weights at time, %f days:' %(t)
            print 'Spectrum 4: %f\nSpectrum 5: %f\nSpectrum 1: %f\nSpectrum 2: %f' %(w4, w5,w1,w2)

        grbspect = (w4*spect4 + w5*spect5 + w1*spect1 + w2*spect2)/(w4+w5+w1+w2)

    elif (specorder == 3):
        [s4_end, s3_beg, s3_end, s2_beg, t43, t32] = edges_calc[2:]
        w4     = 1./(1.+(t/t43)**pl)
        w3     = (1./(1.+(t/t43)**(-1.*pl))) * (1./(1.+(t/t32)**pl))
        w2     = 1./(1.+(t/t32)**(-1.*pl))
     
        if (printweights == True):
            print 'k = 2, decider = %f > 100.0' %(decider)
            print 'Wind Case. Evolution is 4 -> 3 -> 2'
            print 'Spectrum weights at time, %f days:' %(t)
            print 'Spectrum 4: %f\nSpectrum 3: %f\nSpectrum 2: %f' %(w4,w3,w2)

        grbspect = (w4*spect4 + w3*spect3 + w2*spect2)/(w4+w3+w2)
        
    else:
        raise ValueError('Invalid Value, k = %d' %((k)))

    if np.isinf(grbspect).any():
        print 1/0
    return grbspect

def summary(params):
    """Tells you what's going to happen for these set of parameters'"""

    k = params.k
    p = params.p
    z = params.z
    e_e = params.e_e
    e_b = params.e_b
    n_0 = params.n_0
    Astar = params.Astar
    E52 = params.E52
    t_jet = params.t_jet

    edges_calc = edges(params)
    specorder = edges_calc[0]
    decider = edges_calc[1]

    import gs_equalities as gseq
    from numpy import e
    if (specorder == 0):
        [s5_end, s1_beg, s1_end, s2_beg, t51,t12] = edges_calc[2:]
        
        print 'k = 0, decider = %f <= 18.0' %(decider)
        print 'ISM Case. Evolution is 5 -> 1 -> 2'
        print 'Spectrum 5 ends   at %4.2e days' %(s5_end)
        print 'Spectrum 1 begins at %4.2e days' %(s1_beg)
        print 'Spectrum 1 ends   at %4.2e days' %(s1_end)
        print 'Spectrum 2 begins at %4.2e days' %(s2_beg)
        print 't51 = %4.2e days' %(t51)
        print 't12 = %4.2e days' %(t12)

    elif (specorder == 1):
        [s4_end, s3_beg, s3_end, s2_beg, t43, t32] = edges_calc[2:]
        
        print 'k = 0, decider = %f >= 18.0' %(decider)
        print 'ISM Case. Evolution is 4 -> 3 -> 2'
        print 'Spectrum 4 ends   at %4.2e days' %(s4_end)
        print 'Spectrum 3 begins at %4.2e days' %(s3_beg)
        print 'Spectrum 3 ends   at %4.2e days' %(s3_end)
        print 'Spectrum 2 begins at %4.2e days' %(s2_beg)
        print 't43 = %4.2e days' %(t43)
        print 't32 = %4.2e days' %(t32)
        
    elif (specorder == 2):
        [s4_end, s5_beg, s5_end, s1_beg, s1_end, s2_beg, t45, t51, t12] = edges_calc[2:]
        
        print 'k = 2, decider = %f < 100.0' %(decider)
        print 'Wind Case. Evolution is 4 -> 5 -> 1 -> 2'
        print 'Spectrum 4 ends   at %4.2e days' %(s4_end)
        print 'Spectrum 5 begins at %4.2e days' %(s5_beg)
        print 'Spectrum 5 ends   at %4.2e days' %(s5_end)
        print 'Spectrum 1 begins at %4.2e days' %(s1_beg)
        print 'Spectrum 1 ends   at %4.2e days' %(s1_end)
        print 'Spectrum 2 begins at %4.2e days' %(s2_beg)
        print 't45 = %4.2e days' %(t45)
        print 't51 = %4.2e days' %(t51)
        print 't12 = %4.2e days' %(t12)

    elif (specorder == 3):
        [s4_end, s3_beg, s3_end, s2_beg, t43, t32] = edges_calc[2:]
        
        print 'k = 2, decider = %f > 100.0' %(decider)
        print 'Wind Case. Evolution is 4 -> 3 -> 2'
        print 'Spectrum 4 ends   at %4.2e days' %(s4_end)
        print 'Spectrum 3 begins at %4.2e days' %(s3_beg)
        print 'Spectrum 3 ends   at %4.2e days' %(s3_end)
        print 'Spectrum 2 begins at %4.2e days' %(s2_beg)
        print 't43 = %4.2e days' %(t43)
        print 't32 = %4.2e days' %(t32)

    else:
        raise ValueError('Invalid Value, k = %d' %((k)))

def weights(params, t = 1):
    """Returns a the spectrum weights as a function of time, given parameters 'params'"""
    if (np.ndim(t) == 0):
        ntimes = 1
    if (np.ndim(t) == 1):
        ntimes = len(t)

    k = params.k
    p = params.p
    z = params.z
    e_e = params.e_e / params.zeta
    e_b = params.e_b / params.zeta
    n_0 = params.n_0 * params.zeta
    Astar = params.Astar * params.zeta
    E52 = params.E52 * params.zeta
    t_jet = params.t_jet

    import gs_equalities as gseq
    from numpy import e

    edges_calc = edges(params)
    specorder = edges_calc[0]
    decider = edges_calc[1]
    
    if (specorder == 0):
        [s5_end, s1_beg, s1_end, s2_beg, t51,t12] = edges_calc[2:]        
        w5     = 1./(1.+(t/t51)**pl)
        w1     = (1./(1.+(t/t51)**(-1.*pl))) * (1./(1.+(t/t12)**pl))
        w2     = 1./(1.+(t/t12)**(-1.*pl))

        print 'k = 0, decider = %f <= 18.0' %(decider)
        print 'ISM Case. Evolution is 5 -> 1 -> 2'
        return [w5, w1, w2]

    elif (specorder == 1):
        [s4_end, s3_beg, s3_end, s2_beg, t43, t32] = edges_calc[2:]    
        w4     = 1./(1.+(t/t43)**pl)
        w3     = (1./(1.+(t/t43)**(-1.*pl))) * (1./(1.+(t/t32)**pl))
        w2     = 1./(1.+(t/t32)**(-1.*pl))

        print 'k = 0, decider = %f > 18.0' %(decider)
        print 'ISM Case. Evolution is 4 -> 3 -> 2'
        return [w4, w3, w2]

    elif (specorder == 2):
        [s4_end, s5_beg, s5_end, s1_beg, s1_end, s2_beg, t45, t51, t12] = edges_calc[2:]        
        w4     = 1./(1.+(t/t45)**pl)
        w5     = (1./(1.+(t/t45)**(-1.*pl))) * (1./(1.+(t/t51)**pl))
        w1     = (1./(1.+(t/t51)**(-1.*pl))) * (1./(1.+(t/t12)**pl))
        w2     = 1./(1.+(t/t12)**(-1.*pl))

        print 'k = 2, decider = %f < 100.0' %(decider)
        print 'Wind Case. Evolution is 4 -> 5 -> 1 -> 2'
        return [w4, w5, w1, w2]
        
    elif (specorder == 3):
        [s4_end, s3_beg, s3_end, s2_beg, t43, t32] = edges_calc[2:]        
        w4     = 1./(1.+(t/t43)**pl)
        w3     = (1./(1.+(t/t43)**(-1.*pl))) * (1./(1.+(t/t32)**pl))
        w2     = 1./(1.+(t/t32)**(-1.*pl))

        print 'k = 2, decider = %f > 100.0' %(decider)
        print 'Wind Case. Evolution is 4 -> 3 -> 2'
        return [w4, w3, w2]

    else:
        raise ValueError('Invalid Value, k = %d' %((k)))    
        
def lightcurve(f, params, t):
    """Returns a light curve, given frequency, 'f' and parameters 'params', as a function of time, t (days) """
    
    lc = spectrum(f, params, t)#spect(f, params, t[i])
    return lc

def specnum_valid(params, specnum, t):
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
    if (np.ndim(t) == 0):
        ntimes = 1
    if (np.ndim(t) == 1):
        ntimes = len(t)

    # Create array of break frequencies
    nu_b = np.ones([12,ntimes])
    for i in np.arange(11)+1:
        nu_b[i] = f_b(params, i, t)
    
    if (specnum == 1):
        return (nu_b[1] < nu_b[2]) & (nu_b[2] < nu_b[3])
    elif (specnum == 2):
        return (nu_b[4] < nu_b[5]) & (nu_b[5] < nu_b[3])
    elif (specnum == 3):
        return (nu_b[4] < nu_b[6])
    elif (specnum == 4):
        return (nu_b[7] < nu_b[8]) & (nu_b[8] < nu_b[9])
    elif (specnum == 5):
        return (nu_b[7] < nu_b[10]) & (nu_b[10] < nu_b[11]) & (nu_b[11] < nu_b[9])

def edges(params):
    """Tells you what's going to happen for these set of parameters'"""

    k = params.k
    p = params.p
    z = params.z
    e_e = params.e_e / params.zeta
    e_b = params.e_b / params.zeta
    n_0 = params.n_0 * params.zeta
    Astar = params.Astar * params.zeta
    E52 = params.E52 * params.zeta
    t_jet = params.t_jet

    t_l = 1E-10
    t_r = 1E+10

    import gs_equalities as gseq
    from numpy import e
    from gstable2 import Y

    logtmin = -6
    logtmax = +5
    ntimes1 = 300
    ntimes2 = 301 # Odd number helps to avoid accidentally losing the edge!
    scale = 1.3
    times = np.logspace(logtmin,logtmax,ntimes1)

    if (k == 0):
        decider = n_0 * E52** (4./7.) * e_b** (9./7.)*(1.+Y(params))**(10./7.)
        if (decider <= 18.0):
            valid5_coarse = specnum_valid(params,5,times)
            valid1_coarse = specnum_valid(params,1,times)
            valid2_coarse = specnum_valid(params,2,times)

            if(sum(valid5_coarse) == 0):
                s5_end = 10**logtmin
            else:                
                s5_end = times[max(np.where(valid5_coarse)[0])]
                times_fine = np.logspace(np.log10(s5_end/scale),np.log10(s5_end*scale),ntimes2)
                valid = specnum_valid(params,5,times_fine)
                s5_end = times_fine[max(np.where(valid)[0])]
                
            if(sum(valid1_coarse) == 0):
                s1_beg = s5_end
                s1_end = s5_end
            else:
                s1_beg = times[min(np.where(valid1_coarse)[0])]
                times_fine = np.logspace(np.log10(s1_beg/scale),np.log10(s1_beg*scale),ntimes2)
                valid = specnum_valid(params,1,times_fine)
                s1_beg = times_fine[min(np.where(valid)[0])]
                
                s1_end = times[max(np.where(valid1_coarse)[0])]
                times_fine = np.logspace(np.log10(s1_end/scale),np.log10(s1_end*scale),ntimes2)
                valid = specnum_valid(params,1,times_fine)
                s1_end = times_fine[max(np.where(valid)[0])]
                
            if(sum(valid2_coarse) == 0):
                s2_beg = 10**logtmax
            else:
                s2_beg = times[min(np.where(valid2_coarse)[0])]
                times_fine = np.logspace(np.log10(s2_beg/scale),np.log10(s2_beg*scale),ntimes2)
                valid = specnum_valid(params,2,times_fine)
                s2_beg = times_fine[min(np.where(valid)[0])]

            t51    = np.sqrt(s5_end*s1_beg)
            t12    = np.sqrt(s1_end*s2_beg)

            return np.array([0,decider,s5_end,s1_beg,s1_end,s2_beg,t51,t12])

        elif (decider > 18.0):
            valid4_coarse = specnum_valid(params,4,times)
            valid3_coarse = specnum_valid(params,3,times)
            valid2_coarse = specnum_valid(params,2,times)

            if(sum(valid4_coarse) == 0):
                s4_end = 10**logtmin
            else:
                s4_end = times[max(np.where(valid4_coarse)[0])]
                times_fine = np.logspace(np.log10(s4_end/scale),np.log10(s4_end*scale),ntimes2)
                valid = specnum_valid(params,4,times_fine)
                s4_end = times_fine[max(np.where(valid)[0])]

            if(sum(valid3_coarse) == 0):
                s3_beg = s4_end
                s3_end = s4_end
            else:                
                s3_beg = times[min(np.where(valid3_coarse)[0])]
                times_fine = np.logspace(np.log10(s3_beg/scale),np.log10(s3_beg*scale),ntimes2)
                valid = specnum_valid(params,3,times_fine)
                s3_beg = times_fine[min(np.where(valid)[0])]

                s3_end = gseq.fcross(params,3,5, t_l, t_r)
                #s3_end = times[max(np.where(valid3_coarse)[0])]
                #times_fine = np.logspace(np.log10(s3_end/scale),np.log10(s3_end*scale),ntimes2)
                #valid = specnum_valid(params,3,times_fine)
                #s3_end = times_fine[max(np.where(valid)[0])]

            if(sum(valid2_coarse) == 0):
                s2_beg = 10**logtmax
            else:
                s2_beg = times[min(np.where(valid2_coarse)[0])]
                times_fine = np.logspace(np.log10(s2_beg/scale),np.log10(s2_beg*scale),ntimes2)
                valid = specnum_valid(params,2,times_fine)
                s2_beg = times_fine[min(np.where(valid)[0])]

            t43    = np.sqrt(s4_end*s3_beg)
            t32    = np.sqrt(s3_end*s2_beg)

            return np.array([1,decider,s4_end,s3_beg,s3_end,s2_beg,t43,t32])

    elif (k == 2):
        decider = Astar*e_ebar(params)** (-1.) * E52** (-3./7.) * e_b** (2./7.)*(1.+Y(params))**(3./7.)
        if (decider < 100):
            valid4_coarse = specnum_valid(params,4,times)
            valid5_coarse = specnum_valid(params,5,times)
            valid1_coarse = specnum_valid(params,1,times)
            valid2_coarse = specnum_valid(params,2,times)

            # Note: The 4 -> 5 transition is special, since it is uniquely defined
            
            # Calculation 1 -- compute end time of validity of spectrum 4
            # This does not take advantage of the uniqueness of the transition time
            # and the resulting value of s4_end can be very large
            
            #if(sum(valid4_coarse) == 0):
                #s4_end = 10**logtmin
            #else:
                #s4_end = times[max(np.where(valid4_coarse)[0])]
                #times_fine = np.logspace(np.log10(s4_end/scale),np.log10(s4_end*scale),ntimes2)
                #valid = specnum_valid(params,4,times_fine)
                #s4_end = times_fine[max(np.where(valid)[0])]
            
            # Calculation 2 -- Use an analytic prescription for the transition time. 
            # This ignores the jet break and NR transition
            
            #s4_end=9.3*(1.+z)* e_b** (9./7) * Astar**2. * E52** (-3./7) *(1.+Y(params))**(10./7.)
            #s5_beg=s4_end
            
            # Calculation 3 -- Use the crossing point of nu10 and nu11            
            import gs_equalities as gseq
            s4_end = gseq.fcross(params,10,11,-10,10)
            s5_beg = s4_end

            if(sum(valid5_coarse) == 0):
                #s5_beg = s4_end
                s5_end = s4_end
            else:
                #s5_beg = times[min(np.where(valid5_coarse)[0])]
                #times_fine = np.logspace(np.log10(s5_beg/scale),np.log10(s5_beg*scale),ntimes2)
                #valid = specnum_valid(params,5,times_fine)
                #s5_beg = times_fine[min(np.where(valid)[0])]

                s5_end = times[max(np.where(valid5_coarse)[0])]
                times_fine = np.logspace(np.log10(s5_end/scale),np.log10(s5_end*scale),ntimes2)
                valid = specnum_valid(params,5,times_fine)
                s5_end = times_fine[max(np.where(valid)[0])]

            if(sum(valid1_coarse) == 0):
                s1_beg = s5_end
                s1_end = s5_end
            else:
                s1_beg = times[min(np.where(valid1_coarse)[0])]
                times_fine = np.logspace(np.log10(s1_beg/scale),np.log10(s1_beg*scale),ntimes2)
                valid = specnum_valid(params,1,times_fine)
                s1_beg = times_fine[min(np.where(valid)[0])]

                s1_end = times[max(np.where(valid1_coarse)[0])]
                times_fine = np.logspace(np.log10(s1_end/scale),np.log10(s1_end*scale),ntimes2)
                valid = specnum_valid(params,1,times_fine)
                s1_end = times_fine[max(np.where(valid)[0])]

            if(sum(valid2_coarse) == 0):
                s2_beg = 10**logtmax
            else:
                s2_beg = times[min(np.where(valid2_coarse)[0])]
                times_fine = np.logspace(np.log10(s2_beg/scale),np.log10(s2_beg*scale),ntimes2)
                valid = specnum_valid(params,2,times_fine)
                s2_beg = times_fine[min(np.where(valid)[0])]

            t45    = np.sqrt(s4_end*s5_beg)
            t51    = np.sqrt(s5_end*s1_beg)
            t12    = np.sqrt(s1_end*s2_beg)

            return np.array([2,decider,s4_end,s5_beg,s5_end,s1_beg,s1_end,s2_beg,t45,t51,t12])

        elif (decider >= 100):
            valid4_coarse = specnum_valid(params,4,times)
            valid3_coarse = specnum_valid(params,3,times)
            valid2_coarse = specnum_valid(params,2,times)

            if(sum(valid4_coarse) == 0):
                s4_end = 10**logtmin
            else:
                s4_end = times[max(np.where(valid4_coarse)[0])]
                times_fine = np.logspace(np.log10(s4_end/scale),np.log10(s4_end*scale),ntimes2)
                valid = specnum_valid(params,4,times_fine)
                s4_end = times_fine[max(np.where(valid)[0])]

            if(sum(valid3_coarse) == 0):
                s3_beg = s4_end
                s3_end = s4_end
            else:
                s3_beg = times[min(np.where(valid3_coarse)[0])]
                times_fine = np.logspace(np.log10(s3_beg/scale),np.log10(s3_beg*scale),ntimes2)
                valid = specnum_valid(params,3,times_fine)
                s3_beg = times_fine[min(np.where(valid)[0])]

                s3_end = gseq.fcross(params,3,5, t_l, t_r)
                #s3_end = times[max(np.where(valid3_coarse)[0])]
                #times_fine = np.logspace(np.log10(s3_end/scale),np.log10(s3_end*scale),ntimes2)
                #valid = specnum_valid(params,3,times_fine)
                #s3_end = times_fine[max(np.where(valid)[0])]

            if(sum(valid2_coarse) == 0):
                s2_beg = 10**logtmax
            else:
                s2_beg = times[min(np.where(valid2_coarse)[0])]
                times_fine = np.logspace(np.log10(s2_beg/scale),np.log10(s2_beg*scale),ntimes2)
                valid = specnum_valid(params,2,times_fine)
                s2_beg = times_fine[min(np.where(valid)[0])]

            t43    = np.sqrt(s4_end*s3_beg)
            t32    = np.sqrt(s3_end*s2_beg)

            return np.array([3,decider,s4_end,s3_beg,s3_end,s2_beg,t43,t32])

    else:
        raise ValueError('Invalid Value, k = %d' %((k)))

def plotweights(params, tstart = -7, tstop = 6, Ntime = 100):

    import numpy as np
    import matplotlib.pyplot as plt

    t=np.logspace(tstart,tstop,Ntime)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    W = weights(params, t)
    nspec = np.shape(W)[0]

    for wn in range(nspec):
            plt.loglog(t,W[wn],lw=3);
    if (nspec == 3):
        plt.legend(('w1','w2','w3'),loc='best')
    elif (nspec == 4):
        plt.legend(('w1','w2','w3','w4'),loc='best')

    ax.set_xlabel('Time (Days)');
    ax.set_ylabel('Spectral weight (dimensionless)');
    ax.set_xlim([min(t),max(t)]);
    plt.show()
    
def luminosity(f, params, tstart = -1, tstop = 2, Ntime = 1000):
    
    import numpy as np     
    
    t = np.logspace(tstart, tstop, Ntime)    
    lc = spectrum(f, params, t)
    
    return np.trapz(lc,t) 