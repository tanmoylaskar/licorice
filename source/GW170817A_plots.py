plotlc = True

if plotlc:
    nu_LC1 = 8e9
    namelist = ['Radio (8 GHz)']
    LC1 = spectrum_all(params, nua,numax,nuc,nufive,nuseven,Fmax,nu_LC1*ones_like(numax))
    
    figure();
    artistlist = []
    import realspectra
    art = loglog(tobs,LC1,'r-',lw=2); artistlist.append(art[0])
    #F = realspectra.spectrum(nu_LC1,params,tobs); loglog(tobs,F,'r:')
    
