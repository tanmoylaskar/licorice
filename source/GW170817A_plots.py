plotlc = True
plotspectrum = True
import asciitable, realspectra, gstable2

if plotspectrum:
    # Find a time to plot a spectrum
    # Chosen as the time of the peak of the 1 GHz light curve
    nu_LC1 = 3e9
    namelist = ['Radio (3 GHz)']
    LC1 = spectrum_all(params,nua,numax,nuc,nufive,nuseven,Fmax,nu_LC1*ones_like(numax))
    i = where(LC1 == max(LC1))[0]
    
    # Calculate the spectrum
    qlist = logspace(6,21,100)
    S = spectrum_all(params,nua[i],numax[i],nuc[i],nufive[i],nuseven[i],Fmax[i],qlist)
    F = realspectra.spectrum(qlist, params, tobs[i])
    
    figure(); loglog(qlist, S); loglog(qlist, F)

if plotlc:
    nu_LC1 = 3e9
    namelist = ['Radio (3 GHz)']
    LC1 = spectrum_all(params,nua,numax,nuc,nufive,nuseven,Fmax,nu_LC1*ones_like(numax))
    
    figure();
    artistlist = []
    import realspectra
    art = loglog(tobs,LC1,'r-',lw=2); artistlist.append(art[0])
    F = realspectra.spectrum(nu_LC1,params,tobs); loglog(tobs,F,'r:')
    
    xlim(1e4/86400.,3.6e7/86400.)
    
    LC_GW = asciitable.read("3GHz_lc.txt")
    errorbar(LC_GW['col1'],LC_GW['col2']/1e3,LC_GW['col3']/1e3,fmt='o',ls=None)
    
    if (offaxis):
        boxlcfile = "/home/tanmoy/Projects/Manuscripts/Proposals/GMRT/Cycle35/gw/boxfit/170817/lightcurve_035.txt"
    else:
        boxlcfile = "/home/tanmoy/Projects/Manuscripts/Proposals/GMRT/Cycle35/gw/boxfit/170817/lightcurve_00.txt"
    L = asciitable.read(boxlcfile)
    loglog(L['col2']/86400.,L['col4']); ylim(1e-4,20)
    
    if plotspectrum:
        axvline(tobs[i],color='k',ls=':')
