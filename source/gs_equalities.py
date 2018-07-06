def fcross(params, f1, f2, t_l, t_r):
    import gstable2
    import scipy.optimize as opt
    from numpy import sign
    t_l = 1E-10
    t_r = 1E+10
    f1l = gstable2.f_b(params,f1,t_l)
    f1r = gstable2.f_b(params,f1,t_r)
    f2l = gstable2.f_b(params,f2,t_l)
    f2r = gstable2.f_b(params,f2,t_r)
    
    if (sign(f1l-f2l) == sign(f1r-f2r)):
        flmax = max(f1l, f2l)
        flmin = min(f1l, f2l)
        frmax = max(f1r, f2r)
        frmin = min(f1r, f2r)
        if (flmax / flmin < frmax / frmin):
            return t_l
        else:
            return t_r
    else:
        #crosspoint = opt.fsolve(lambda t:gstable2.f_b(params,f1,t)-gstable2.f_b(params,f2,t),initguess)
        #crosspoint = opt.newton(lambda t:gstable2.f_b(params,f1,t)-gstable2.f_b(params,f2,t),initguess)
        crosspoint = opt.brentq(lambda t:gstable2.f_b(params,f1,t)-gstable2.f_b(params,f2,t),t_l,t_r)
    return crosspoint
