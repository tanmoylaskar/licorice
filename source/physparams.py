class Physparams(object):
    'Physical parameters for a spectrum, viz k, p, z, e_e, e_b, n_0, Astar and E52'
    k = 0; p = 2.5; z = 0.0; e_e = 0.5; e_b = 0.5; n_0 = 1.0; Astar = 1.0; E52 = 1.0;
    t_jet = 1; zeta = 1;
    A_B = 0 # Host extinction in the B band

    def __init__(self, k = 0, p = 2.5, z = 0, e_e = 0.5, e_b = 0.5, n_0 = 1, Astar = 1, E52 = 1, t_jet = 1, A_B = 0, zeta = 1.0, applyIC = True):
        self.k = k; self.p = p; self.z = z; self.e_e = e_e; self.e_b = e_b;
        self.n_0 = n_0; self.Astar = Astar; self.E52 = E52;
        self.t_jet = t_jet;
        self.A_B = A_B # Host extinction in the B band
        self.zeta = zeta
        self.applyIC = applyIC
        from numpy import pi, cos
        if (k == 0):
            self.theta_jet = 0.1*(self.E52/(self.n_0*1.0))**(-0.125) * (24*self.t_jet/(6.2*(1.+self.z)))**(0.375)*180.0/pi
        else:
            self.theta_jet = 0.17*(2*self.t_jet*self.Astar/((1.+self.z)*self.E52))**0.25*180./pi # Chevalier and Li (2000)
        self.E_cor = (1-cos(self.theta_jet*pi/180.0))*self.E52
        self.t_nr = max(1./(self.theta_jet*pi/180.)**2,1.0)*self.t_jet
        
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=68.0,Om0=0.31,Tcmb0=0.)
        self.dL28 = 3.08567758e-4*cosmo.luminosity_distance(self.z).value
        
    def printPhysparams(self):
        print 'k = %i \np = %f \nz = %f \ne_e = %f \ne_b = %f \
                \nn_0 = %f \nAstar = %f \nE52 = %f \nt_jet = %f\nt_nr = %f\nA_B = %f\ntheta_jet = %f\nE_cor = %f\nzeta = %f' \
            % (self.k, self.p, self.z, self.e_e, self.e_b, \
                    self.n_0, self.Astar, self.E52, self.t_jet, self.t_nr, self.A_B, self.theta_jet, self.E_cor, self.zeta)
    def updatelumdist(self):
        import cosmolopy.distance as cd
        self.dL28 = 3.08567758e-4*cd.luminosity_distance(self.z, **cosmology)

    def updatejetparams(self):
        from numpy import pi, cos
        if (self.k == 0):
            self.theta_jet = 0.1*(self.E52/(self.n_0*1.0))**(-0.125) * (24*self.t_jet/(6.2*(1.+self.z)))**(0.375)*180.0/pi
            #self.t_nr      = 24.*7*0.5*(1.+self.z)*(self.E52/self.n_0)**(1./3.) # Waxman, Kulkarni and Frail (1998)
            self.t_nr = 18.81*(1.+self.z)*(self.E52/self.n_0)**(1./3.) # By setting Gamma_tnr = 1.215, 
                                                                       # where the expression for Gamma is from GS02 
                                                                       # and the value of 1.215 comes from Waxman, Kulkarni and Frail (1998)
        else:
            self.theta_jet = 0.17*(2*self.t_jet*self.Astar/((1.+self.z)*self.E52))**0.25*180./pi # Chevalier and Li (2000)
            #self.t_nr      = 1.9*365*(1+self.z)*self.E52/self.Astar # Chevalier and Li (2000)
            self.t_nr = 88.23*(1.+self.z)*(self.E52/self.Astar) # See note for ISM case above
        self.t_nr = max(self.t_nr,self.t_jet)
        self.E_cor = (1-cos(self.theta_jet*pi/180.0))*self.E52

