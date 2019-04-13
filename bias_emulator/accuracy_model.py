import numpy as np

class bias_accuracy(object):

    def __init__(self, parameters=None):
        #Parameters are D, E, F, Lnu, Lz
        if parameters is None:
            #Optimized externally
            self.parameters = np.array([ 0.00549773, -0.00229455,
                                         0.03623199,  1.28157324,
                                         1.77962876])
        else:
            self.parameters = parameters
            
    def accuracy_at_nu_z(self, nu, z):
        """Accuracy at a given peak height and redshift
        """
        z = np.asarray(z)
        nu = np.asarray(nu)
        if not (z.shape == nu.shape or z.size == 1):
            raise Exception("z must be scalar or the same shape as nu.")
        D, E, F, _, _ = self.parameters

        a0 = (1./(1+z) - 0.5)*np.ones_like(nu)
        return D + (F*a0**2 + E*a0)

    def covariance_model(self, nu, z, dnu=None, dz=None):
        z = np.asarray(z)
        nu = np.asarray(nu)
        if not (z.shape == nu.shape or z.size == 1):
            raise Exception("z must be scalar or the same shape as nu.")
        
        z_nu = np.ones_like(nu)*z
        sigma = self.accuracy_at_nu_z(nu, z_nu)
        C = np.outer(sigma, sigma)

        if dnu is None:
            dnu = np.array([np.fabs(nu - nu_i) for nu_i in nu])
        if dz is None:
            dz = np.array([np.fabs(z_nu - z_nu_i) for z_nu_i in z_nu])

        _,_,_, l_nu, l_z = self.parameters
        R = np.exp(-dnu/l_nu - dz/l_z)
        return C*R

    def set_parameters(self, parameters):
        assert len(parameters) == 5
        self.parameters = parameters
        return
