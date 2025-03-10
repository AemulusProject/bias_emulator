from Aemulator import *
import numpy as np
import cffi, glob, os, inspect, pickle, warnings
import scipy.optimize as op
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import george

#Create the CFFI library
bias_dir = os.path.dirname(__file__)
include_dir = os.path.join(bias_dir,'include')
lib_file    = os.path.join(bias_dir,'_bias_emulator.so')
# Some installation (e.g. Travis with python 3.x)
# name this e.g. _bias.cpython-34m.so,
# so if the normal name doesn't exist, look for something else.
if not os.path.exists(lib_file):
    alt_files = glob.glob(os.path.join(os.path.dirname(__file__),'_bias_emulator*.so'))
    if len(alt_files) == 0:
        raise IOError("No file '_bias_emulator.so' found in %s"%bias_dir)
    if len(alt_files) > 1:
        raise IOError("Multiple files '_bias_emulator*.so' found in %s: %s"%(bias_dir,alt_files))
    lib_file = alt_files[0]
_ffi = cffi.FFI()
for file_name in glob.glob(os.path.join(include_dir,'*.h')):
    _ffi.cdef(open(file_name).read())
_lib = _ffi.dlopen(lib_file)

#Used to cast things correctly
def _dc(x):
    if isinstance(x, list): x = np.asarray(x, dtype=np.float64, order='C')
    return _ffi.cast('double*', x.ctypes.data)

class bias_emulator(Aemulator):

    def __init__(self):
        Aemulator.__init__(self)
        self.loaded_data = False
        self.built       = False
        self.trained     = False
        self.load_data()
        self.build_emulator()
        self.train_emulator()
        self.cosmology_is_set = False

    def load_data(self, path_to_training_data_directory = None):
        """
        Load training data directly from file, and attach it to this object. 
        This method does not need to be called by the user.
        :param path_to_training_data_directory:
            Location of the training data. Must be in .npy format.
        :return:
            None
        """
        if path_to_training_data_directory is None:
            #Determine the local path to the data files folder
            data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1])) + "/data_files/"
        else:
            data_path = path_to_training_data_directory
        self.data_path = data_path
            
        #Load all training data
        self.training_cosmologies = \
            np.load(data_path+"training_cosmologies.npy")
        self.N_cosmological_params = len(self.training_cosmologies[0])
        self.rotation_matrix      = \
            np.load(data_path+"rotation_matrix.npy")
        self.training_data        = \
            np.load(data_path+"rotated_bias_parameters.npy")
        self.training_mean   = self.training_data[:,:,0] #sample means
        self.training_stddev = self.training_data[:,:,1] #sample stddevs
        self.loaded_data = True
        return

    def build_emulator(self, hyperparams=None):
        """
        Build the emulator directly from loaded training data.
        Optionally provide hyperparameters, 
        if something other than the default is preferred.
        :param hyperparams:
            A dictionary of hyperparameters for the emulator. Default is None.
        :return:
            None
        """
        if not self.loaded_data:
            raise Exception("Need to load training data before building.")

        if hyperparams is None:
            hyperparams = np.std(self.training_cosmologies, 0)

        N_cosmological_params = self.N_cosmological_params
        means  = self.training_mean
        stddev = self.training_stddev

        #Assemble the list of GPs
        self.N_GPs = len(means[0])
        self.GP_list = []
        for i in range(self.N_GPs):
            y    = means[:, i]
            ystd = stddev[:, i]
            kernel = george.kernels.ExpSquaredKernel(hyperparams, ndim=N_cosmological_params)
            gp = george.GP(kernel, mean=np.mean(y))
            gp.compute(self.training_cosmologies, ystd)
            self.GP_list.append(gp)
            continue
        self.built = True
        return

    def train_emulator(self):
        """
        Optimize the hyperparmeters of a built emulator against training data.
        :return:
            None
        """
        if not self.built:
            raise Exception("Need to build before training.")

        means  = self.training_mean

        for i, gp in enumerate(self.GP_list):
            y    = means[:, i]
            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(y, quiet=True)
                return -ll if np.isfinite(ll) else 1e25
            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y, quiet=True)
            p0 = gp.get_parameter_vector()
            result = op.minimize(nll, p0, jac=grad_nll)
            gp.set_parameter_vector(result.x)
        self.trained = True
        return

    def cache_emulator(self, filename):
        """
        Cache the emulator to a file for easier re-loadig. 
        Note: this function and load_emulator() do not work correctly,
        because many attributes are not pickle-able.
        :param filename:
            The filename where the trained emulator will be cached.
        :return:
            None
        """
        with open(filename, "wb") as output_file:
            pickle.dump(self, output_file)
        return

    def load_emulator(self, filename):
        """
        Load an emulator directly from file, pre-trained.
        Note: this function and cache_emulator() do not work correctly,
        because many attributes are not pickle-able.
        :param filename:
            The filename where the trained emulator is located, in a format compatible with
            this object.
        :return:
            None
        """
        if not os.path.isfile(filename):
            raise Exception("%s does not exist to load."%filename)
        with open(filename, "rb") as input_file:
            emu = pickle.load(input_file)
            #Loop over attributes and assign them to this emulator
            for a in dir(emu):
                if not a.startwith("__") and not callable(getattr(emu, a)):
                    setattr(self, a, getattr(emu, a))
                continue
        return

    def predict(self, params):
        """
        Use the emulator to make a prediction at a point in parameter space.
        Note: this returns the slopes and intercepts for the fit function.
        :param params:
            A dictionary of parameters, where the key is the parameter name and
            value is its value.
        :return:
            pred, the emulator prediction at params. Will be a float or numpy array,
            depending on the quantity being emulated.
        """
        if not self.trained:
            raise Exception("Need to train the emulator first.")

        #Properly organize the data
        cos_arr = np.zeros(7)
        cos_arr[0] = params['omega_b'] #Omega_b*h^2
        cos_arr[1] = params['omega_cdm'] #Omega_cdm*h^2
        cos_arr[2] = params['w0']
        cos_arr[3] = params['n_s']
        cos_arr[4] = params['ln10As']
        cos_arr[5] = params['H0']
        cos_arr[6] = params['N_eff']
        cos_arr = np.atleast_2d(cos_arr)

        means = self.training_mean.T #Transpose of mean data
        output = np.array([gp.predict(y, cos_arr)[0] for y,gp in zip(means, self.GP_list)])
        return np.dot(self.rotation_matrix, output).flatten()


    def set_cosmology(self, params, cc=None):
        """
        Set the cosmological parameters of the emulator. One must
        call this function before actually computing the bias.
        :param params:
            A dictionary of parameters, where the key is the parameter name and
            value is its value.
        :return:
            None
        """
        try:
            from classy import Class
        except ImportError:
            print("Class not installed. Cannot compute the bias directly, only predict "+
                  "parameters from the GPs using the predict() function.")
            return
        self.bias_slopes_and_intercepts = self.predict(params)
        #Set up a CLASS dictionary
        self.h = params['H0']/100.
        self.Omega_m = (params["omega_b"]+params["omega_cdm"])/self.h**2
        if cc is None:
            class_cosmology = {
                'output': 'mPk',
                'H0':           params['H0'],
                'ln10^{10}A_s': params['ln10As'],
                'n_s':          params['n_s'],
                'w0_fld':       params['w0'],
                'wa_fld':       0.0,
                'omega_b':      params['omega_b'],
                'omega_cdm':    params['omega_cdm'],
                'Omega_Lambda': 1 - self.Omega_m,
                'N_eff':        params['N_eff'],
                'P_k_max_1/Mpc': 10.,
                'z_max_pk':      5.03
            }
            #Seed splines in CLASS
            cc = Class()
            cc.set(class_cosmology)
            cc.compute()

        #Make everything attributes
        self.cc = cc
        self.k = np.logspace(-5, 1, num=1000) # Mpc^-1 comoving
        self.M = np.logspace(10, 16.5, num=1000) # Msun/h
        self.computed_sigma2      = {}
        self.computed_peak_height = {}
        self.peak_height_splines  = {}
        self.computed_pk          = {}
        self.cosmology_is_set = True
        return

    def predict_bias_parameters(self, redshifts):
        x = 1./(1+redshifts)-0.5
        A0 = 4.2828605
        a0 = 0.4722138
        b0 = 1.5170196
        C0 = 0.888452
        a1 = -0.56318698
        b1 = -0.63010135
        C1 = -0.5956625
        c1 = -1.85148405
        B0, c0, A1, B1 = self.bias_slopes_and_intercepts
        a = a0 + x * a1
        A = A0 + x * A1
        b = b0 + x * b1
        B = B0 + x * B1
        c = c0 + x * c1
        C = C0 + x * C1        
        return A,a,B,b,C,c

    def _compute_peak_height(self, redshifts):
        if not self.cosmology_is_set:
            raise Exception("Must set_cosmology() first.")
        redshifts = np.array(redshifts)
        if redshifts.ndim > 1:
            raise Exception("Redshifts be either a scalar or 1D array.")
        h, Omega_m = self.h, self.Omega_m #Hubble constant and matter fraction
        k, M = self.k, self.M #wavenumbers and halo masses
        Nk = len(k)
        NM = len(M)
        kh = k/h #h/Mpc
        for i,z in enumerate(np.atleast_1d(redshifts)):
            if z in self.computed_sigma2.keys():
                continue
            p = np.array([self.cc.pk_cb_lin(ki, z) for ki in k])*h**3 #[Mpc/h]^3
            sigma2    = np.zeros_like(M)
            _lib.sigma2_at_M_arr(   _dc(M), NM, _dc(kh), _dc(p), Nk, Omega_m, _dc(sigma2))
            self.computed_sigma2[z] = sigma2
            #self.computed_dsigma2dM = {}
            self.computed_peak_height[z] = 1.686/np.sqrt(sigma2)
            self.computed_pk[z]          = p
            continue
        return

    def bias(self, Masses, redshifts, delta=200):
        if not self.cosmology_is_set:
            raise Exception("Must set_cosmology() first.")
        Masses    = np.atleast_1d(Masses)
        redshifts = np.atleast_1d(redshifts)
        if Masses.ndim > 1:
            raise Exception("Masses must be either scalar or 1D array.")
        if redshifts.ndim > 1:
            raise Exception("Redshifts must be either scalar or 1D array.")
        if any(Masses < 1e10) or any(Masses > 10**16.5):
            raise Exception("Mass outside of range 1e10-1e16.5 Msun/h.")
        if any(redshifts > 5):
            raise Exception("Redshift greater than 5.")
        if any(redshifts > 3):
            print("Warning: redshift greather than 3. Accuracy not guaranteed.")

        self._compute_peak_height(redshifts)
        lnMasses = np.log(Masses)
        NM = len(Masses)
        Nz = len(redshifts)
        bias_out = np.zeros((Nz, NM))
        for i,z in enumerate(redshifts):
            A,a,B,b,C,c = self.predict_bias_parameters(z)
            peak_height_spline = IUS(np.log(self.M), self.computed_peak_height[z])
            nu        = peak_height_spline(lnMasses)
            output    = np.zeros_like(Masses)
            _lib.bias_at_nu_arr_FREEPARAMS(_dc(nu), NM, delta,A,a,B,b,C,c, _dc(output)); 
            bias_out[i] = output
            continue
        if Nz == 1:
            return bias_out.flatten()
        return bias_out    

if __name__=="__main__":
    e = bias_emulator()
    print(e)
    cosmology={
        "omega_b": 0.02268325,
        "omega_cdm": 0.1140598,
        "w0": -0.8165972,
        "n_s": 0.975589,
        "ln10As": 3.092918,
        "H0": 63.36569,
        "N_eff": 2.91875
    }
    print(e.predict(cosmology))
    e.set_cosmology(cosmology)
    print(e.predict_bias_parameters(0))
    
