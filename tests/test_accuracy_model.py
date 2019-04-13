import pytest
from bias_emulator import *
import numpy as np
import numpy.testing as npt

ac = accuracy_model.bias_accuracy()

def test_acc_and_cov():
    acc = ac.accuracy_at_nu_z(1, 1)
    npt.assert_equal(1, acc)
    cov = ac.covariance_model([1,1],1)
    npt.assert_equal([[1,1],[1,1]], cov)

    #Test for symmetry
    cov = ac.covariance_model([.2,1],1)
    npt.assert_equal(cov, cov.T)

    return

def test_param_set():
    newac = accuracy_model.bias_accuracy()
    npt.assert_equal(np.ones(5), newac.parameters)
    newac.set_parameters(2*np.ones(5))
    npt.assert_equal(2*np.ones(5), newac.parameters)
    return

