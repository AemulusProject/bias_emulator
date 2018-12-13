import pytest
from bias import *
import numpy as np
import numpy.testing as npt

#Create an emulator
h = bias.bias_emulator()

def test_bias_emulator_load_data():
    npt.assert_equal(h.loaded_data, True)
    attrs = ["data_path", "training_cosmologies",
             "rotation_matrix", "training_data",
             "training_mean", "training_stddev"]
    for attr in attrs:
        npt.assert_equal(hasattr(h, attr), True)
        continue
    return

def test_bias_emulator_build_emulator():
    npt.assert_equal(h.built, True)
    attrs = ["N_GPs", "GP_list"]
    for attr in attrs:
        npt.assert_equal(hasattr(h, attr), True)
        continue
    npt.assert_equal(h.N_GPs, len(h.GP_list))
    return

def test_bias_emulator_train_emulator():
    npt.assert_equal(h.trained, True)
    return
