import sys, os, glob
from setuptools import setup, Extension
import subprocess

# Make a symlink between the bias directory and the src directory
# in order to see the .h files.
os.system('ln -h -s ../include bias_emulator/include')

sources = glob.glob(os.path.join('src','*.c'))
headers = glob.glob(os.path.join('include','*.h'))
try:
    cflags = subprocess.check_output(['gsl-config', '--cflags'], universal_newlines=True).split()
    lflags = subprocess.check_output(['gsl-config', '--libs'], universal_newlines=True).split()
except OSError:
    raise Exception("Error: must have GSL installed and gsl-config working")

ext=Extension("bias_emulator._bias_emulator",
              sources,
              depends=headers,
              include_dirs=['include'],
              extra_compile_args=[os.path.expandvars(flag) for flag in cflags],
              extra_link_args=[os.path.expandvars(flag) for flag in lflags])

dist = setup(name="bias_emulator",
             author="Tom McClintock",
             author_email="mcclintock@bnl.gov",
             description="Emulator for the halo mass function.",
             license="GNU General Public License v2.0",
             url="https://github.com/AemulusProject/bias_emulator",
             packages=['bias_emulator'],
             package_data={'bias_emulator' : headers},
             include_package_data=True,
             ext_modules=[ext],
             install_requires=['cffi','numpy','scipy','george'],
             setup_requires=['pytest_runner'],
             tests_require=['pytest'])

# setup.py doesn't put the .so file in the bias directory, 
# so this bit makes it possible to
# import bias from the root directory.  
# Not really advisable, but everyone does it at some
# point, so might as well facilitate it.
build_lib = glob.glob(os.path.join('build','*','bias_emulator','_bias_emulator*.so'))
if len(build_lib) >= 1:
    lib = os.path.join('bias_emulator','_bias_emulator.so')
    if os.path.lexists(lib): os.unlink(lib)
    os.link(build_lib[0], lib)
