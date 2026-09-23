"""SI constants; Geant4 unit conversion happens only in the native adapter."""

import math

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
M_E = 9.1093837139e-31
C = 299792458.0
K_B = 1.380649e-23
TWO_PI = 2.0 * math.pi
COULOMB_K = 1.0 / (4.0 * math.pi * EPS0)
FWHM_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

