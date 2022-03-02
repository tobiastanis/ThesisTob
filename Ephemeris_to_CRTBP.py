""""
Conversion from ephemeris state to CRTBP
"""

import numpy as np
import LUMIO_States_reader as lsr
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
spice_interface.load_standard_kernels()

G = constants.GRAVITATIONAL_CONSTANT                                    # Gravitational constant
m1 = spice_interface.get_body_gravitational_parameter("Earth")/G        # Mass P1 (aka Earth)
m2 = spice_interface.get_body_gravitational_parameter("Moon")/G         # Mass P2 (aka Moon)

m_char = m1+m2              # Characteristic mass [kg]
mu = m2/m_char              # Non dimensional unit of mass

t_ET = float(lsr.LUMIOdata_timespan[0, 1])
X_LUMIO_ephem = lsr.state_LUMIO[0, :]*10**3
