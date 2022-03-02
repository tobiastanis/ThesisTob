"""
Doesn't work yet in defined methods
"""
from tudatpy.kernel import constants
from astropy.time import TimeMJD, Time

# Two epochs in which initial states are considered for LUMIO datapack
t0_1_iso = '2024-03-21 00:00:00.000'
t0_2_iso = '2024-04-18 00:00:00.000'
t0_1_mjd = Time(t0_1_iso, format='iso')
t0_2_mjd = Time(t0_2_iso, format='iso')
t0_1_mjd.format = 'mjd'
t0_2_mjd.format = 'mjd'
# A third epoch which is considered as the final time is 21-03-2025
tend_iso = '2025-03-21 00:00:00.000'
tend_mjd = Time(tend_iso, format='iso')
tend_mjd.format = 'mjd'



