from pathlib import Path
from astropy.units import Rjup, Rsun, m, AU

from uncertainties import ufloat

zero_epoch = ufloat(2458491.877197, 0.000411)
period = ufloat(1.265223, 3.6e-5)

star_teff = (3225, 100)
#star_logg = (4.9, 0.1)
#star_z    = (0.0, 0.1)

rnep = (24622000 * m).to(Rjup)
rstar, rstare = 0.251*Rsun, 0.056*Rsun

root = Path(__file__).parent.resolve()
tess_file = root.joinpath('photometry/tess2019006130736-s0007-0000000218795833-0131-s_lc.fits').resolve()
reduced_m2_files = sorted(root.joinpath('results').glob('*fits'))