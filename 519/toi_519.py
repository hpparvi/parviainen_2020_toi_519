import pandas as pd

from typing import Union
from pathlib import Path
from astropy.units import Rjup, Rsun, m, AU
from numpy import diff, sqrt, arange

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

N = lambda a: a/a.median()

def normalize(a):
    if isinstance(a, pd.DataFrame):
        return (a - a.mean()) / a.std()

def read_lco_table(fname: Union[Path, str], bjd_name: str = 'BJD_TDB'):
    dff = pd.read_csv(fname, sep='\t', index_col=0)
    refcols = [c for c in dff.columns if 'rel_flux_C' in c]
    df = dff[f'{bjd_name} rel_flux_T1'.split() + refcols]
    covariates = normalize(dff[['AIRMASS', 'FWHM_Mean', 'X(IJ)_T1', 'Y(IJ)_T1']]).values
    return df, covariates

def read_lco_data():
    dfi, ci = read_lco_table('photometry/lco/TIC218795833.01_20190401_LCO1m_ip_Measurements.xls',
                             bjd_name='BJD_TDB_MOBS')
    dfz, cz = read_lco_table('photometry/lco/TIC218795833-01_20190416_LCO-CTIO-1m_zs_measurements.tbl')
    dfs = dfi, dfz
    covs = [ci, cz]

    times = [dfs[0].BJD_TDB_MOBS.values, dfs[1].BJD_TDB.values]
    fluxes = [N(df.rel_flux_T1).values for df in dfs]
    ins = len(times) * ['LCO']
    piis = list(arange(len(times)))
    return times, fluxes, 'i z_s'.split(), [diff(f).std() / sqrt(2) for f in fluxes], covs, ins, piis
