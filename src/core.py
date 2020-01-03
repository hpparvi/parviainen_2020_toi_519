import pandas as pd
import xarray as xa
import seaborn as sb

from copy import copy
from pathlib import Path

from astropy.io import fits as pf
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.units import Rjup, Rsun, m, AU

from numpy import zeros, diff, concatenate, sqrt, degrees, radians, array
from numpy.random.mtrand import normal, uniform
from uncertainties import ufloat, nominal_value

from pytransit.orbits import as_from_rhop, i_from_ba, d_from_pkaiews, epoch
from pytransit.utils.eclipses import Teq
from pytransit.utils.keplerlc import KeplerLC

N = lambda a: a/a.median()

def normalize(a):
    if isinstance(a, pd.DataFrame):
        return (a - a.mean()) / a.std()

# TESS routines
# -------------

def read_tess(dfile: Path, zero_epoch: float, period: float, use_pdc: bool = False,
              transit_duration_d: float = 0.1, baseline_duration_d: float = 0.3):
    fcol = 'PDCSAP_FLUX' if use_pdc else 'SAP_FLUX'
    tb = Table.read(dfile)
    bjdrefi = tb.meta['BJDREFI']
    df = tb.to_pandas().dropna(subset=['TIME', 'SAP_FLUX', 'PDCSAP_FLUX'])
    lc = KeplerLC(df.TIME.values + bjdrefi, df[fcol].values, zeros(df.shape[0]),
                  nominal_value(zero_epoch), nominal_value(period), transit_duration_d, baseline_duration_d)
    times, fluxes = copy(lc.time_per_transit), copy(lc.normalized_flux_per_transit)
    return times, fluxes, len(times)*['tess'], [diff(concatenate(fluxes)).std() / sqrt(2)]

# MuSCAT2 routines
# ----------------

def read_m2(files: list):
    times, fluxes, pbs, wns, covs = [], [], [], [], []
    for f in files:
        with pf.open(f) as hdul:
            npb = (len(hdul)-1)//2
            for hdu in hdul[1:1+npb]:
                fobs = hdu.data['flux'].astype('d').copy()
                fmod = hdu.data['model'].astype('d').copy()
                time = hdu.data['time_bjd'].astype('d').copy()
                mask = ~sigma_clip(fobs-fmod, sigma=5).mask
                times.append(time[mask])
                fluxes.append(fobs[mask])
                pbs.append(hdu.header['filter'])
                wns.append(hdu.header['wn'])
            for i in range(npb):
                covs.append(Table.read(f, 1+npb+i).to_pandas().values[:,1:])
    return times, fluxes, pbs, wns, covs


# LCO routines
# ------------

def read_lco_table(fname: Path):
    dff = pd.read_csv(fname, sep='\t', index_col=0)
    refcols = [c for c in dff.columns if 'rel_flux_C' in c]
    df = dff['BJD_TDB rel_flux_T1'.split() + refcols]
    covariates = normalize(dff[['AIRMASS', 'FWHM_Mean', 'X(IJ)_T1', 'Y(IJ)_T1']]).values
    return df, covariates

def read_lco_data():
    dfg, cg = read_lco_table('photometry/lco/TIC120916706-01_20190121_LCO-SAAO-1m_gp_measurements.tbl')
    dfr, cr = read_lco_table('photometry/lco/TIC120916706.01_20181226_LCO-CTIO-1m_rp_measurements.tbl')
    dfi, ci = read_lco_table('photometry/lco/TIC120916706-01_20181213_LCO-SAAO-1m_ip_measurements.tbl')
    dfs = dfg, dfr, dfi
    covs = [cg, cr, ci]

    times = [df.BJD_TDB.values for df in dfs]
    fluxes = [N(df.rel_flux_T1).values for df in dfs]
    return times, fluxes, 'g r i'.split(), [diff(f).std() / sqrt(2) for f in fluxes], covs

# Result dataframe routines
# -------------------------

def read_mcmc(fname, flatten=True):
    with xa.open_dataset(fname) as ds:
        npt = ds.lm_mcmc.shape[-1]
        if flatten:
            df = pd.DataFrame(array(ds.lm_mcmc).reshape([-1, npt]), columns=ds.coords['name'].values)
            return df
        else:
            return array(ds.lm_mcmc)

def read_tess_mcmc(fname):
    with xa.open_dataset(fname) as ds:
        npt = ds.lm_mcmc.shape[-1]
        df = pd.DataFrame(array(ds.lm_mcmc).reshape([-1, npt]), columns=ds.coords['lm_parameter'].values)
    return df

def derive_qois(df_original):
    df = df_original.copy()
    ns = df.shape[0]

    rstar_d = normal(rstar.value, rstare.value, size=ns) * Rsun
    period = df.p.values if 'p' in df.columns else df.pr.values

    df['period'] = period
    df['k_true'] = sqrt(df.k2_true)
    df['k_app'] = sqrt(df.k2_app)
    df['cnt'] = 1. - df.k2_app / df.k2_true
    df['a_st'] = as_from_rhop(df.rho.values, period)
    df['a_au'] = df.a_st * rstar_d.to(AU)
    df['inc'] = degrees(i_from_ba(df.b.values, df.a_st.values))
    df['t14'] = d_from_pkaiews(period, df.k_true.values, df.a_st.values, radians(df.inc.values), 0.0, 0.0, 1)
    df['t14_h'] = 24 * df.t14

    df['r_app'] = df.k_app.values * rstar_d.to(Rjup)
    df['r_true'] = df.k_true.values * rstar_d.to(Rjup)
    df['r_app_point'] = df.k_app.values * rstar.to(Rjup)
    df['r_true_point'] = df.k_true.values * rstar.to(Rjup)

    df['r_app_rsun'] = df.k_app.values * rstar_d.to(Rsun)
    df['r_true_rsun'] = df.k_true.values * rstar_d.to(Rsun)
    df['teff_p'] = Teq(normal(*star_teff, size=ns), df.a_st, uniform(0.25, 0.50, ns), uniform(0, 0.4, ns))
    return df
