from typing import Union

import pandas as pd
import xarray as xa
import seaborn as sb

from copy import copy
from pathlib import Path

from astropy.io import fits as pf
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.units import Rjup, Rsun, m, AU

from numpy import zeros, diff, concatenate, sqrt, degrees, radians, array, arange, where
from numpy.random.mtrand import normal, uniform
from uncertainties import ufloat, nominal_value

from pytransit.orbits import as_from_rhop, i_from_ba, d_from_pkaiews, epoch
from pytransit.utils.eclipses import Teq
from pytransit.utils.keplerlc import KeplerLC

from muscat2ta.m2lpf import downsample_time

N = lambda a: a/a.median()

def normalize(a):
    if isinstance(a, pd.DataFrame):
        return (a - a.mean()) / a.std()

def split_normal(mu, lower_sigma, upper_sigma, size=1):
    z = normal(0, 1, size=size)
    return where(z<0, mu+z*lower_sigma, mu+z*upper_sigma)
    
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
    if use_pdc:
        contamination = 1 - tb.meta['CROWDSAP']
        fluxes = [contamination + (1 - contamination)*f for f in fluxes]
    ins = len(times)*["TESS"]
    piis = list(arange(len(times)))
    return times, fluxes, len(times)*['tess'], [diff(concatenate(fluxes)).std() / sqrt(2)], ins, piis

# MuSCAT2 routines
# ----------------

def read_m2(files: list, downsample=None):
    times, fluxes, pbs, wns, covs = [], [], [], [], []
    for inight, f in enumerate(files):
        with pf.open(f) as hdul:
            npb = (len(hdul)-1)//2
            for ipb in range(npb):
                hdu = hdul[1+ipb]
                fobs = hdu.data['flux'].astype('d').copy()
                fmod = hdu.data['model'].astype('d').copy()
                time = hdu.data['time_bjd'].astype('d').copy()
                mask = ~sigma_clip(fobs-fmod, sigma=5).mask
                
                pbs.append(hdu.header['filter'])
                wns.append(hdu.header['wn'])
                
                if downsample is None:                
                    times.append(time[mask])
                    fluxes.append(fobs[mask])
                    covs.append(Table.read(f, 1+npb+ipb).to_pandas().values[mask,1:])
                else:
                    cov = Table.read(f, 1+npb+ipb).to_pandas().values[mask,1:]
                    tb, fb = downsample_time(time[mask], fobs[mask], downsample)
                    _,  cb = downsample_time(time[mask], cov, downsample)
                    times.append(tb)
                    fluxes.append(fb)
                    covs.append(cb)
    ins = len(times)*["M2"]
    piis = list(arange(len(times)))
    return times, fluxes, pbs, wns, covs, ins, piis

# LCO routines
# ------------

def read_lco_table(fname: Union[Path, str], bjd_name='BJD_TDB', sep='\t'):
    dff = pd.read_csv(fname, sep=sep, index_col=0)
    refcols = [c for c in dff.columns if 'rel_flux_C' in c]
    df = dff[f'{bjd_name} rel_flux_T1'.split() + refcols]
    covariates = normalize(dff[['AIRMASS', 'FWHM_Mean', 'X(IJ)_T1', 'Y(IJ)_T1']]).values
    return df, covariates

# Result dataframe routines
# -------------------------

def read_mcmc(fname, flatten=True):
    with xa.open_dataset(fname) as ds:
        if flatten:
            try:
                npt = ds.lm_mcmc.shape[-1]
                df = pd.DataFrame(array(ds.lm_mcmc).reshape([-1, npt]), columns=ds.coords['name'].values)
            except AttributeError:
                npt = ds.mcmc_samples.shape[-1]
                df = pd.DataFrame(array(ds.mcmc_samples).reshape([-1, npt]), columns=ds.parameter)
            return df
        else:
            try:
                return array(ds.lm_mcmc)
            except AttributeError:
                return array(ds.mcmc_samples)


def read_tess_mcmc(fname):
    with xa.open_dataset(fname) as ds:
        npt = ds.lm_mcmc.shape[-1]
        df = pd.DataFrame(array(ds.lm_mcmc).reshape([-1, npt]), columns=ds.coords['lm_parameter'].values)
    return df

def derive_qois(df_original, rstar, star_teff):
    df = df_original.copy()
    ns = df.shape[0]

    rstar_d = normal(rstar.n, rstar.s, size=ns) * Rsun
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

    df['r_app_rsun'] = df.k_app.values * rstar_d.to(Rsun)
    df['r_true_rsun'] = df.k_true.values * rstar_d.to(Rsun)
    df['teff_p'] = Teq(normal(star_teff.n, star_teff.s, size=ns), df.a_st, uniform(0.25, 0.50, ns), uniform(0, 0.4, ns))
    return df
