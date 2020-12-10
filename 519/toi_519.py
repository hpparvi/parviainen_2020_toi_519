import sys
import seaborn as sb
import pandas as pd
import pytransit.contamination as cn

from pathlib import Path

from astropy.table import Table
from numba import njit
from ldtk import tess
from numpy import diff, sqrt, arange, array, ndarray, pi, sin, cos, log10, inf, atleast_2d, squeeze, newaxis, exp, \
    atleast_1d, zeros, log, sum, isfinite, nanmedian, ones, full
from pytransit import BaseLPF, UniformModel
from pytransit.param import ParameterSet, GParameter, UniformPrior as UP, NormalPrior as NP
from pytransit.utils.eclipses import reflected_fr
from pytransit.utils.misc import fold
from uncertainties import ufloat

from pytransit.lpf.tess.tgclpf import BaseTGCLPF

sys.path.append('..')
from src.core import read_lco_table, N, read_tess, read_m2

import astropy.units as u

mj2kg = u.M_jup.to(u.kg)
ms2kg = u.M_sun.to(u.kg)
d2s = u.day.to(u.s)

# Stellar parameters from ALFOSC spectrum
# ---------------------------------------
star_teff = ufloat(3300,  100)
star_logg = ufloat( 5.5,  0.5)
star_z    = ufloat( 0.5,  0.1)
star_r    = ufloat(0.373, 0.050)
star_m    = ufloat(0.369, 0.050)

# Prior orbital parameters
# ------------------------
zero_epoch = ufloat(2458491.877197, 0.000411)
period = ufloat(1.265223, 3.6e-5)

# Photometry files
# ----------------
root = Path(__file__).parent.resolve()
m2_files = sorted(root.joinpath('results').glob('*fits'))
tess_file = (root / 'photometry' / 'tess2019006130736-s0007-0000000218795833-0131-s_lc.fits').resolve()
lco_files = (root / 'photometry' / 'lco' / 'TIC218795833.01_20190401_LCO1m_ip_Measurements.xls',
             root / 'photometry' / 'lco' / 'TIC218795833-01_20190416_LCO-CTIO-1m_zs_measurements.tbl')


def read_lco_data():
    dfi, ci = read_lco_table(lco_files[0], bjd_name='BJD_TDB_MOBS')
    dfz, cz = read_lco_table(lco_files[1])
    dfs = dfi, dfz
    covs = [ci, cz]

    times = [dfs[0].BJD_TDB_MOBS.values, dfs[1].BJD_TDB.values]
    fluxes = [N(df.rel_flux_T1).values for df in dfs]
    ins = len(times) * ['LCO']
    piis = list(arange(len(times)))
    return times, fluxes, 'i z_s'.split(), [diff(f).std() / sqrt(2) for f in fluxes], covs, ins, piis


def read_m1_data():
    m1datadir = root / 'photometry' / 'm1'
    m1files = ('TIC218795833-01_20191130_muscat_g_measurements.csv',
               'TIC218795833-01_20191130_muscat_r_measurements.csv',
               'TIC218795833-01_20191130_muscat_z_measurements.csv')

    times, fluxes, covs, wns = [], [], [], []
    for fname in m1files:
        df = pd.read_csv(m1datadir / fname)
        times.append(df.BJD_TDB.values.copy())
        fluxes.append(df.Flux.values.copy())
        covs.append(df.iloc[:, 3:-1].values.copy())
        wns.append(diff(fluxes[-1]).std() / sqrt(2))

    pbs = 'g r z_s'.split()
    ins = len(times) * ['M1']
    piis = list(arange(len(times)))
    return times, fluxes, pbs, wns, covs, ins, piis


def create_eleanor_sector_mask(f, i):
    m = ones(f.size, bool)

    if i == 0:
        m[f<0.8] = 0
        m[:40] = 0
        m[540:587] = 0
    else:
        m[:80] = 0
        m[555:720] = 0
        m[f<0.91] = 0
    return m    


def decontaminate(f, c):
    n = nanmedian(f)
    return n + (f-n) / (1-c)


def read_eleanor(zero_epoch, period, bjdrefi, mask_transits=False, trwidth=0.053, contamination=None):
    dd = root / 'photometry'
    files = sorted(dd.glob('hlsp_eleanor*.fits'))
    times, fluxes, wnids, lcids = [], [], [], []
    
    s = 0
    for i,f in enumerate(files):
        df =  Table.read(f)
        time = df['TIME'].data + bjdrefi
        flux = df['CORR_FLUX'].data
        m = isfinite(time) & isfinite(flux)
        if mask_transits:
            p = (fold(time, period, zero_epoch, 0.5) - 0.5) * period
            m &= (abs(p) > 0.5*trwidth)
        flux /= nanmedian(flux)
        m &= create_eleanor_sector_mask(flux, i)
        
        if contamination is not None:
            flux = decontaminate(flux, contamination[i])
        
        if m.sum() > 0:
            s += 1
            times.append(time[m])
            fluxes.append(flux[m])
            lcids.append(full(time[m].size, i, 'int'))
    wns = [diff(f).std()/sqrt(2) for f in fluxes]
    pbs = s * ['TESS']
    return times, fluxes, pbs, wns, pbs, list(arange(len(times)))


# Define the log posterior functions
# ----------------------------------
# The `pytransit.lpf.tess.BaseTGCLPF` class can be used directly to model TESS photometry together with ground-based
# multicolour photometry with physical contamination on the latter. The only thing that needs to be implemented is the
# `BaseTGCLPF.read_data()` method that reads and sets up the data. However, we also use the `_post_initialisation`
# hook to set the parameter priors for the so that we don't need to remember to set these later.

class LPF(BaseTGCLPF):
    def __init__(self, name: str, use_ldtk: bool = False, use_opencl: bool = False, use_pdc: bool = True, heavy_baseline: bool = True):
        self.use_pdc = use_pdc
        self.use_opencl = use_opencl
        self.heavy_baseline = heavy_baseline
        super().__init__(name, use_ldtk)

    def read_data(self):
        times_t, fluxes_t, pbs_t, wns_t, ins_t, piis_t = read_tess(tess_file, zero_epoch.n, period.n,
                                                                   baseline_duration_d=0.15, use_pdc=self.use_pdc)
        times_m2, fluxes_m2, pbs_m2, wns_m2, covs_m2, ins_m2, piis_m2 = read_m2(m2_files)
        times_m1, fluxes_m1, pbs_m1, wns_m1, covs_m1, ins_m1, piis_m1 = read_m1_data()
        times_l, fluxes_l, pbs_l, wns_l, covs_l, ins_l, piis_l = read_lco_data()

        times = times_t + times_m2 + times_m1 + times_l
        fluxes = fluxes_t + fluxes_m2 + fluxes_m1 + fluxes_l
        pbs = pbs_t + pbs_m2 + pbs_m1 + pbs_l
        wns = wns_t + wns_m2 + wns_m1 + wns_l
        if self.heavy_baseline:
            covs = len(times_t) * [array([[]])] + covs_m2 + covs_m1 + covs_l
        else:
            covs = (len(times_t) + len(times_m2) + len(times_m1) + len(times_l))* [array([[]])]

        pbnames = 'tess g r i z_s'.split()

        self._stess = len(times_t)
        self._ntess = sum([t.size for t in times_t])
        self.ins = ins_t + ins_m2 + ins_m1 + ins_l
        self.piis = piis_t + piis_m2 + piis_m1 + piis_l

        return times, fluxes, pbnames, pbs, wns, covs

    def _post_initialisation(self):
        if self.use_opencl:
            self.tm = self.tm.to_opencl()
        self.set_prior('tc', 'NP', zero_epoch.n, zero_epoch.s)
        self.set_prior('p', 'NP', period.n, period.s)
        self.set_prior('rho', 'UP', 10, 15)
        self.set_prior('k2_app', 'UP', 0.2**2, 0.4**2)
        self.set_prior('k2_app_tess', 'UP', 0.2**2, 0.4**2)
        self.set_prior('teff_h', 'NP', star_teff.n, star_teff.s)

    def create_pv_population(self, npv: int = 50) -> ndarray:
        pvp = super().create_pv_population(npv)
        for p in self.ps[self._sl_lm]:
            if 'lm_i' in p.name:
                pvp[:, p.pid] = 0.01 * (pvp[:, p.pid] - 1.0) + 1.0
            else:
                pvp[:, p.pid] *= 0.01
        return pvp

from scipy.constants import c,h,k,G

def calculate_beaming_alpha(teff: int = 3300):
    spfile = Path(cn.__file__).parent / 'data' / 'spectra.h5'
    spectra = pd.read_hdf(spfile)
    spectrum = spectra[teff]
    wl_nm = spectrum.index.values
    wl = wl_nm * 1e-9
    fl = spectrum.values

    b = 5 + diff(log(fl)) / diff(log(wl))
    w = tess(wl_nm)[1:] * wl[1:] * fl[1:]
    return sum(w * b) / sum(w)

@njit
def summed_planck(teff, wl, tm):
    teff = atleast_1d(teff)
    flux = zeros(teff.shape[0])
    for i in range(flux.size):
        flux[i] = sum(tm*(2*h*c**2 / wl**5 / (exp(h*c / (wl*k*teff[i])) - 1.)))
    return flux

def boosting_amplitude(mp, ms, period, alpha):
    """
    mp: float or ndarray
        Planetary mass [MJup]
    ms: float or ndarray
        Stellar mass [MSun]
    period: float or ndarray
        Orbital period [d]
    alpha: float or ndarray
        Doppler boosting alpha [-]
    """
    return alpha / c *(2*pi*G/(d2s*period))**(1/3) * ((mp*mj2kg)/(ms*ms2kg)**(2/3))

def ev_amplitude(mp, ms, a, u=0.55, g=0.3):
    ae = 0.15 * (15 + u) * (1 + g) / (3 - g)
    return ae * (mp*mj2kg)/(ms*ms2kg) * a**-3


class PhaseLPF(BaseLPF):
    def __init__(self, name: str):
        times, fluxes, _, _, _, _ = read_tess(tess_file, zero_epoch, period, use_pdc=True, baseline_duration_d=period)
        phase, time, flux = [], [], []
        for t, f in zip(times, fluxes):
            ph = (fold(t, period.n, zero_epoch.n, 0.5) - 0.5) * period.n
            mask = abs(ph) > 0.03
            phase.append(ph[mask])
            time.append(t[mask])
            flux.append(f[mask])
        super().__init__(name, ['TESS'], time, flux, wnids=arange(len(flux)), lnlikelihood='celerite')


    def _post_initialisation(self):
        self.t0 = 2458491.8771169
        self.period = 1.2652328
        self.k2 = 0.3 ** 2
        self.a = 10.
        self.phase = fold(self.timea, self.period, self.t0)
        self.em = UniformModel()
        self.em.set_data(self.timea)

        self._mec = self.em.evaluate(sqrt(self.k2), self.t0 + 0.5 * self.period, self.period, self.a, 0.5 * pi) - 1
        self._mec = 1 + self._mec / self._mec.ptp()

        phi = 2 * pi * self.phase
        self.alpha = a = abs(phi - pi)
        self._rff = (sin(a) + (pi - a) * cos(a)) / pi * self._mec  # Reflected light
        self._dbf = sin(phi)  # Doppler boosting
        self._evf = -cos(2 * phi)  # Ellipsoidal variations

        self._cwl = 1e-9 * tess.wl
        self._ctm = tess.tm

        self.set_prior('ms', 'NP', star_m.n, star_m.s)
        self.set_prior('teffh', 'NP', 3300, 100)
        self.set_prior('teffc', 'UP', 100, 3300)
        self.set_prior('gp_log10_wn', 'NP', -1.74, 0.15)


    def _init_parameters(self):
        self.ps = ParameterSet()
        ppc = [
            GParameter('ab', 'Bond albedo', '', UP(0, 1), (0, 1)),
            GParameter('mp', 'log10 planet mass', 'MJup', UP(log10(0.1), log10(300)), (0, inf)),
            GParameter('ms', 'Star mass', 'MSun', NP(1.0, 0.1), (0, inf)),
            GParameter('teffh', 'Host effective temperature', '', UP(2000, 10000), (0, inf)),
            GParameter('teffc', 'Companion effective temperature', '', UP(500, 10000), (0, inf)),
            GParameter('bl', 'Baseline level', '', NP(1, 0.005), (0, inf))]
        self.ps.add_global_block('phase_curve', ppc)
        self.ps.freeze()

    def baseline(self, pv):
        pv = atleast_2d(pv)
        return pv[:, 5:6]

    def emitted_flux_ratio(self, pv):
        pv = atleast_2d(pv)
        return summed_planck(pv[:, 4:5], self._cwl, self._ctm) / summed_planck(pv[:, 3:4], self._cwl, self._ctm)

    def emitted_light(self, pv):
        pv = atleast_2d(pv)
        return squeeze(self.emitted_flux_ratio(pv)[:, newaxis] * self.k2 * self._mec)

    def reflected_light(self, pv):
        pv = atleast_2d(pv)
        return squeeze(reflected_fr(self.a, pv[:, 0:1]) * self.k2 * self._rff)

    def boosting(self, pv):
        pv = atleast_2d(pv)
        a = boosting_amplitude(10 ** pv[:, 1:2], pv[:, 2:3], self.period, alpha=8.5)
        return squeeze(a * self._dbf)

    def ellipsoidal_variation(self, pv):
        pv = atleast_2d(pv)
        a = ev_amplitude(10 ** pv[:, 1:2], pv[:, 2:3], 10, 0.55, 0.3)
        return squeeze(a * self._evf)

    def phase_model(self, pv):
        pv = atleast_2d(pv)
        return squeeze(
            self.ellipsoidal_variation(pv) + self.boosting(pv) + self.reflected_light(pv) + self.emitted_light(pv))

    def flux_model(self, pv):
        pv = atleast_2d(pv)
        return squeeze(self.baseline(pv) + self.phase_model(pv))

    def create_pv_population(self, npop: int = 50):
        return self.ps.sample_from_prior(npop)