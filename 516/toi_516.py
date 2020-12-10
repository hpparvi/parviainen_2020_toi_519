import sys
sys.path.append('..')
from numpy import array, arange, diff, sqrt, ndarray, inf, atleast_2d, zeros
from src.core import read_lco_table, N, read_tess, read_m2
from numba import njit

import pandas as pd

from typing import Union
from pathlib import Path
from astropy.units import Rjup, Rsun, m, AU
from numpy import diff, sqrt

from uncertainties import ufloat
from pytransit.lpf.tess.tgclpf import BaseTGCLPF, map_ldc, contaminate
from pytransit import QuadraticModelCL
from pytransit.param import UniformPrior as UP, NormalPrior as NP, GParameter
from pytransit.orbits.orbits_py import duration_eccentric, as_from_rhop, i_from_ba, i_from_baew, d_from_pkaiews, epoch

zero_epoch = ufloat(2458492.096558, 0.000239)
period = ufloat(0.8994850000000001, 1.6e-05)

star_teff = (3800, 100)
star_logg = (4.8, 0.1)
star_z    = (-1.5, 0.1)

rnep = (24622000 * m).to(Rjup)
rstar, rstare = 0.4*Rsun, 0.02*Rsun

root = Path(__file__).parent.resolve()
tess_file = root.joinpath('photometry/tess2019006130736-s0007-0000000348538431-0131-s_lc.fits').resolve()
reduced_m2_files = sorted(root.joinpath('results').glob('*fits'))

@njit(fastmath=True)
def map_pv_pclpf(pv):
    pv = atleast_2d(pv)
    pvt = zeros((pv.shape[0], 7))
    pvt[:, 0] = sqrt(pv[:, 5])
    b = pv[:, 3] * (1 + pvt[:, 0])  # grazing parameter -> impact parameter
    pvt[:, 1:3] = pv[:, 0:2]
    pvt[:, 3] = as_from_rhop(pv[:, 2], pv[:, 1])
    pvt[:, 4] = i_from_ba(pv[:, 3], pvt[:, 3])
    return pvt

def read_lco_data():
    dfg, cg = read_lco_table('photometry/TIC348538431-01_20190506_LCO-SSO-1m_gp_measurements.tbl')
    dfi, ci = read_lco_table('photometry/TIC348538431-01_20190326_LCO-CTIO-1m_i_measurements.csv', 'BJD-OBS', sep=',')
    dfs = dfg, dfi
    covs = [cg, ci]

    times = [df.iloc[:, 0].values for df in dfs]
    fluxes = [N(df.rel_flux_T1).values for df in dfs]

    # Cut the flare in LCO i observations
    # -----------------------------------
    m = ~ ((times[1] > 2458568.58) & (times[1] < 2458568.595))
    times[1] = times[1][m]
    fluxes[1] = fluxes[1][m]
    covs[1] = covs[1][m, :]

    ins = len(times) * ['LCO']
    piis = list(arange(len(times)))
    return times, fluxes, 'g i'.split(), [diff(f).std() / sqrt(2) for f in fluxes], covs, ins, piis

class LPF(BaseTGCLPF):
    def _post_initialisation(self):
        self.tm = QuadraticModelCL(klims=self.tm.klims)
        self.tm.set_data(self.timea, self.lcids, self.pbids, self.nsamples, self.exptimes)
        zero_epoch = ufloat(2458844.68771965, 0.005)
        self.set_prior('tc',  'NP', zero_epoch.n, zero_epoch.s)
        self.set_prior('p',   'NP', period.n, period.s)
        self.set_prior('rho', 'UP', 1, 15)
        self.set_prior('teff_h', 'NP', star_teff[0], star_teff[1])
        self.set_prior('k2_app', 'UP', 0.15**2, 0.75**2)
        self.set_prior('k2_app_tess', 'UP', 0.15**2, 0.75**2)  
        self.ps.bounds[3][1] = inf
        
    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [
            GParameter('tc',  'zero_epoch',        'd',      NP(0.0,  0.1), (-inf, inf)),
            GParameter('p',   'period',            'd',      NP(1.0, 1e-5), (0,    inf)),
            GParameter('rho', 'stellar_density',   'g/cm^3', UP(0.1, 25.0), (0,    inf)),
            GParameter('g',   'grazing parameter', '',       UP(0.0,  1.0), (0,      1))]
        self.ps.add_global_block('orbit', porbit)
        
    def create_pv_population(self, npv: int = 50) -> ndarray:
        pvp = super().create_pv_population(npv)
        for p in self.ps[self._sl_bl]:
            if 'bli' in p.name:
                pvp[:,p.pid] = 0.01*(pvp[:,p.pid] - 1.0) + 1.0
            else:
                pvp[:,p.pid] *= 0.01
        return pvp
    
    def transit_model(self, pvp):
        pvp = atleast_2d(pvp)
        cnt = zeros((pvp.shape[0], self.npb))
        pvt = map_pv_pclpf(pvp)
        pvt[:,1] -= self._tref
        ldc = map_ldc(pvp[:, self._sl_ld])
        flux = self.tm.evaluate_pv(pvt, ldc)
        cnt[:, 0] = 1 - pvp[:, 8] / pvp[:, 5]
        cnref = 1. - pvp[:, 4] / pvp[:, 5]
        cnt[:, 1:] = self.cm.contamination(cnref, pvp[:, 6], pvp[:, 7])
        return contaminate(flux, cnt, self.lcids, self.pbids)
        
class LPFTM(LPF):
    def read_data(self):
        times_t, fluxes_t, pbs_t, wns_t, ins_t, piis_t = read_tess(tess_file, zero_epoch.n, period.n, 
                                                    baseline_duration_d=0.1, transit_duration_d=0.06)
        times_m, fluxes_m, pbs_m, wns_m, covs_m , ins_m, piis_m = read_m2(reduced_m2_files, 60)
        times = times_t + times_m
        fluxes= fluxes_t + fluxes_m
        pbs = pbs_t + pbs_m
        wns = wns_t + wns_m
        covs = len(times_t)*[array([[]])] + covs_m
        pbnames = 'tess g r i z_s'.split()
        self._stess = len(times_t)
        self._ntess = sum([t.size for t in times_t])
        self.ins =  ins_t + ins_m
        self.piis = piis_t + piis_m
        return times, fluxes, pbnames, pbs, wns, covs 
    
class LPFTML(LPF):  
    def read_data(self):
        times_t, fluxes_t, pbs_t, wns_t, ins_t, piis_t = read_tess(tess_file, zero_epoch.n, period.n, 
                                                    baseline_duration_d=0.1, transit_duration_d=0.06)
        times_m, fluxes_m, pbs_m, wns_m, covs_m , ins_m, piis_m = read_m2(reduced_m2_files, 60)
        times_l, fluxes_l, pbs_l, wns_l, covs_l, ins_l, piis_l = read_lco_data()
        times = times_t + times_m + times_l
        fluxes= fluxes_t + fluxes_m + fluxes_l
        pbs = pbs_t + pbs_m + pbs_l
        wns = wns_t + wns_m + wns_l
        covs = len(times_t)*[array([[]])] + covs_m + covs_l
        pbnames = 'tess g r i z_s'.split()
        self._stess = len(times_t)
        self._ntess = sum([t.size for t in times_t])
        self.ins =  ins_t + ins_m +  ins_l
        self.piis = piis_t + piis_m + piis_l
        return times, fluxes, pbnames, pbs, wns, covs 