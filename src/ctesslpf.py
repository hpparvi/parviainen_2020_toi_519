"""Contaminated TESS LPF

This module contains the log posterior function (LPF) for a TESS light curve with possible third-light
contamination from an unresolved source inside the photometry aperture.
"""
from pathlib import Path
from typing import Union

from astropy.io.fits import getval
from numpy import repeat, inf, where, newaxis, squeeze, atleast_2d, isfinite, concatenate, zeros
from numpy.random.mtrand import uniform
from pytransit.lpf.tesslpf import TESSLPF
from pytransit.param import UniformPrior as UP, NormalPrior as NP, PParameter
from ldtk import tess
from uncertainties import ufloat, UFloat

class CTESSLPF(TESSLPF):
    """Contaminated TESS LPF

    This class implements a log posterior function for a TESS light curve that allows for unknown flux contamination.
    The amount of flux contamination is not constrained.
    """
    def __init__(self, name: str, fname: Union[Path, str], zero_epoch: Union[float, UFloat], period: Union[float, UFloat],
                 nsamples: int = 2, bldur: float = 0.1, trdur: float = 0.04, use_pdc: bool = True):

        if isinstance(zero_epoch, float):
            zero_epoch: UFloat = ufloat(zero_epoch, 1e-3)

        if isinstance(period, float):
            period: UFloat = ufloat(period, 1e-5)

        super().__init__(name, fname, zero_epoch=zero_epoch.n, period=period.n, nsamples=nsamples,
                         bldur=bldur, trdur=trdur, use_pdc=use_pdc)

        self.set_prior('zero_epoch', NP(zero_epoch.n, 3 * zero_epoch.s))
        self.set_prior('period', NP(period.n, 3 * period.s))
        self.set_prior('k2_true', UP(0.1 ** 2, 0.75 ** 2))

    def _init_p_planet(self):
        ps = self.ps
        pk2 = [PParameter('k2_true', 'true_area_ratio', 'A_s', UP(0.10 ** 2, 0.75 ** 2), (0.10 ** 2, 0.75 ** 2)),
               PParameter('k2_app', 'apparent_area_ratio', 'A_s', UP(0.10 ** 2, 0.50 ** 2), (0.10 ** 2, 0.50 ** 2))]
        ps.add_passband_block('k2', 1, 2, pk2)
        self._pid_k2 = repeat(ps.blocks[-1].start, self.npb)
        self._start_k2 = ps.blocks[-1].start
        self._sl_k2 = ps.blocks[-1].slice
        self.add_prior(lambda pv: where(pv[:, 5] < pv[:, 4], 0, -inf))

    def add_ldtk_prior(tetff, logg, z):
        super().add_ldtk_prior(teff, logg, z, passbands=(tess,))
        
    def transit_model(self, pv):
        pv = atleast_2d(pv)
        flux = super().transit_model(pv)
        cnt = 1. - pv[:, 5] / pv[:, 4]
        return squeeze(cnt[:, newaxis] + (1. - cnt[:, newaxis]) * flux)

    def create_pv_population(self, npop=50):
        pvp = zeros((0, len(self.ps)))
        npv, i = 0, 0
        while npv < npop and i < 10:
            pvp_trial = self.ps.sample_from_prior(npop)
            pvp_trial[:, 5] = pvp_trial[:, 4]
            cref = uniform(0, 0.99, size=npop)
            pvp_trial[:, 4] = pvp_trial[:, 5] / (1. - cref)
            lnl = self.lnposterior(pvp_trial)
            ids = where(isfinite(lnl))
            pvp = concatenate([pvp, pvp_trial[ids]])
            npv = pvp.shape[0]
            i += 1
        pvp = pvp[:npop]
        return pvp
