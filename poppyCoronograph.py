'''
Basic implementation of an Ideal AGPM
following the POPPY nomenclature (v0.5.1)

@author Gilles Orban de Xivry (ULg)
@date 12 / 02 / 2017
'''
from __future__ import division
import numpy as np
from poppy import AnalyticOpticalElement
from poppy.poppy_core import Wavefront, _IMAGE
from poppy import utils

import astropy.units as u


class IdealAGPM(AnalyticOpticalElement):
    """ Defines an ideal vortex phase mask coronagraph.

    Parameters
    ----------
    name : string
        Descriptive name
    wavelength : float
        Wavelength in meters.
    charge : int
        Charge of the vortex

    """
    @utils.quantity_input(wavelength=u.meter)
    def __init__(self, name="unnamed AGPM ",
                 wavelength=3.5e-6 * u.meter,
                 charge=2,
                 **kwargs):
        AnalyticOpticalElement.__init__(self, planetype=_IMAGE, **kwargs)
        self.name = name

        self.lp= charge
        self.central_wavelength= wavelength

    def getPhasor(self, wave):
        """
        Compute the amplitude transmission appropriate for a vortex for
        some given pixel spacing corresponding to the supplied Wavefront
        """

        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("AGPM getPhasor must be called with a Wavefront"
                             "to define the spacing")
        assert (wave.planetype == _IMAGE)

        y, x= self.get_coordinates(wave)
        phase = np.arctan2(y, x)

        AGPM_phasor = np.exp(1.j * self.lp * phase)

        idx= np.where(x==0)[0][0]
        idy= np.where(y==0)[0][0]
        AGPM_phasor[idx, idy]=0
        return AGPM_phasor

    def get_opd(self, wave):
        y, x= self.get_coordinates(wave)
        phase = np.arctan2(y, x)
        return self.lp * phase * self.central_wavelength.to(u.meter).value /\
            (2 * np.pi)

    def get_transmission(self, wave):
        y, x= self.get_coordinates(wave)
        trans= np.ones(y.shape)
        return trans
