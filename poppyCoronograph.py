import numpy as np
from poppy import AnalyticOpticalElement
from poppy.poppy_core import Wavefront, _IMAGE
from poppy import utils

import astropy.units as u


class IdealAGPM(AnalyticOpticalElement):
    """ Defines an ideal 4-quadrant phase mask coronagraph, with its retardance
    set perfectly to 0.5 waves at one specific wavelength and varying linearly on
    either side of that.  "Ideal" in the sense of ignoring chromatic effects other
    than just the direct scaling of the wavelength.

    Parameters
    ----------
    name : string
        Descriptive name
    wavelength : float
        Wavelength in meters for which the FQPM was designed, and at which there
        is exactly 1/2 a wave of retardance.

    """
    @utils.quantity_input(wavelength=u.meter)
    def __init__(self, name="unnamed AGPM ",
                 wavelength=3.5e-6 * u.meter,
                 charge=2,
                 **kwargs):
        AnalyticOpticalElement.__init__(self, planetype=_IMAGE, **kwargs)
        self.name = name

        # self.central_wavelength = wavelength
        self.lp= charge
        self.central_wavelength= wavelength

    def getPhasor(self, wave):
        """
        Compute the amplitude transmission appropriate for a 4QPM for
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
        return self.lp * phase * self.central_wavelength.to(u.meter).value / (2 * np.pi)

    def get_transmission(self, wave):
        y, x= self.get_coordinates(wave)
        trans= np.ones(y.shape)
#         idx= np.where(x==0)[0][0]
#         idy= np.where(y==0)[0][0]
#         trans[idx, idy]= 0
        return trans
#         return np.angle(self.getPhasor(wave)) *\
#             self.central_wavelength.to(u.meter).value


if __name__ == '__main__':
    '''
        11-07-2016 : with current poppy version, if pyfftw installed, this is failing.
            -> 'pip3 uninstall pyfftw'....
    '''
    import poppy
    import matplotlib.pyplot as plt
    N= 264
    D= 3
    cobs= 0.24
    fLyot= 0.9
    wvl= 3.2e-6
    doFresnel= False
    if doFresnel is False:
        ''' Define pupil stops '''
        primary = poppy.CircularAperture(radius=D, pad_factor=4)
        # sec= poppy.SecondaryObscuration(secondary_radius=D * cobs)
        sec= poppy.InverseTransmission(poppy.CircularAperture(radius= D * cobs))
        aperture = poppy.CompoundAnalyticOptic(name='Entrance Pupil',
                                                opticslist = [primary, sec])
        lyot0= poppy.CircularAperture(radius=D * fLyot)
        lyot1= poppy.SecondaryObscuration(secondary_radius=D * cobs / fLyot)
        lyot= poppy.CompoundAnalyticOptic(name='Lyot pupil',
                                          opticslist = [lyot0, lyot1])

        ''' Define optical planes '''
        optsys = poppy.OpticalSystem("Test", npix=N)
        optsys.add_pupil(aperture)
        optsys.add_pupil(poppy.FQPM_FFT_aligner())
        optsys.addImage(IdealAGPM(wavelength=wvl))
        # optsys.add_image(poppy.IdealFQPM(wavelength=wvl))
        optsys.add_pupil(poppy.FQPM_FFT_aligner(direction='backward'))
        optsys.add_pupil(lyot)
        optsys.add_detector(pixelscale=0.001, fov_arcsec=2.0)

        ''' Shift on-axis source '''
        optsys.source_offset_r= 0.022 / 2    # [arcsec]

        ''' Compute and display '''
        plt.figure()
        # optsys.display()

        psf = optsys.calcPSF(wavelength=wvl, display_intermediates=False)
        poppy.display_PSF(psf, vmax=0.95, normalize='peak', scale='linear')
    else:
        z= 600 * u.mm

        primary = poppy.CircularAperture(radius=D)  # , pad_factor=4)
        sec= poppy.InverseTransmission(poppy.CircularAperture(radius= D * cobs))
        aperture = poppy.CompoundAnalyticOptic(name='Entrance Pupil',
                                               opticslist = [primary, sec])

        lyot0= poppy.CircularAperture(radius=D * fLyot)
        lyot1= poppy.SecondaryObscuration(secondary_radius=D * cobs / fLyot)
        lyot= poppy.CompoundAnalyticOptic(name='Lyot pupil',
                                          opticslist = [lyot0, lyot1])


        optsys= poppy.FresnelOpticalSystem(pupil_diameter= D * u.m,
                                           npix= N, beam_ratio=0.25)
        lens1= poppy.QuadraticLens(z, 'Converging lens')
        optsys.add_optic(aperture)
        optsys.add_optic(lens1, distance=z)
        optsys.add_optic(IdealAGPM(wavelength=wvl), distance=z)
        optsys.add_optic(lyot, distance=z)
        optsys.add_optic(lens1, distance=z)
        # optsys.add_optic(poppy._IMAGE, distance=z)
        # optsys.add_optic(poppy.ScalarTransmission(planetype=poppy.PlaneType.image, name='focus'),
        #               distance=z)
        # optsys.add_detector(pixelscale=0.001, distance= z)

        # optsys.source_offset_r= 0.022 / 2 
        psf = optsys.calcPSF(wavelength=wvl, display_intermediates=False)
        poppy.display_PSF(psf, vmax=0.95, normalize='peak', scale='linear')

#         wf= poppy.FresnelWavefront(D * u.m,
#                                    wavelength= wvl,
#                                    npix= N,
#                                    oversample=4)
#         wf *= aperture
#         wf.propagate_fresnel(z)
# 
#         conv_lens= poppy.QuadraticLens(z)
#         wf*= conv_lens
#         wf.propagate_fresnel(z)
#         vortex= IdealAGPM(wavelength=wvl)
#         wf *= vortex
#         wf.propagate_fresnel(z)
# 
#         conv_lens= poppy.QuadraticLens(z)
#         wf*= conv_lens
#         wf.propagate_fresnel(z)
#         wf*=lyot
#         wf.propagate_fresnel(z)
#         wf*= conv_lens
#         wf.propagate_fresnel(z)
# 
#         plt.figure(figsize=(10,5))
#         wf.display('both', colorbar=True, imagecrop=4e-5, scale='linear')
#         plt.suptitle("Wavefront at focus of lens", fontsize=18)
        pass

