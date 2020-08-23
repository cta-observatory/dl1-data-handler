from ctapipe.io.eventsource import EventSource
import uproot
from ctapipe.containers import DataContainer, TelescopePointingContainer
from ctapipe.instrument import TelescopeDescription, SubarrayDescription, OpticsDescription, CameraGeometry, CameraReadout, CameraDescription
import glob
import re
import numpy as np
from ctapipe.core import Container, Field
from numpy import nan
from astropy import units as u
from astropy.coordinates import Angle
from scipy.stats import norm
X_MAX_UNIT = u.g / (u.cm ** 2)

class MCHeaderContainer(Container):
    corsika_version = Field(nan, "CORSIKA version *  1000")
    refl_version = Field(nan, "refl version")
    cam_version = Field(nan, "camera version")
    run_number = Field(nan, "MC Run Number")
    prod_site = Field(nan, "Production site")
    date_run_mmcs = Field(nan, "Date Run MMCs")
    date_run_cam = Field(nan, "Date Run Camera")
    shower_theta_max = Field(nan * u.deg, "Shower Theta Maximum", unit = u.deg)
    shower_theta_min = Field(nan * u.deg, "Shower Theta Minimum", unit = u.deg)
    shower_phi_max = Field(nan * u.deg, "Shower Phi Maximum", unit = u.deg)
    shower_phi_min = Field(nan * u.deg, "Shower Phi Minimum", unit = u.deg)
    c_wave_lower = Field(nan, "C Wave Lower")
    c_wave_upper = Field(nan, "C Wave Upper")
    num_obs_lev = Field(nan, "Number Observations Level")
    height_lev = Field([], "Height Level")
    slope_spec = Field(nan, "Slope Spec")
    rand_pointing_cone_semi_angle = Field(nan * u.deg, "Random Pointing Cone Semi Angle", unit = u.deg)
    impact_max = Field(nan, "Imapact Maximum")
    star_field_rotate = Field(nan, "Star Field Rotate")
    star_field_ra_h = Field(nan, "Star Field RA H")
    star_field_ra_m = Field(nan, "Star Field RA M")
    star_field_ra_s = Field(nan, "Star Field RA S")
    star_field_dec_d = Field(nan, "Star Field DEC D")
    star_field_dec_m = Field(nan, "Star Field DEC M")
    star_field_dec_s = Field(nan, "Star Field DEC S")
    num_trig_cond = Field(nan, "Number Trigger Condition")
    all_evts_trig = Field(nan,  "All Events Triggered")
    mc_evt = Field(nan, "MC Event")
    mc_trig = Field(nan, "MC Trigger")
    mc_fadc = Field(nan, "MC Fadc")
    raw_evt = Field(nan, "Raw Event")
    num_analised_pix = Field(nan, "Number of Analised Pixels")
    num_simulated_showers = Field(nan, "Number of Simulated Showers")
    num_stored_showers = Field(nan, "Number of Stored Showers")
    num_events = Field(nan, "Number of Events")
    num_phe_from_dnsb = Field(nan, "Number Phe from DNSB")
    elec_noise = Field(nan, "Elec Noise")
    optic_links_noise = Field(nan, "Optic Links Noise")


class DLMAGICEventSource(EventSource):
    def __init__(self, **kwargs):
        """
        Constructor
        
        Parameters
        ----------
        kwargs: dict
            Parameters to be passed.
            NOTE: The file mask of the data to read can be passed with
            the 'input_url' parameter.
        """
        self.file_list = glob.glob(kwargs['input_url'])
        self.file_list.sort()

        # Since EventSource can not handle file wild cards as input_url
        # We substitute the input_url with first file matching
        # the specified file mask.
        
        del kwargs['input_url']
        super().__init__(input_url=self.file_list[0], **kwargs)
        
        # get run number
        mask = r".*_M\d_za\d+to\d+_\d_(\d+)_Y_.*"
        parsed_info = re.findall(mask, self.file_list[0])
        self.run_number = parsed_info[0]
        
        # MAGIC telescope positions in m wrt. to the center of CTA simulations

        self.magic_tel_positions = {
            1: [-27.24, -146.66, 50.00] * u.m,
            2: [-96.44, -96.77, 51.00] * u.m
        }
        self.magic_tel_positions = self.magic_tel_positions
        # MAGIC telescope description
        optics = OpticsDescription.from_name('MAGIC')
        geom = CameraGeometry.from_name('MAGICCam')
        # Camera Readout for NectarCam used as a placeholder
        readout = CameraReadout('MAGICCam', sampling_rate = u.Quantity(1, u.GHz), reference_pulse_shape = np.array([norm.pdf(np.arange(96),48,6)]), reference_pulse_sample_width = u.Quantity(1, u.ns))
        camera = CameraDescription('MAGICCam', geom, readout)
        self.magic_tel_description = TelescopeDescription(name='MAGIC', tel_type = 'LST', optics=optics, camera=camera)
        self.magic_tel_descriptions = {1: self.magic_tel_description, 2: self.magic_tel_description}
        self.magic_subarray = SubarrayDescription('MAGIC', self.magic_tel_positions, self.magic_tel_descriptions)
        # Open ROOT files
        file1 = uproot.open(self.file_list[0])
        self.eventM1 = file1["Events"]
        file2 = uproot.open(self.file_list[1])
        self.eventM2 = file2["Events"]
        self.meta = file1["RunHeaders"]
        self._mc_header = self._parse_mc_header()
        
    @property
    def is_simulation(self):
        """
        Whether the currently open file is simulated


        Returns
        -------
        bool

        """
        return True

    @property
    def datalevels(self):
        """
        The datalevels provided by this event source


        Returns
        -------
        tuple[str]

        """
        return ('R0','R1','DL0')
    @property
    def subarray(self):
        """
        Obtain the subarray from the EventSource


        Returns
        -------
        ctapipe.instrument.SubarrayDescription

        """
        return self.magic_subarray

    @property
    def obs_id(self):
        """
        The current observation id


        Returns
        -------
        int

        """
        return self.run_number

    @staticmethod
    def is_compatible(file_mask):
        """
        This method checks if the specified file mask corresponds
        to MAGIC data files. The result will be True only if all
        the files are of ROOT format and contain an 'Events' tree.
        Parameters
        ----------
        file_mask: str
            A file mask to check
        Returns
        -------
        bool:
            True if the masked files are MAGIC data runs, False otherwise.
        """

        is_magic_root_file = True

        file_list = glob.glob(file_mask)

        for file_path in file_list:
            try:
                import uproot

                try:
                    with uproot.open(file_path) as input_data:
                        if 'Events' not in input_data:
                            is_magic_root_file = False
                except ValueError:
                    # uproot raises ValueError if the file is not a ROOT file
                    is_magic_root_file = False
                    pass

            except ImportError:
                if re.match(r'.+_m\d_.+root', file_path.lower()) is None:
                    is_magic_root_file = False

        return is_magic_root_file  

    def _generator(self):
        """
        Stereo event generator. Yields DataContainer instances, filled
        with the read event data.
        
        Returns
        -------
        
        """  
        counter = 0
        data = DataContainer()
        data.meta['origin'] = "MAGIC"
        data.meta['input_url'] = self.input_url
        data.meta['is_simulation'] = True
        data.mcheader = self._mc_header
        #Reading data from root file for Events table
        eventidM1 = np.asarray(self.eventM1["MRawEvtHeader.fStereoEvtNumber"].array())
        eventidM2 = np.asarray(self.eventM2["MRawEvtHeader.fStereoEvtNumber"].array())
        
        zenith = np.asarray(self.eventM1["MMcEvt.fTheta"].array())
        
        pointing_altitude = np.asarray(self.eventM1["MPointingPos.fZd"].array())
        
        azimuth = np.asarray(self.eventM1["MMcEvt.fPhi"].array())
        
        pointing_azimuth = np.asarray(self.eventM1["MPointingPos.fAz"].array())
        
        core_x = np.asarray(self.eventM1["MMcEvt.fCoreX"].array())
        core_y = np.asarray(self.eventM1["MMcEvt.fCoreY"].array())
        
        mc_energy = np.asarray(self.eventM1["MMcEvt.fEnergy"].array())/1000
        h_first_int = np.asarray(self.eventM1["MMcEvt.fZFirstInteraction"].array())
        
        mask = r".([A-Z]+)_M\d_za\d+to\d+_\d_\d+_Y_.*"
        primary_id = re.findall(mask, self.file_list[0])[0]
        if primary_id == 'GA':
            shower_primary_id = 1
            
        stereo_total = np.max(eventidM1)
        event_index = np.zeros(shape = (stereo_total,1))
        
        #Reading data from root file for Image table
        
        chargeM1 = self.eventM1["MCerPhotEvt.fPixels.fPhot"].array()
        peak_timeM1 = self.eventM1["MArrivalTime.fData"].array()
        chargeM1 = np.asarray(chargeM1)
        peak_timeM1 = np.asarray(peak_timeM1)
        
        chargeM2 = self.eventM2["MCerPhotEvt.fPixels.fPhot"].array()
        peak_timeM2 = self.eventM2["MArrivalTime.fData"].array()
        chargeM2 = np.asarray(chargeM2)
        peak_timeM2 = np.asarray(peak_timeM2)
        
        total_events = len(self.eventM1["MCerPhotEvt.fPixels.fPhot"].array())
        #Iterating over all events, and saving only stereo ones
        tels_in_file = ["m1", "m2"]
        tels_with_data = {1,2}
        for i in range(0, total_events):
            if eventidM1[i] != 0:
                obs_id = self.run_number
                event_id = eventidM1[i]
                i2 = np.where(eventidM2==eventidM1[i])
                i2 = int(i2[0])
                data.count = counter

                # Setting up the Data container
                data.index.obs_id = obs_id
                data.index.event_id = event_id
                data.r0.tel.clear()
                data.r1.tel.clear()
                data.dl0.tel.clear()
                
                # Filling the DL1 container with the event data
                for tel_i, tel_id in enumerate(tels_in_file):
                    
                    #Adding telescope pointing container

                    data.pointing.tel[tel_i+1].azimuth = u.Quantity(np.deg2rad(pointing_azimuth[i]), u.rad)
                    data.pointing.tel[tel_i+1].altitude = u.Quantity(np.deg2rad(90 - pointing_altitude[i]), u.rad)
                    
                    
                    #Adding MC data
                    data.mc.alt = Angle(np.pi/2 - zenith[i], u.rad)
                    data.mc.az = Angle(np.deg2rad(180-7) - azimuth[i], u.rad)
                    data.mc.x_max = u.Quantity(0, X_MAX_UNIT)
                    data.mc.h_first_int = u.Quantity(h_first_int[i], u.m)
                    data.mc.core_x = u.Quantity(core_x[i], u.m)
                    data.mc.core_y = u.Quantity(core_y[i], u.m)
                    data.mc.energy = u.Quantity(mc_energy[i], u.TeV)
                    data.mc.shower_primary_id = shower_primary_id
                    # Adding event charge and peak positions per pixel
                    if tel_i == 0:
                        data.dl1.tel[tel_i + 1].image = chargeM1[i][:1039]
                        data.dl1.tel[tel_i + 1].peak_time = peak_timeM1[i][:1039]
                    else:
                        data.dl1.tel[tel_i + 1].image = chargeM2[i][:1039]
                        data.dl1.tel[tel_i + 1].peak_time = peak_timeM2[i][:1039]                      
                
                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data


                yield data
                counter += 1
        return
    def _parse_mc_header(self):
        return MCHeaderContainer(
            corsika_version = self.meta["MMcRunHeader.fCorsikaVersion"].array()[0],
            refl_version = self.meta["MMcRunHeader.fReflVersion"].array()[0],
            cam_version = self.meta["MMcRunHeader.fCamVersion"].array()[0],
            run_number = self.meta["MMcRunHeader.fMcRunNumber"].array()[0],
            prod_site = self.meta["MMcRunHeader.fProductionSite"].array()[0],
            date_run_mmcs = self.meta["MMcRunHeader.fDateRunMMCs"].array()[0],
            date_run_cam = self.meta["MMcRunHeader.fDateRunCamera"].array()[0],
            shower_theta_max = Angle(self.meta["MMcRunHeader.fShowerThetaMax"].array()[0], u.deg),
            shower_theta_min = Angle(self.meta["MMcRunHeader.fShowerThetaMin"].array()[0], u.deg),
            shower_phi_max = Angle(self.meta["MMcRunHeader.fShowerPhiMax"].array()[0], u.deg),
            shower_phi_min = Angle(self.meta["MMcRunHeader.fShowerPhiMin"].array()[0], u.deg),
            c_wave_lower = self.meta["MMcRunHeader.fCWaveLower"].array()[0],
            c_wave_upper = self.meta["MMcRunHeader.fCWaveUpper"].array()[0],
            num_obs_lev = self.meta["MMcRunHeader.fNumObsLev"].array()[0],
            height_lev = self.meta["MMcRunHeader.fHeightLev[10]"].array(),
            slope_spec = self.meta["MMcRunHeader.fSlopeSpec"].array()[0],
            rand_pointing_cone_semi_angle = Angle(self.meta["MMcRunHeader.fRandomPointingConeSemiAngle"].array()[0], u.deg),
            impact_max = self.meta["MMcRunHeader.fImpactMax"].array()[0],
            star_field_rotate = self.meta["MMcRunHeader.fStarFieldRotate"].array()[0],
            star_field_ra_h = self.meta["MMcRunHeader.fStarFieldRaH"].array()[0],
            star_field_ra_m = self.meta["MMcRunHeader.fStarFieldRaM"].array()[0],
            star_field_ra_s = self.meta["MMcRunHeader.fStarFieldRaS"].array()[0],
            star_field_dec_d = self.meta["MMcRunHeader.fStarFieldDeD"].array()[0],
            star_field_dec_m = self.meta["MMcRunHeader.fStarFieldDeM"].array()[0],
            star_field_dec_s = self.meta["MMcRunHeader.fStarFieldDeS"].array()[0],
            num_trig_cond = self.meta["MMcRunHeader.fNumTrigCond"].array()[0],
            all_evts_trig = self.meta["MMcRunHeader.fAllEvtsTriggered"].array()[0],
            mc_evt = self.meta["MMcRunHeader.fMcEvt"].array()[0],
            mc_trig = self.meta["MMcRunHeader.fMcTrig"].array()[0],
            mc_fadc = self.meta["MMcRunHeader.fMcFadc"].array()[0],
            raw_evt = self.meta["MMcRunHeader.fRawEvt"].array()[0],
            num_analised_pix = self.meta["MMcRunHeader.fNumAnalisedPixels"].array()[0],
            num_simulated_showers = self.meta["MMcRunHeader.fNumSimulatedShowers"].array()[0],
            num_stored_showers = self.meta["MMcRunHeader.fNumStoredShowers"].array()[0],
            num_events = self.meta["MMcRunHeader.fNumEvents"].array()[0],
            num_phe_from_dnsb = self.meta["MMcRunHeader.fNumPheFromDNSB"].array()[0],
            elec_noise = self.meta["MMcRunHeader.fElecNoise"].array()[0],
            optic_links_noise = self.meta["MMcRunHeader.fOpticLinksNoise"].array()[0] )
