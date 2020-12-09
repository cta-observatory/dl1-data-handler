# If you would like to use the DLMAGICEventSource with superstar files,
# make sure you had run "star" with flags "-saveimages -saveimagesclean -savecerevt".
# i.e.:
# $ star -b -f -mc -q -saveimages -saveimagesclean -savecerevt --config=mrcfiles/star_M{1,2}_OSA.rc --ind="/home/tjark/MAGIC_files/cal/*M{1,2}*_Y_*.root" --out="/home/tjark/MAGIC_files/starM{1,2}/" --log=/home/tjark/MAGIC_files/starM{1,2}/LogFile.txt
# $ superstar -q -b -f -mc --config=mrcfiles/superstar.rc --ind1=/home/tjark/MAGIC_files/starM1/GA_M1_za05to35_8_*_I_w0.root --ind2=/home/tjark/MAGIC_files/starM2/GA_M2_za05to35_8_*_I_w0.root --out=/home/tjark/MAGIC_files/superstar/ --log=/home/tjark/MAGIC_files/superstar/logfile.txt

from astropy import units as u
from astropy.coordinates import Angle
from ctapipe.containers import DataContainer, TelescopePointingContainer, LeakageContainer, HillasParametersContainer, ConcentrationContainer, TimingParametersContainer, MorphologyContainer
from ctapipe.instrument import TelescopeDescription, SubarrayDescription, OpticsDescription, CameraGeometry, CameraReadout, CameraDescription
from ctapipe.io.eventsource import EventSource
from dl1_data_handler import containers
import glob
import numpy as np
from scipy.stats import norm
import re
X_MAX_UNIT = u.g / (u.cm ** 2)


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
        try:
            import uproot
        except ImportError:
            raise ImportError("The 'uproot' package is required for the DLMAGICEventSource class.")

        self.file_list = glob.glob(kwargs['input_url'])
        self.file_list.sort()

        # Since EventSource can not handle file wild cards as input_url
        # We substitute the input_url with first file matching
        # the specified file mask.
        del kwargs['input_url']
        super().__init__(input_url=self.file_list[0], **kwargs)

        # get run number
        mask = r".*_za\d+to\d+_\d_(\d+)_([A-Z]+)_.*"
        parsed_info = re.findall(mask, self.file_list[0])
        self.run_number = parsed_info[0][0]

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
        self.calib_M1, self.calib_M2, self.star_M1, self.star_M2, self.superstar = None, None, None, None, None
        for file in self.file_list:
            uproot_file = uproot.open(file)
            if "_Y_" in file:
                if "_M1_" in file:
                    self.calib_M1 = uproot_file["Events"]
                    self.meta = uproot_file["RunHeaders"]
                elif "_M2_" in file:
                    self.calib_M2 = uproot_file["Events"]
            if "_I_"in file:
                if "_M1_" in file:
                    self.star_M1 = uproot_file["Events"]
                elif "_M2_" in file:
                    self.star_M2 = uproot_file["Events"]
            if "_S_" in file:
                self.superstar = uproot_file["Events"]
                self.meta = uproot_file["RunHeaders"]
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
    #This function was taken from the general MAGICEventSource by ctapipe_io_magic (https://github.com/cta-observatory/ctapipe_io_magic).
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
                with uproot.open(file_path) as input_data:
                    if 'Events' not in input_data:
                        is_magic_root_file = False
            except ValueError:
                    # uproot raises ValueError if the file is not a ROOT file
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

        if self.calib_M1 is not None and self.calib_M2 is not None:
            #Reading data from root file for Events table
            eventid_M1 = np.asarray(self.calib_M1["MRawEvtHeader.fStereoEvtNumber"].array())
            eventid_M2 = np.asarray(self.calib_M2["MRawEvtHeader.fStereoEvtNumber"].array())
            zenith = np.asarray(self.calib_M1["MMcEvt.fTheta"].array())
            pointing_altitude = np.asarray(self.calib_M1["MPointingPos.fZd"].array())
            azimuth = np.asarray(self.calib_M1["MMcEvt.fPhi"].array())
            pointing_azimuth = np.asarray(self.calib_M1["MPointingPos.fAz"].array())
            core_x = np.asarray(self.calib_M1["MMcEvt.fCoreX"].array())
            core_y = np.asarray(self.calib_M1["MMcEvt.fCoreY"].array())
            mc_energy = np.asarray(self.calib_M1["MMcEvt.fEnergy"].array())/1000.0
            h_first_int = np.asarray(self.calib_M1["MMcEvt.fZFirstInteraction"].array())

            #Reading data from root file for Image table
            charge_M1 = np.asarray(self.calib_M1["MCerPhotEvt.fPixels.fPhot"].array())
            peak_time_M1 = np.asarray(self.calib_M1["MArrivalTime.fData"].array())
            charge_M2 = np.asarray(self.calib_M2["MCerPhotEvt.fPixels.fPhot"].array())
            peak_time_M2 = np.asarray(self.calib_M2["MArrivalTime.fData"].array())

        if self.superstar is not None:
            #Reading data from root file for Events table
            eventid_M1 = np.asarray(self.superstar["MRawEvtHeader_1.fStereoEvtNumber"].array())
            eventid_M2 = np.asarray(self.superstar["MRawEvtHeader_2.fStereoEvtNumber"].array())
            zenith = np.asarray(self.superstar["MMcEvt_1.fTheta"].array())
            pointing_altitude = np.asarray(self.superstar["MPointingPos_1.fZd"].array())
            azimuth = np.asarray(self.superstar["MMcEvt_1.fPhi"].array())
            pointing_azimuth = np.asarray(self.superstar["MPointingPos_1.fAz"].array())
            core_x = np.asarray(self.superstar["MMcEvt_1.fCoreX"].array())
            core_y = np.asarray(self.superstar["MMcEvt_1.fCoreY"].array())
            mc_energy = np.asarray(self.superstar["MMcEvt_1.fEnergy"].array())/1000.0
            h_first_int = np.asarray(self.superstar["MMcEvt_1.fZFirstInteraction"].array())

            #Reading data from root file for Parameter table
            hillas_intensity_M1 = np.asarray(self.superstar["MHillas_1.fSize"].array())
            hillas_intensity_M2 = np.asarray(self.superstar["MHillas_2.fSize"].array())
            hillas_x_M1 = np.asarray(self.superstar["MHillas_1.fMeanX"].array())
            hillas_x_M2 = np.asarray(self.superstar["MHillas_2.fMeanX"].array())
            hillas_y_M1 = np.asarray(self.superstar["MHillas_1.fMeanY"].array())
            hillas_y_M2 = np.asarray(self.superstar["MHillas_2.fMeanY"].array())
            hillas_r_M1 = np.sqrt(np.power(hillas_x_M1,2)+np.power(hillas_y_M1,2))
            hillas_r_M2 = np.sqrt(np.power(hillas_x_M2,2)+np.power(hillas_y_M2,2))
            hillas_phi_M1 = np.arctan2(hillas_y_M1, hillas_x_M1)
            hillas_phi_M2 = np.arctan2(hillas_y_M2, hillas_x_M2)
            hillas_length_M1 = np.asarray(self.superstar["MHillas_1.fLength"].array())
            hillas_length_M2 = np.asarray(self.superstar["MHillas_2.fLength"].array())
            hillas_width_M1 = np.asarray(self.superstar["MHillas_1.fWidth"].array())
            hillas_width_M2 = np.asarray(self.superstar["MHillas_2.fWidth"].array())
            hillas_psi_M1 = np.asarray(self.superstar["MHillas_1.fDelta"].array())
            hillas_psi_M2 = np.asarray(self.superstar["MHillas_2.fDelta"].array())
            hillas_skewness_M1 = np.asarray(self.superstar["MHillasExt_1.fM3Long"].array())
            hillas_skewness_M2 = np.asarray(self.superstar["MHillasExt_2.fM3Long"].array())

            leakage_intensity_1_M1 = np.asarray(self.superstar["MNewImagePar_1.fLeakage1"].array())
            leakage_intensity_1_M2 = np.asarray(self.superstar["MNewImagePar_2.fLeakage1"].array())
            leakage_intensity_2_M1 = np.asarray(self.superstar["MNewImagePar_1.fLeakage2"].array())
            leakage_intensity_2_M2 = np.asarray(self.superstar["MNewImagePar_2.fLeakage2"].array())

            num_islands_M1 = np.asarray(self.superstar["MCerPhotEvt_1.fNumIslands"].array())
            num_islands_M2 = np.asarray(self.superstar["MCerPhotEvt_2.fNumIslands"].array())

            #Reading data from root file for Image table (peak time and image mask not )
            charge_M1 = np.asarray(self.superstar["MCerPhotEvt_1.fPixels.fPhot"].array())
            peak_time_M1 = np.zeros((charge_M1.shape[0], 1039))
            image_mask_M1 = np.zeros((charge_M1.shape[0], 1039), dtype=bool)
            charge_M2 = np.asarray(self.superstar["MCerPhotEvt_2.fPixels.fPhot"].array())
            peak_time_M2 = np.zeros((charge_M2.shape[0], 1039))
            image_mask_M2 = np.zeros((charge_M2.shape[0], 1039), dtype=bool)

        # Get the shower primary id
        shower_primary_id = 1
        if self.file_list[0].split("/")[-1].startswith("GA"):
            shower_primary_id = 1

        #Iterating over all events, and saving only stereo ones
        total_events = min(len(charge_M1), len(charge_M2))
        tels_with_data = {1,2}
        for i in range(0, total_events):
            if eventid_M1[i] != 0:
                obs_id = self.run_number
                event_id = eventid_M1[i]
                i2 = np.where(eventid_M2==eventid_M1[i])
                i2 = i2[0].astype(int)
                data.count = counter

                # Setting up the Data container
                data.index.obs_id = obs_id
                data.index.event_id = event_id
                data.r0.tel.clear()
                data.r1.tel.clear()
                data.dl0.tel.clear()

                # Adding the array pointing in the pointing container
                data.pointing.array_altitude = u.Quantity(np.deg2rad(90.0 - pointing_altitude[i]), u.rad)
                data.pointing.array_azimuth = u.Quantity(np.deg2rad(pointing_azimuth[i]), u.rad)

                # Filling the DL1 container with the event data
                for tel_id in tels_with_data:
                    #Adding telescope pointing container
                    data.pointing.tel[tel_id].azimuth = u.Quantity(np.deg2rad(pointing_azimuth[i]), u.rad)
                    data.pointing.tel[tel_id].altitude = u.Quantity(np.deg2rad(90.0 - pointing_altitude[i]), u.rad)

                    #Adding MC data
                    data.mc.alt = Angle(np.pi/2.0 - zenith[i], u.rad)
                    data.mc.az = Angle(np.deg2rad(180.0-7.0) - azimuth[i], u.rad)
                    data.mc.x_max = u.Quantity(0, X_MAX_UNIT)
                    data.mc.h_first_int = u.Quantity(h_first_int[i], u.m)
                    data.mc.core_x = u.Quantity(core_x[i], u.m)
                    data.mc.core_y = u.Quantity(core_y[i], u.m)
                    data.mc.energy = u.Quantity(mc_energy[i], u.TeV)
                    data.mc.shower_primary_id = shower_primary_id

                    if self.superstar is not None:
                        leakage_values = LeakageContainer()
                        hillas_parameters_values = HillasParametersContainer()
                        concentration_values = ConcentrationContainer()
                        timing_values = TimingParametersContainer()
                        morphology_values = MorphologyContainer()

                    # Adding charge, peak time and parameters
                    if tel_id == 1:
                        data.dl1.tel[tel_id].image = charge_M1[i][:1039]
                        data.dl1.tel[tel_id].peak_time = peak_time_M1[i][:1039]
                        if self.superstar is not None:
                            data.dl1.tel[tel_id].image_mask = image_mask_M1[i][:1039]

                            hillas_parameters_values["intensity"] = hillas_intensity_M1[i]
                            hillas_parameters_values["x"] = u.Quantity(hillas_x_M1[i], unit=u.mm)
                            hillas_parameters_values["y"] = u.Quantity(hillas_y_M1[i], unit=u.mm)
                            hillas_parameters_values["r"] = u.Quantity(hillas_r_M1[i], unit=u.mm)
                            hillas_parameters_values["phi"] = u.Quantity(hillas_phi_M1[i], unit=u.rad)
                            hillas_parameters_values["length"] = u.Quantity(hillas_length_M1[i], unit=u.mm)
                            hillas_parameters_values["width"] = u.Quantity(hillas_width_M1[i], unit=u.mm)
                            hillas_parameters_values["psi"] = u.Quantity(hillas_psi_M1[i], unit=u.rad)
                            hillas_parameters_values["skewness"] = hillas_skewness_M1[i]

                            leakage_values["intensity_width_1"] = leakage_intensity_1_M1[i]
                            leakage_values["intensity_width_2"] = leakage_intensity_2_M1[i]

                            morphology_values["num_pixels"] = 1039
                            morphology_values["num_islands"] = num_islands_M1[i]

                    else:
                        data.dl1.tel[tel_id].image = charge_M2[i][:1039]
                        data.dl1.tel[tel_id].peak_time = peak_time_M2[i][:1039]
                        if self.superstar is not None:
                            data.dl1.tel[tel_id].image_mask = image_mask_M2[i][:1039]

                            hillas_parameters_values["intensity"] = hillas_intensity_M2[i]
                            hillas_parameters_values["x"] = u.Quantity(hillas_x_M2[i], unit=u.mm)
                            hillas_parameters_values["y"] = u.Quantity(hillas_y_M2[i], unit=u.mm)
                            hillas_parameters_values["r"] = u.Quantity(hillas_r_M2[i], unit=u.mm)
                            hillas_parameters_values["phi"] = u.Quantity(hillas_phi_M2[i], unit=u.rad)
                            hillas_parameters_values["length"] = u.Quantity(hillas_length_M2[i], unit=u.mm)
                            hillas_parameters_values["width"] = u.Quantity(hillas_width_M2[i], unit=u.mm)
                            hillas_parameters_values["psi"] = u.Quantity(hillas_psi_M2[i], unit=u.rad)
                            hillas_parameters_values["skewness"] = hillas_skewness_M2[i]

                            leakage_values["intensity_width_1"] = leakage_intensity_1_M2[i]
                            leakage_values["intensity_width_2"] = leakage_intensity_2_M2[i]

                            morphology_values["num_pixels"] = 1039
                            morphology_values["num_islands"] =  num_islands_M2[i]

                    if self.superstar is not None:
                        data.dl1.tel[tel_id].parameters.leakage = leakage_values
                        data.dl1.tel[tel_id].parameters.hillas = hillas_parameters_values
                        data.dl1.tel[tel_id].parameters.concentration = concentration_values
                        data.dl1.tel[tel_id].parameters.timing = timing_values
                        data.dl1.tel[tel_id].parameters.morphology = morphology_values

                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data


                yield data
                counter += 1
        return
    def _parse_mc_header(self):
        run_header = "MMcRunHeader" if self.superstar is None else "MMcRunHeader_1"
        return containers.MAGICMCHeaderContainer(
            corsika_version = self.meta["{}.fCorsikaVersion".format(run_header)].array()[0],
            refl_version = self.meta["{}.fReflVersion".format(run_header)].array()[0],
            cam_version = self.meta["{}.fCamVersion".format(run_header)].array()[0],
            run_number = self.meta["{}.fMcRunNumber".format(run_header)].array()[0],
            prod_site = self.meta["{}.fProductionSite".format(run_header)].array()[0],
            date_run_mmcs = self.meta["{}.fDateRunMMCs".format(run_header)].array()[0],
            date_run_cam = self.meta["{}.fDateRunCamera".format(run_header)].array()[0],
            shower_theta_max = Angle(self.meta["{}.fShowerThetaMax".format(run_header)].array()[0], u.deg),
            shower_theta_min = Angle(self.meta["{}.fShowerThetaMin".format(run_header)].array()[0], u.deg),
            shower_phi_max = Angle(self.meta["{}.fShowerPhiMax".format(run_header)].array()[0], u.deg),
            shower_phi_min = Angle(self.meta["{}.fShowerPhiMin".format(run_header)].array()[0], u.deg),
            c_wave_lower = self.meta["{}.fCWaveLower".format(run_header)].array()[0],
            c_wave_upper = self.meta["{}.fCWaveUpper".format(run_header)].array()[0],
            num_obs_lev = self.meta["{}.fNumObsLev".format(run_header)].array()[0],
            height_lev = self.meta["{}.fHeightLev[10]".format(run_header)].array(),
            slope_spec = self.meta["{}.fSlopeSpec".format(run_header)].array()[0],
            rand_pointing_cone_semi_angle = Angle(self.meta["{}.fRandomPointingConeSemiAngle".format(run_header)].array()[0], u.deg),
            impact_max = self.meta["{}.fImpactMax".format(run_header)].array()[0],
            star_field_rotate = self.meta["{}.fStarFieldRotate".format(run_header)].array()[0],
            star_field_ra_h = self.meta["{}.fStarFieldRaH".format(run_header)].array()[0],
            star_field_ra_m = self.meta["{}.fStarFieldRaM".format(run_header)].array()[0],
            star_field_ra_s = self.meta["{}.fStarFieldRaS".format(run_header)].array()[0],
            star_field_dec_d = self.meta["{}.fStarFieldDeD".format(run_header)].array()[0],
            star_field_dec_m = self.meta["{}.fStarFieldDeM".format(run_header)].array()[0],
            star_field_dec_s = self.meta["{}.fStarFieldDeS".format(run_header)].array()[0],
            num_trig_cond = self.meta["{}.fNumTrigCond".format(run_header)].array()[0],
            all_evts_trig = self.meta["{}.fAllEvtsTriggered".format(run_header)].array()[0],
            mc_evt = self.meta["{}.fMcEvt".format(run_header)].array()[0],
            mc_trig = self.meta["{}.fMcTrig".format(run_header)].array()[0],
            mc_fadc = self.meta["{}.fMcFadc".format(run_header)].array()[0],
            raw_evt = self.meta["{}.fRawEvt".format(run_header)].array()[0],
            num_analised_pix = self.meta["{}.fNumAnalisedPixels".format(run_header)].array()[0],
            num_simulated_showers = self.meta["{}.fNumSimulatedShowers".format(run_header)].array()[0],
            num_stored_showers = self.meta["{}.fNumStoredShowers".format(run_header)].array()[0],
            num_events = self.meta["{}.fNumEvents".format(run_header)].array()[0],
            num_phe_from_dnsb = self.meta["{}.fNumPheFromDNSB".format(run_header)].array()[0],
            elec_noise = self.meta["{}.fElecNoise".format(run_header)].array()[0],
            optic_links_noise = self.meta["{}.fOpticLinksNoise".format(run_header)].array()[0] )
