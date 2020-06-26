import uproot
import numpy as np
import tables
from dl1_data_handler import table_definitions as table_defs
from ctapipe.instrument.camera import CameraGeometry
import re

def root2hdf5(filename1,filename2, output):

	#open output h5 file
	h5 = tables.open_file(output, mode='w')
	#File format settings
	filter_settings = {
		'complib': 'lzo',
		'complevel': 1
	}
	filters = tables.Filters(**filter_settings)
	tel_table_desc = table_defs.TelTableRow
	tel_table_desc.columns['pixel_positions'] = tables.Float32Col(shape=(1039, 2))
	columns_dict = {
		            "event_index": tables.Int32Col(),
		            "charge": tables.Float32Col(shape=(1039)),
		            "pulse_time": tables.Float32Col(shape=(1039))
		        }

	description = type('description', (tables.IsDescription,), columns_dict)
	event_table_desc = table_defs.EventTableRow
	event_table_desc.columns["LST_MAGIC_MAGICCam_indices"] = (tables.UInt32Col(shape=(2)))
	event_table_desc.columns["LST_MAGIC_MAGICCam_multiplicity"] = (tables.UInt32Col())

	#Create tables
	tel_table = h5.create_table(h5.root,'Telescope_Type_Information',tel_table_desc, "Table of telescope type information",filters=filters)
	array_table = h5.create_table(h5.root,'Array_Information',table_defs.ArrayTableRow,"Table of array/subarray information", filters=filters)
	image_table = h5.create_table(h5.root, 'LST_MAGIC_MAGICCam', description,"Table of MAGIC Images", filters = filters, expectedrows=100)
	event_table = h5.create_table(h5.root,'Events', event_table_desc, "Table of Event Information", filters=filters)

	#Array_Information table

	row = array_table.row
	row["type"] = "LST_MAGIC_MAGICCam"
	row["id"] = 1
	row["x"] = -27.24
	row["y"] = -146.66
	row["z"] = 50.00
	row.append()
	row["type"] = "LST_MAGIC_MAGICCam"
	row["id"] = 2
	row["x"] = -96.44
	row["y"] = -96.77
	row["z"] = 51.00
	row.append()

	#Telescope_Type_Information table

	camtype="MAGICCam"
	camgeo = CameraGeometry.from_name(camtype)
	pixel_positions = np.column_stack([camgeo.pix_x.value, camgeo.pix_y.value])
	row = tel_table.row
	row["type"] = "LST_MAGIC_MAGICCam"
	row["optics"] = "MAGIC"
	row["camera"] = camtype
	row["num_pixels"] = 1039
	row["pixel_positions"] = pixel_positions
	row.append()
	#reading the 2 MAGIC files
	file = uproot.open(filename1)
	eventM1 = file["Events"]
	file2 = uproot.open(filename2)
	eventM2 = file2["Events"]
	mask = r".*_M\d_za\d+to\d+_\d_(\d+)_Y_.*"
	parsed_info = re.findall(mask, filename1)
	obs_id = parsed_info

	#Event table

	row = event_table.row
	eventidM1 = np.asarray(eventM1["MRawEvtHeader.fStereoEvtNumber"].array())
	eventidM2 = np.asarray(eventM2["MRawEvtHeader.fStereoEvtNumber"].array())
	zenith = np.asarray(eventM1["MPointingPos.fZd"].array())
	zenith = np.deg2rad(zenith)
	azimuth = np.asarray(eventM1["MPointingPos.fAz"].array())
	azimuth = np.deg2rad(azimuth)
	core_x = np.asarray(eventM1["MMcEvt.fCoreX"].array())
	core_y = np.asarray(eventM1["MMcEvt.fCoreY"].array())
	mc_energy = np.asarray(eventM1["MMcEvt.fEnergy"].array())/1000
	h_first_int = np.asarray(eventM1["MMcEvt.fZFirstInteraction"].array())
	k = 1
	stereo_total = np.max(eventidM1)
	event_index = np.zeros(shape = (stereo_total,1))
	#Image table

	chargeM1 = eventM1["MCerPhotEvt.fPixels.fPhot"].array()
	pulse_timeM1 = eventM1["MArrivalTime.fData"].array()
	chargeM1 = np.asarray(chargeM1)
	pulse_timeM1 = np.asarray(pulse_timeM1)
	chargeM2 = eventM2["MCerPhotEvt.fPixels.fPhot"].array()
	pulse_timeM2 = eventM2["MArrivalTime.fData"].array()
	chargeM2 = np.asarray(chargeM2)
	pulse_timeM2 = np.asarray(pulse_timeM2)
	image = image_table.row
	image["charge"] = np.zeros(shape = (1039), dtype=np.float32)
	image["pulse_time"] = np.zeros(shape = (1039), dtype=np.float32)
	image["event_index"] = -1
	image.append()
	k=1
	total_events = len(chargeM1)
	for i in range(0,total_events):
		if eventid1[i] != 0:
			i2 = np.where(eventidM2==eventidM1[i])
			i2 = int(i2[0])
			row["LST_MAGIC_MAGICCam_multiplicity"] = 2
			row["LST_MAGIC_MAGICCam_indices"] = [2*(k-1)+1,2*(k-1)+2]
			row["event_id"] = eventidM1[i]
			row["alt"] = zenith[i]
			row["az"] = azimuth[i]
			row["core_x"] = core_x[i]
			row["core_y"] = core_y[i]
			row["event_id"] = eventidM1[i]
			row["h_first_int"] = h_first_int[i]
			row["mc_energy"] = mc_energy[i]
			row["obs_id"] = obs_id
			row["shower_primary_id"] = 0
			row["x_max"] = 0
			row.append()
			chargeM1[i] = chargeM1[i][:1039]
			pulse_timeM1[i] = pulse_timeM1[i][:1039]
			chargeM2[i2] = chargeM2[i2][:1039]
			pulse_timeM2[i2] = pulse_timeM2[i2][:1039]
			image["charge"] = chargeM1[i]
			image["pulse_time"] = pulse_timeM1[i]
			image["event_index"] = eventidM1[i]
			image.append()
			image["charge"] = chargeM2[i2]
			image["pulse_time"] = pulse_timeM2[i2]
			image["event_index"] = eventidM1[i]
			image.append()
			k+=1
