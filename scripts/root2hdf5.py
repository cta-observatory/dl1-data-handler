import uproot
import numpy as np
import tables
from dl1_data_handler import table_definitions as table_defs
from ctapipe.instrument.camera import CameraGeometry
import re

def root2hdf5(filename1,filename2, output):
	#reading the 2 MAGIC files
	file = uproot.open(filename1)
	event = file["Events"]
	file2 = uproot.open(filename2)
	event2 = file2["Events"]
	file_name = 'GA_M1_za05to35_8_821319_Y_w0.root'
	mask = r".*_M\d_za\d+to\d+_\d_(\d+)_Y_.*"
	parsed_info = re.findall(mask, file_name)
	obs_id = parsed_info
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
	event_table_desc.columns["LST_MAGIC_MAGICCam" + '_indices'] = (tables.UInt32Col(shape=(2)))
	event_table_desc.columns["LST_MAGIC_MAGICCam" + '_multiplicity'] = (tables.UInt32Col())

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
	pixel_positions = np.column_stack([camgeo.pix_x.value, camgeo.pix_y.value]).T
	row = tel_table.row
	row["type"] = "LST_MAGIC_MAGICCam"
	row["optics"] = "MAGIC"
	row["camera"] = camtype
	row["num_pixels"] = 1039
	row["pixel_positions"] = pixel_positions.T
	row.append()

	#Event table

	row = event_table.row
	eventid1 = np.asarray(event["MRawEvtHeader.fStereoEvtNumber"].array())
	eventid2 = np.asarray(event2["MRawEvtHeader.fStereoEvtNumber"].array())
	zenith = np.asarray(event["MPointingPos.fZd"].array())
	zenith = np.deg2rad(zenith)
	azimuth = np.asarray(event["MPointingPos.fAz"].array())
	azimuth = np.deg2rad(azimuth)
	core_x = np.asarray(event["MMcEvt.fCoreX"].array())
	core_y = np.asarray(event["MMcEvt.fCoreY"].array())
	mc_energy = np.asarray(event["MMcEvt.fEnergy"].array())
	h_first_int = np.asarray(event["MMcEvt.fZFirstInteraction"].array())
	k = 1
	event_index = np.zeros(shape = (157,1))

	#Image table

	charge1 = event["MCerPhotEvt.fPixels.fPhot"].array()
	pulse_time1 = event["MArrivalTime.fData"].array()
	charge1 = np.asarray(charge1)
	pulse_time1 = np.asarray(pulse_time1)
	charge2 = event2["MCerPhotEvt.fPixels.fPhot"].array()
	pulse_time2 = event2["MArrivalTime.fData"].array()
	charge2 = np.asarray(charge2)
	pulse_time2 = np.asarray(pulse_time2)
	image = image_table.row
	image["charge"] = np.zeros(shape = (1039), dtype=np.float32)
	image["pulse_time"] = np.zeros(shape = (1039), dtype=np.float32)
	image["event_index"] = -1
	image.append()
	k=1
	for i in range(0,230):
		if eventid1[i] != 0:
			i2 = np.where(eventid2==eventid1[i])
			i2 = int(i2[0])
			row["LST_MAGIC_MAGICCam_multiplicity"] = 2
			row["LST_MAGIC_MAGICCam_indices"] = [2*(k-1)+1,2*(k-1)+2]
			row["event_id"] = eventid1[i]
			row["alt"] = zenith[i]
			row["az"] = azimuth[i]
			row["core_x"] = core_x[i]
			row["core_y"] = core_y[i]
			row["event_id"] = eventid1[i]
			row["h_first_int"] = h_first_int[i]
			row["mc_energy"] = mc_energy[i]/1000
			row["obs_id"] = obs_id
			row["shower_primary_id"] = 0
			row["x_max"] = 0
			row.append()
			charge1[i] = charge1[i][0:1039]
			#charge1[i] = np.reshape(charge1[i], (1039,1))
			pulse_time1[i] = pulse_time1[i][0:1039]
			#pulse_time1[i] = np.reshape(pulse_time1[i], (1039,1))
			charge2[i2] = charge2[i2][0:1039]
			#charge2[i2] = np.reshape(charge2[i2], (1039,1))
			pulse_time2[i2] = pulse_time2[i2][0:1039]
			#pulse_time2[i2] = np.reshape(pulse_time2[i2], (1039,1))
			image["charge"] = charge1[i]
			image["pulse_time"] = pulse_time1[i]
			image["event_index"] = eventid1[i]
			image.append()
			image["charge"] = charge2[i2]
			image["pulse_time"] = pulse_time2[i2]
			image["event_index"] = eventid1[i]
			image.append()
			k+=1
