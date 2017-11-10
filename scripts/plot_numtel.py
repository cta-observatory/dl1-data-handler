import image_extractor

# telescope constants
SCT_IMAGE_WIDTH = 120
SCT_IMAGE_LENGTH = 120
LST_NUM_PIXELS = 1855
SCT_NUM_PIXELS = 11328
SST_NUM_PIXELS = 0  # update

input_simtel_file = ""

# collect telescope lists
source_temp = hessio_event_source(input_simtel_file, max_events=1)

LST_list = []
SCT_list = []
SST_list = []

for event in source_temp:
    for i in event.inst.telescope_ids:
        if event.inst.num_pixels[i] == SCT_NUM_PIXELS:
            SCT_list.append(i)
        elif event.inst.num_pixels[i] == LST_NUM_PIXELS:
            LST_list.append(i)
        else:
            SST_list.append(i)
        # elif event.inst.num_pixels[i] ==

    all_tels = {'SST': SST_list, 'SCT': SCT_list, 'LST': LST_list}

# select telescopes by type
selected = {}
TEL_MODE = 'SCT'
if TEL_MODE == 'SST':
    selected['SST'] = SST_list
elif TEL_MODE == 'SCT':
    selected['SCT'] = SCT_list
elif TEL_MODE == 'LST':
    selected['LST'] = LST_list
elif TEL_MODE == 'SCT+LST':
    selected['LST'] = LST_list
    selected['SCT'] = SCT_list
elif TEL_MODE == 'SST+SCT':
    selected['SST'] = SST_list
    selected['SCT'] = SCT_list
elif TEL_MODE == 'SST+LST':
    selected['LST'] = LST_list
    selected['SST'] = SST_list
elif TEL_MODE == 'ALL':
    selected['LST'] = LST_list
    selected['SCT'] = SCT_list
    selected['SST'] = SST_list
else:
    raise ValueError('Telescope selection mode not recognized.')

print("Telescope Mode: ", TEL_MODE)

NUM_TEL = 0

for i in selected.keys():
    print(i + ": " + str(len(selected[i])) + " out of " + str(
        len(all_tels[i])) + " telescopes selected.")
    NUM_TEL += len(selected[i])

source = hessio_event_source(input_simtel_file)

trig_nums_list = []

passing_trig_nums_list = []

# bins cuts dict

bins_dict = pkl.load(
    open("./aux/bins_cuts_dict/bins_cuts_dict_1bin.pkl", "rb"))

for event in source:
    num_trig = 0

    for tel_type in selected.keys():
        for tel_id in selected[tel_type]:
            if tel_id in event.r0.tels_with_data:
                num_trig += 1

    trig_nums_list.append(num_trig)
    if (event.r0.run_id, event.r0.event_id) in bins_dict:
        passing_trig_nums_list.append(num_trig)


# plot
plt.hist(trig_nums_list, bins=range(NUM_TEL + 1))
plt.xlabel('Number of triggered telescopes')
plt.ylabel('Number of events')
plt.axvline(np.mean(trig_nums_list), color='b', linestyle='dashed')
plt.savefig('all_events_num_trig.png')
plt.gcf().clear()

plt.hist(passing_trig_nums_list, bins=range(NUM_TEL + 1))
plt.xlabel('Number of triggered telescopes')
plt.ylabel('Number of events')
plt.axvline(np.mean(passing_trig_nums_list), color='b', linestyle='dashed')
plt.savefig('passing_num_trig.png')
plt.gcf().clear()

plt.hist(trig_nums_list, bins=range(NUM_TEL + 1))
plt.hist(passing_trig_nums_list, bins=range(NUM_TEL + 1))
plt.xlabel('Number of triggered telescopes')
plt.ylabel('Number of events')
plt.axvline(np.mean(trig_nums_list), color='b', linestyle='dashed')
plt.axvline(np.mean(passing_trig_nums_list), color='r', linestyle='dashed')
plt.savefig('combined_num_trig.png')

print(
    "Mean number of triggered telescopes (all) : {}".format(
        np.mean(trig_nums_list)))
print("Mean number of triggered telescopes (passing) : {}".format(
        np.mean(passing_trig_nums_list)))
