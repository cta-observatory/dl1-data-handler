import copy

import numpy as np

class DL1DataProcessor():

    def __init__(mode, input_description, transforms=None):
        if mode in ['mono', 'stereo', 'multi-stereo']:
            self.mode = mode
        else:
            raise ValueError("Invalid mode selection '{}'. Valid options: "
                    "'mono', 'stereo', 'multi-stereo'".format(mode))
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.input_description = input_description
        for transform in self.transforms:
            output_description = transform.define(input_description)
            input_description = copy.deepcopy(output_description)
        self.output_description = input_description

    def process(example):
        for transform in self.transforms:
            example = transform.transform(example, self.reader_example_dfn)
        return example

class Transform():

    def __init__(self):
        pass

    def define(self, input_description):
        self.input_description = input_description
        output_description = input_description
        return output_description
    
    def transform(self, example):
        return example

class ConvertParticleIDToClassLabel(Transform):

    def __init__(self):
        self.particle_id_to_class = {
                0: 1, # gamma
                101: 0 # proton
                }

    def define(self, input_description):
        self.input_description = input_description
        output_description = [{**des, 'name': 'class_label'} for des
                in input_description if des['name'] == 'particle_id']
        return output_description
    
    def transform(self, example):
        for i, (arr, des) in enumerate(zip(example, self.input_description)):
            if des['name'] == 'particle_id':
                class_label = np.array(self.particle_id_to_class[arr],
                        dtype=des['dtype'])
                example[i] = class_label
        return example

class NormalizeTelescopePositions(Transform):

    def __init__(self, norm_x=1.0, norm_y=1.0, norm_z=1.0):
        self.norms = {'tel_x': norm_x, 'tel_y': norm_y, 'tel_z': norm_z}

    def transform(self, example):
        for i, (arr, des) in enumerate(zip(example, self.input_description)):
            if des['base_name'] in self.norms:
                normed_pos = arr / self.norms[des['base_name']]
                example[i] = np.array(normed_pos, dtype=des['dtype'])
        return example
