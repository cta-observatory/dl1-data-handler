import copy

import numpy as np

class DL1DataProcessor():

    def __init__(self, mode, input_description, transforms=None):
        if mode in ['mono', 'stereo', 'multi-stereo']:
            self.mode = mode
        else:
            raise ValueError("Invalid mode selection '{}'. Valid options: "
                             "'mono', 'stereo', 'multi-stereo'".format(mode))
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.input_description = copy.deepcopy(input_description)
        for transform in self.transforms:
            input_description = transform.describe(input_description)
        self.output_description = input_description

    def process(self, example):
        for transform in self.transforms:
            example = transform(example)
        return example

class Transform():

    def __init__(self):
        self.description = []

    def describe(self, description):
        self.description = description
        return self.description

    def __call__(self, example):
        return example

class ConvertShowerPrimaryIDToClassLabel(Transform):

    def __init__(self):
        super().__init__()
        self.shower_primary_id_to_class = {
            0: 1, # gamma
            101: 0 # proton
            }

    def describe(self, description):
        self.description = [
            {**des, 'name': 'class_label'} if des['name'] == 'shower_primary_id'
            else des for des in description]
        return self.description

    def __call__(self, example):
        for i, (arr, des) in enumerate(zip(example, self.description)):
            if des['name'] == 'shower_primary_id':
                class_label = np.array(
                    self.shower_primary_id_to_class[arr],
                    dtype=des['dtype'])
                example[i] = class_label
        return example

class NormalizeTelescopePositions(Transform):

    def __init__(self, norm_x=1.0, norm_y=1.0, norm_z=1.0):
        super().__init__()
        self.norms = {'x': norm_x, 'y': norm_y, 'z': norm_z}

    def transform(self, example):
        for i, (arr, des) in enumerate(zip(example, self.description)):
            if des['base_name'] in self.norms:
                normed_pos = arr / self.norms[des['base_name']]
                example[i] = np.array(normed_pos, dtype=des['dtype'])
        return example
