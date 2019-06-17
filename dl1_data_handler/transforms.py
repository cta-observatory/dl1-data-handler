import numpy as np
from .processor import Transform


class ConvertShowerPrimaryIDToClassLabel(Transform):

    def __init__(self):
        super().__init__()
        self.shower_primary_id_to_class = {
            0: 1,  # gamma
            101: 0  # proton
        }
        self.name = 'class_label'
        self.dtype = np.dtype('int8')

    def describe(self, description):
        self.description = [
            {**des, 'name': self.name, 'dtype': self.dtype}
            if des['name'] == 'shower_primary_id'
            else des for des in description]
        return self.description

    def __call__(self, example):
        for i, (arr, des) in enumerate(zip(example, self.description)):
            if des['name'] == self.name:
                class_label = np.array(
                    self.shower_primary_id_to_class[arr.tolist()],
                    dtype=self.dtype)
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


class EnergyToLog(Transform):

    def describe(self, description):
        self.description = description
        for des in self.description:
            if des['base_name'] == 'mc_energy':
                des['unit'] = 'log(TeV)'
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des['base_name'] == 'mc_energy':
                example[i] = np.log10(val)
        return example


class ImpactToKm(Transform):

    def describe(self, description):
        self.description = description
        for des in self.description:
            if des['base_name'] in ['core_x', 'core_y']:
                des['unit'] = 'km'
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des['base_name'] in ['core_x', 'core_y']:
                example[i] = val / 1000
        return example


class XmaxToKm(Transform):

    def describe(self, description):
        self.description = description
        for des in self.description:
            if des['base_name'] == 'x_max':
                des['unit'] = 'km'
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des['base_name'] == 'x_max':
                example[i] = val / 1000
        return example


class HfirstIntToKm(Transform):

    def describe(self, description):
        self.description = description
        for des in self.description:
            if des['base_name'] == 'h_first_int':
                des['unit'] = 'km'
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des['base_name'] == 'h_first_int':
                example[i] = val / 1000
        return example


class TelescopePositionToKm(Transform):

    def describe(self, description):
        self.description = description
        for des in self.description:
            if des['base_name'] in ['x', 'y', 'z']:
                des['unit'] = 'km'
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des['base_name'] in ['x', 'y', 'z']:
                example[i] = val / 1000
        return example


class DataForGammaLearn(Transform):

    def __init__(self):
        super().__init__()
        self.mc_infos = ['mc_energy', 'core_x', 'core_y', 'alt', 'az', 'shower_primary_id', 'x_max', 'h_first_int']
        self.array_infos = ['x', 'y', 'z']

    def describe(self, description):
        self.description = description
        label_description = []
        for des in self.description:
            if des['base_name'] in self.mc_infos:
                label_description.append(des['base_name'])
        if len(label_description) > 0:
            self.description.append({'name': 'label',
                                     'tel_type': None,
                                     'base_name': 'label',
                                     'shape': None,
                                     'dtype': None,
                                     'label_description': label_description})
        return self.description

    def __call__(self, example):
        image = None
        mc_energy = None
        labels = []
        array = []
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des['base_name'] == 'image':
                image = val
            elif des['base_name'] in self.mc_infos:
                labels.append(val)
                if des['base_name'] == 'mc_energy':
                    mc_energy = val
            elif des['base_name'] in self.array_infos:
                array.append(val)
        return {'image': image, 'label': np.stack(labels), 'mc_energy': mc_energy, 'telescope': np.stack(array)}

