import numpy as np
import itertools
from .processor import Transform


class ShowerPrimaryID(Transform):

    def __init__(self, name='particletype'):
        super().__init__()
        self.shower_primary_id_to_class = {
            0: 1,    # gamma
            101: 0,  # proton
            1: 2     # electron
        }
        self.name = name
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
                particletype = np.array(
                    self.shower_primary_id_to_class[arr.tolist()],
                    dtype=self.dtype)
                example[i] = particletype
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


class MCEnergy(Transform):

    def __init__(self, name='energy', unit='log(TeV)'):
        super().__init__()
        self.name = name
        self.shape = (1)
        self.dtype = np.dtype('float32')
        self.unit = unit

    def describe(self, description):
        self.description = [
            {**des, 'name': self.name, 'dtype': self.dtype, 'unit': self.unit}
            if des['name'] == 'mc_energy'
            else des for des in description]
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des['base_name'] == 'mc_energy':
                example[i] = np.array([np.log10(val)]) if self.unit == 'log(TeV)' else np.array([val])
        return example


class DeltaAltAz(Transform):

    def __init__(self):
        super().__init__()
        self.name = 'deltaAltAz'
        self.base_name = 'direction'
        self.shape = (2)
        self.dtype = np.dtype('float32')
        self.unit = 'rad'
        self.tel_pointing = np.array([0.0, 0.0], dtype=np.float32)

    def describe(self, description):
        self.description = description
        self.description.append(
            {
                'name': self.base_name,
                'tel_type': None,
                'base_name': self.base_name,
                'shape': self.shape,
                'dtype': self.dtype,
                'unit': self.unit
                }
            )
        return self.description

    def set_tel_pointing(self, tel_pointing):
        self.tel_pointing = tel_pointing
        return

    def __call__(self, example):
        for i, (val, des) in enumerate(itertools.zip_longest(example, self.description)):
            if des['base_name'] == 'alt':
                alt = example[i] - self.tel_pointing[1]
            elif des['base_name'] == 'az':
                az = example[i] - self.tel_pointing[0]
            elif des['base_name'] == self.base_name:
                example.append(np.array([alt,az]))
        return example


class AltAz(Transform):

    def __init__(self, name='direction'):
        super().__init__()
        self.name = name
        self.shape = (2)
        self.dtype = np.dtype('float32')
        self.unit = 'rad'

    def describe(self, description):
        self.description = description
        self.description.append(
            {
                'name': self.name,
                'tel_type': None,
                'base_name': self.name,
                'shape': self.shape,
                'dtype': self.dtype,
                'unit': self.unit
                }
            )
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(itertools.zip_longest(example, self.description)):
            if des['base_name'] == 'alt':
                alt = example[i]
            elif des['base_name'] == 'az':
                az = example[i]
            elif des['base_name'] == self.name:
                example.append(np.array([alt,az]))
        return example


class CoreXY(Transform):

    def __init__(self, name='impact'):
        super().__init__()
        self.name = name
        self.shape = (2)
        self.dtype = np.dtype('float32')
        self.unit = 'km'

    def describe(self, description):
        self.description = description
        self.description.append(
            {
                'name': self.name,
                'tel_type': None,
                'base_name': self.name,
                'shape': self.shape,
                'dtype': self.dtype,
                'unit': self.unit
                }
            )
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(itertools.zip_longest(example, self.description)):
            if des['base_name'] == 'core_x':
                example[i] = val / 1000
                core_x_km = example[i]
            elif des['base_name'] == 'core_y':
                example[i] = val / 1000
                core_y_km = example[i]
            elif des['base_name'] == self.name:
                example.append(np.array([core_x_km,core_y_km]))
        return example


class Xmax(Transform):

    def __init__(self, name='showermaximum', unit='km'):
        super().__init__()
        self.name = name
        self.shape = (1)
        self.dtype = np.dtype('float32')
        self.unit = unit

    def describe(self, description):
        self.description = [
            {**des, 'name': self.name, 'dtype': self.dtype, 'unit': self.unit}
            if des['name'] == 'showermaximum'
            else des for des in description]
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des['base_name'] == 'showermaximum':
                example[i] = np.array([val/1000]) if self.unit == 'km' else np.array([val])
        return example


class HfirstInt(Transform):

    def __init__(self, name='h_first_int', unit='km'):
        super().__init__()
        self.name = name
        self.shape = (1)
        self.dtype = np.dtype('float32')
        self.unit = unit

    def describe(self, description):
        self.description = [
            {**des, 'name': self.name, 'dtype': self.dtype, 'unit': self.unit}
            if des['name'] == 'h_first_int'
            else des for des in description]
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des['base_name'] == 'h_first_int':
                example[i] = np.array([val/1000]) if self.unit == 'km' else np.array([val])
        return example


class TelescopePositionInKm(Transform):

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
                image = val.T
            elif des['base_name'] in self.mc_infos:
                if des['name'] == 'class_label':
                    val = val.astype(np.float32)
                labels.append(val)
                if des['base_name'] == 'mc_energy':
                    mc_energy = val
            elif des['base_name'] in self.array_infos:
                array.append(val)
        sample = {'image': image}
        if len(labels) > 0:
            sample['label'] = np.stack(labels)
        if mc_energy is not None:
            sample['mc_energy'] = mc_energy
        if len(array) > 0:
            sample['telescope'] = np.stack(array)
        return sample


class SortTelescopes(Transform):

    def __init__(self, sorting='trigger'):
        super().__init__()
        params = {
            # List triggered telescopes first
            'trigger': {'reverse': True, 'key': lambda x: x['trigger']},
            # List from largest to smallest sum of pixel charges
            'size': {'reverse': True,
                     'key': lambda x: np.sum(x['image'][..., 0], (1, 2))}
            }
        if sorting in params:
            self.step = -1 if params[sorting]['reverse'] else 1
            self.key = params[sorting]['key']
        else:
            raise ValueError("Invalid image sorting method: {}. Select "
                             "'trigger' or 'size'.".format(sorting))

    def __call__(self, example):
        outputs = {des['name']: arr for arr, des
                   in zip(example, self.description)}
        indices = np.argsort(self.key(outputs))
        for i, (arr, des) in enumerate(zip(example, self.description)):
            if des['base_name'] in ['image', 'trigger', 'x', 'y', 'z']:
                example[i] = arr[indices[::self.step]]
        return example
