import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import itertools
from .processor import Transform


class ShowerPrimaryID(Transform):
    def __init__(self):
        super().__init__()
        self.particle_id_col_name = "true_shower_primary_id"
        self.shower_primary_id_to_class = {
            0: 1,
            101: 0,
            1: 2,
            255: 3,
        }
        self.shower_primary_id_to_name = {
            0: "gamma",
            101: "proton",
            1: "electron",
            255: "hadron",
        }

        self.name = "true_shower_primary_id"
        self.dtype = np.dtype("int8")

    def describe(self, description):
        self.description = [
            {**des, "name": self.name, "dtype": self.dtype}
            if des["name"] == self.particle_id_col_name
            else des
            for des in description
        ]
        return self.description

    def __call__(self, example):
        for i, (arr, des) in enumerate(zip(example, self.description)):
            if des["name"] == self.name:
                particletype = np.array(
                    self.shower_primary_id_to_class[arr.tolist()], dtype=self.dtype
                )
                example[i] = particletype
        return example


class NormalizeTelescopePositions(Transform):
    def __init__(self, norm_x=1.0, norm_y=1.0, norm_z=1.0):
        super().__init__()
        self.norms = {"x": norm_x, "y": norm_y, "z": norm_z}

    def transform(self, example):
        for i, (arr, des) in enumerate(zip(example, self.description)):
            if des["base_name"] in self.norms:
                normed_pos = arr / self.norms[des["base_name"]]
                example[i] = np.array(normed_pos, dtype=des["dtype"])
        return example


class LogEnergy(Transform):
    def __init__(self):
        super().__init__()
        self.name = "energy"
        self.shape = 1
        self.dtype = np.dtype("float32")
        self.unit = "log(TeV)"

    def describe(self, description):
        self.description = [
            {**des, "name": self.name, "dtype": self.dtype, "unit": self.unit}
            if des["name"] == "true_energy"
            else des
            for des in description
        ]
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des["base_name"] == "true_energy":
                example[i] = np.log10(val)
        return example


class SkyOffsetSeparation(Transform):
    def __init__(self, transform_to_rad=False):
        super().__init__()

        self.name = "SkyOffsetSeparation"
        self.base_name = "direction"
        self.shape = 3
        self.dtype = np.dtype("float32")
        self.unit = u.rad if transform_to_rad else u.deg
        self.fix_pointing = None

    def describe(self, description):
        self.description = description
        self.description.append(
            {
                "name": self.base_name,
                "tel_type": None,
                "base_name": self.base_name,
                "shape": self.shape,
                "dtype": self.dtype,
                "unit": str(self.unit),
            }
        )
        return self.description

    def set_pointing(self, fix_pointing):
        self.fix_pointing = fix_pointing

    def __call__(self, example):
        for i, (val, des) in enumerate(
            itertools.zip_longest(example, self.description)
        ):
            if des["base_name"] == "true_alt":
                alt = example[i]
            elif des["base_name"] == "true_az":
                az = example[i]
            elif des["base_name"] == self.base_name:
                true_direction = SkyCoord(
                    az * u.deg,
                    alt * u.deg,
                    frame="altaz",
                    unit="deg",
                )
                sky_offset = self.fix_pointing.spherical_offsets_to(true_direction)
                angular_separation = self.fix_pointing.separation(true_direction)
                example.append(
                    np.array(
                        [
                            sky_offset[0].to_value(self.unit),
                            sky_offset[1].to_value(self.unit),
                            angular_separation.to_value(self.unit),
                        ]
                    )
                )
        return example


class CoreXY(Transform):
    def __init__(self):
        super().__init__()
        self.name = "impact"
        self.shape = 2
        self.dtype = np.dtype("float32")
        self.unit = "km"

    def describe(self, description):
        self.description = description
        self.description.append(
            {
                "name": self.name,
                "tel_type": None,
                "base_name": self.name,
                "shape": self.shape,
                "dtype": self.dtype,
                "unit": self.unit,
            }
        )
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(
            itertools.zip_longest(example, self.description)
        ):
            if des["base_name"] == "true_core_x":
                example[i] = val / 1000
                core_x_km = example[i]
            elif des["base_name"] == "true_core_y":
                example[i] = val / 1000
                core_y_km = example[i]
            elif des["base_name"] == self.name:
                example.append(np.array([core_x_km, core_y_km]))
        return example


class Xmax(Transform):
    def __init__(self, name="showermaximum", unit="km"):
        super().__init__()
        self.name = name
        self.shape = 1
        self.dtype = np.dtype("float32")
        self.unit = unit

    def describe(self, description):
        self.description = [
            {**des, "name": self.name, "dtype": self.dtype, "unit": self.unit}
            if des["name"] == "showermaximum"
            else des
            for des in description
        ]
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des["base_name"] == "showermaximum":
                example[i] = (
                    np.array([val / 1000]) if self.unit == "km" else np.array([val])
                )
        return example


class HfirstInt(Transform):
    def __init__(self, name="true_h_first_int", unit="km"):
        super().__init__()
        self.name = name
        self.shape = 1
        self.dtype = np.dtype("float32")
        self.unit = unit

    def describe(self, description):
        self.description = [
            {**des, "name": self.name, "dtype": self.dtype, "unit": self.unit}
            if des["name"] == "true_h_first_int"
            else des
            for des in description
        ]
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des["base_name"] == "true_h_first_int":
                example[i] = (
                    np.array([val / 1000]) if self.unit == "km" else np.array([val])
                )
        return example


class TelescopePositionInKm(Transform):
    def describe(self, description):
        self.description = description
        for des in self.description:
            if des["base_name"] in ["x", "y", "z"]:
                des["unit"] = "km"
        return self.description

    def __call__(self, example):
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des["base_name"] in ["x", "y", "z"]:
                example[i] = val / 1000
        return example


class DataForGammaLearn(Transform):
    def __init__(self):
        super().__init__()
        self.mc_infos = [
            "true_energy",
            "true_core_x",
            "true_core_y",
            "true_alt",
            "true_az",
            "true_shower_primary_id",
            "true_x_max",
            "true_h_first_int",
        ]
        self.array_infos = ["x", "y", "z"]

    def describe(self, description):
        self.description = description
        label_description = []
        for des in self.description:
            if des["base_name"] in self.mc_infos:
                label_description.append(des["base_name"])
        if len(label_description) > 0:
            self.description.append(
                {
                    "name": "label",
                    "tel_type": None,
                    "base_name": "label",
                    "shape": None,
                    "dtype": None,
                    "label_description": label_description,
                }
            )
        return self.description

    def __call__(self, example):
        image = None
        mc_energy = None
        labels = []
        array = []
        for i, (val, des) in enumerate(zip(example, self.description)):
            if des["base_name"] == "image":
                image = val.T
            elif des["base_name"] in self.mc_infos:
                if des["name"] == "class_label":
                    val = val.astype(np.float32)
                labels.append(val)
                if des["base_name"] == "true_energy":
                    mc_energy = val
            elif des["base_name"] in self.array_infos:
                array.append(val)
        sample = {"image": image}
        if len(labels) > 0:
            sample["label"] = np.stack(labels)
        if mc_energy is not None:
            sample["true_energy"] = mc_energy
        if len(array) > 0:
            sample["telescope"] = np.stack(array)
        return sample


class SortTelescopes(Transform):
    def __init__(self, sorting="trigger", tel_desc="LST_LST_LSTCam"):
        super().__init__()
        self.name = "sortTelescopes"
        self.tel_desc = tel_desc
        params = {
            # List triggered telescopes first
            "trigger": {
                "reverse": True,
                "key": lambda x: x[self.tel_desc + "_triggers"],
            },
            # List from largest to smallest sum of pixel charges
            "size": {
                "reverse": True,
                "key": lambda x: np.sum(x[self.tel_desc + "_images"][..., 0], (1, 2)),
            },
        }
        if sorting in params:
            self.step = -1 if params[sorting]["reverse"] else 1
            self.key = params[sorting]["key"]
        else:
            raise ValueError(
                "Invalid image sorting method: {}. Select "
                "'trigger' or 'size'.".format(sorting)
            )

    def __call__(self, example):
        outputs = {des["name"]: arr for arr, des in zip(example, self.description)}
        indices = np.argsort(self.key(outputs))
        for i, (arr, des) in enumerate(zip(example, self.description)):
            if des["name"] in [
                self.tel_desc + "_images",
                self.tel_desc + "_triggers",
                "x",
                "y",
                "z",
            ]:
                example[i] = arr[indices[:: self.step]]
        return example
