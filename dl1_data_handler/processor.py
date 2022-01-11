import copy


class DL1DataProcessor:
    def __init__(self, mode, input_description, transforms=None, validate=False):
        if mode in ["mono", "stereo", "multi-stereo"]:
            self.mode = mode
        else:
            raise ValueError(
                "Invalid mode selection '{}'. Valid options: "
                "'mono', 'stereo', 'multi-stereo'".format(mode)
            )
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.validate = validate
        self.input_description = copy.deepcopy(input_description)
        for transform in self.transforms:
            input_description = transform.describe(input_description)
        self.output_description = input_description

    def process(self, example):
        for transform in self.transforms:
            example = transform(example)
            if self.validate:
                transform.validate(example)
        return example


class Transform:
    def __init__(self):
        self.description = []

    def __call__(self, example):
        return example

    def describe(self, description):
        self.description = description
        return self.description

    def validate(self, example):
        if len(example) != len(self.description):
            raise ValueError(
                "{}: Length mismatch. Description: {}. "
                "Example: {}.".format(
                    self.__class__.__name__, len(self.description), len(example)
                )
            )
        for arr, des in zip(example, self.description):
            if arr.shape != des["shape"]:
                raise ValueError(
                    "{}: Shape mismatch. Description item: {}. "
                    "Example item: {}.".format(self.__class__.__name__, des, arr)
                )
            if arr.dtype != des["dtype"]:
                raise ValueError(
                    "{}: Dtype mismatch. Description: {}. "
                    "Example: {}.".format(self.__class__.__name__, des, arr)
                )
