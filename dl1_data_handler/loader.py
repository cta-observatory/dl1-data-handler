import numpy as np
import astropy.units as u
from keras.utils import Sequence, to_categorical


class DLDataLoader(Sequence):
    "Generates batches for Keras application"

    def __init__(
        self,
        DLDataReader,
        indices,
        tasks,
        batch_size=64,
        shuffle=True,
        random_seed=1234,
    ):
        "Initialization"
        self.DLDataReader = DLDataReader
        self.indices = indices
        self.tasks = tasks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.on_epoch_end()

        # Get the input shape for the convolutional neural network
        self.image_shape = self.DLDataReader.image_mapper.image_shape
        if self.DLDataReader.__class__.__name__ == "DLImageReader":
            self.channel_shape = len(self.DLDataReader.img_channels)
        elif self.DLDataReader.__class__.__name__ == "DLWaveformReader":
            self.channel_shape = self.DLDataReader.sequence_length

        self.input_shape = (
            self.image_shape,
            self.image_shape,
            self.channel_shape,
        )

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        "Generate one batch of data"
        # If shuffle is set to false, CTLearn is predicting and therfore all DL1b
        # parameters are retrieved.
        dl1b_parameter_list = None
        if not self.shuffle:
            dl1b_parameter_list = self.DLDataReader.dl1b_parameter_colnames

        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if self.DLDataReader.mode == "mono":
            features, batch = self.DLDataReader.mono_batch_generation(
                batch_indices=batch_indices,
                dl1b_parameter_list=dl1b_parameter_list
            )
        elif self.DLDataReader.mode == "stereo":
            features, batch = self.DLDataReader.stereo_batch_generation(
                batch_indices=batch_indices,
                dl1b_parameter_list=dl1b_parameter_list
            )
        # Generate the labels for each task
        label = {}
        if "type" in self.tasks:
            label["type"] = to_categorical(
                batch["true_shower_primary_class"].data,
                num_classes=self.DLDataReader.n_classes,
            )
        if "energy" in self.tasks:
            label["energy"] = batch["log_true_energy"].data
        if "direction" in self.tasks:
            label["direction"] = np.stack(
                (
                    batch["spherical_offset_az"].data,
                    batch["spherical_offset_alt"].data,
                    batch["angular_separation"].data,
                ),
                axis=1,
            )
        return features, label
