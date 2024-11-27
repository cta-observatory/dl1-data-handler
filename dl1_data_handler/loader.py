import numpy as np
import astropy.units as u
import keras
from keras.utils import Sequence, to_categorical


class DLDataLoader(Sequence):
    """
    Generates batches for Keras application.

    DLDataLoader is a data loader class that inherits from ``~keras.utils.Sequence``.
    It is designed to handle and load data for deep learning models in a batch-wise manner.

    Attributes:
    -----------
    data_reader : DLDataReader
        An instance of DLDataReader to read the input data.
    indices : list
        List of indices to specify the data to be loaded.
    tasks : list
        List of tasks to be performed on the data to properly set up the labels.
    batch_size : int
        Size of the batch to load the data.
    random_seed : int, optional
        Whether to shuffle the data after each epoch with a provided random seed.

    Methods:
    --------
    __len__():
        Returns the number of batches per epoch.
    __getitem__(index):
        Generates one batch of data.
    on_epoch_end():
        Updates indices after each epoch if random seed is provided.
    """

    def __init__(
        self,
        DLDataReader,
        indices,
        tasks,
        batch_size=64,
        random_seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        "Initialization"
        self.DLDataReader = DLDataReader
        self.indices = indices
        self.tasks = tasks
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.on_epoch_end()

        # Get the input shape for the convolutional neural network
        self.image_shape = self.DLDataReader.image_mappers[self.DLDataReader.cam_name].image_shape
        if self.DLDataReader.__class__.__name__ == "DLImageReader":
            self.channel_shape = len(self.DLDataReader.channels)
        elif self.DLDataReader.__class__.__name__ == "DLWaveformReader":
            self.channel_shape = self.DLDataReader.sequence_length

        self.input_shape = (
            self.image_shape,
            self.image_shape,
            self.channel_shape,
        )

    def __len__(self):
        """
        Returns the number of batches per epoch.

        This method calculates the number of batches required to cover the entire dataset
        based on the batch size.

        Returns:
        --------
        int
            Number of batches per epoch.
        """
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        """
        Updates indices after each epoch. If a random seed is provided, the indices are shuffled.

        This method is called at the end of each epoch to ensure that the data is shuffled
        if the shuffle attribute is set to True. This helps in improving the training process
        by providing the model with a different order of data in each epoch.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """
        Generates one batch of data.

        Parameters:
        -----------
        index : int
            Index of the batch to generate.

        Returns:
        --------
        tuple
            A tuple containing the input data as features and the corresponding labels.
        """
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if self.DLDataReader.mode == "mono":
            features, batch = self.DLDataReader.mono_batch_generation(
                batch_indices=batch_indices,
            )
        elif self.DLDataReader.mode == "stereo":
            features, batch = self.DLDataReader.stereo_batch_generation(
                batch_indices=batch_indices,
            )
        # Generate the labels for each task
        labels = {}
        if "type" in self.tasks:
            labels["type"] = to_categorical(
                batch["true_shower_primary_class"].data,
                num_classes=2,
            )
            # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
            # https://github.com/keras-team/keras/issues/11735
            if len(self.tasks) == 1:
                labels = to_categorical(
                    batch["true_shower_primary_class"].data,
                    num_classes=2,
                )
        if "energy" in self.tasks:
            labels["energy"] = batch["log_true_energy"].data
        if "direction" in self.tasks:
            labels["direction"] = np.stack(
                (
                    batch["spherical_offset_az"].data,
                    batch["spherical_offset_alt"].data,
                    batch["angular_separation"].data,
                ),
                axis=1,
            )
        # Temp fix for supporting keras2 & keras3
        if int(keras.__version__.split(".")[0]) >= 3:
            features = features["input"]
        return features, labels
