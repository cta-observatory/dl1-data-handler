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
    sort_by_intensity : bool, optional
        Whether to sort the events based on the hillas intensity for stereo analysis.
    stack_telescope_images : bool, optional
        Whether to stack the telescope images for stereo analysis.

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
        sort_by_intensity=False,
        stack_telescope_images=False,
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
        self.stack_telescope_images = stack_telescope_images
        self.sort_by_intensity = sort_by_intensity

        # Set the input shape based on the mode of the DLDataReader
        if self.DLDataReader.mode == "mono":
            self.input_shape = self.DLDataReader.input_shape
        elif self.DLDataReader.mode == "stereo":
            self.input_shape = self.DLDataReader.input_shape[
                list(self.DLDataReader.selected_telescopes)[0]
            ]
            # Reshape inputs into proper dimensions for the stereo analysis with stacked images
            if self.stack_telescope_images:
                self.input_shape = (
                    self.input_shape[1],
                    self.input_shape[2],
                    self.input_shape[0] * self.input_shape[3],
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
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        labels = {}
        if self.DLDataReader.mode == "mono":
            batch = self.DLDataReader.generate_mono_batch(batch_indices)
            # Retrieve the telescope images and store in the features dictionary 
            features = {"input": batch["features"].data}
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
        elif self.DLDataReader.mode == "stereo":
            batch = self.DLDataReader.generate_stereo_batch(batch_indices)
            if self.DLDataReader.process_type == ProcessType.Simulation:
                batch_grouped = batch.group_by(["obs_id", "event_id", "tel_type_id", "true_shower_primary_class"])
            elif self.DLDataReader.process_type == ProcessType.Observation:
                batch_grouped = batch.group_by(["obs_id", "event_id", "tel_type_id"])
            features = []
            true_shower_primary_class = []
            log_true_energy = []
            spherical_offset_az, spherical_offset_alt, angular_separation = [], [], []
            for group_element in batch_grouped.groups:
                if self.sort_by_intensity:
                    # Sort images by the hillas intensity in a given batch if requested
                    group_element.sort(["hillas_intensity"], reverse=True)
                # Stack the telescope images for stereo analysis
                if self.stack_telescope_images:
                    # Retrieve the telescope images
                    plain_features = group_element["features"].data
                    # Stack the telescope images along the last axis
                    stacked_features = np.concatenate(
                        [
                            plain_features[i]
                            for i in range(plain_features.shape[0])
                        ],
                        axis=-1,
                    )
                    # Append the stacked images to the features list
                    # shape: (batch_size, image_shape, image_shape, n_channels * n_tel)
                    features.append(stacked_features)
                else:
                    # Append the plain images to the features list
                    # shape: (batch_size, n_tel, image_shape, image_shape, n_channels)
                    features.append(group_element["features"].data)
                # Retrieve the labels for the tasks
                # FIXME: This won't work for divergent pointing directions
                if "type" in self.tasks:
                    true_shower_primary_class.append(group_element["true_shower_primary_class"].data[0])
                if "energy" in self.tasks:
                    log_true_energy.append(group_element["log_true_energy"].data[0])
                if "direction" in self.tasks:
                    spherical_offset_az.append(group_element["spherical_offset_az"].data[0])
                    spherical_offset_alt.append(group_element["spherical_offset_alt"].data[0])
                    angular_separation.append(group_element["angular_separation"].data[0])
            # Store the labels in the labels dictionary
            if "type" in self.tasks:
                labels["type"] = to_categorical(
                    np.array(true_shower_primary_class),
                    num_classes=2,
                )
                # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
                # https://github.com/keras-team/keras/issues/11735
                if len(self.tasks) == 1:
                    labels = to_categorical(
                        np.array(true_shower_primary_class),
                        num_classes=2,
                    )
            if "energy" in self.tasks:
                labels["energy"] = np.array(log_true_energy)
            if "direction" in self.tasks:
                labels["direction"] = np.stack(
                    (
                        np.array(spherical_offset_az),
                        np.array(spherical_offset_alt),
                        np.array(angular_separation),
                    ),
                    axis=1,
                )
            # Store the fatures in the features dictionary 
            features = {"input": np.array(features)}
        # Temp fix for supporting keras2 & keras3
        if int(keras.__version__.split(".")[0]) >= 3:
            features = features["input"]
        return features, labels
