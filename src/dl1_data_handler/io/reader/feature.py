"""
This module defines the ``DLDataReader`` subclass ``DLFeatureVectorReader`` that hold the basic reading and processing functionality for feature vector data.
"""

__all__ = [
    "DLFeatureVectorReader",
    "get_feature_vectors",
]

import numpy as np
import threading
from astropy.table import (
    Table,
)

from ctapipe.core.traits import (
    Bool,
    List,
    CaselessStrEnum,
    Unicode,
)
from dl1_data_handler.io import DLDataReader

lock = threading.Lock()

def get_feature_vectors(dl1_event, prefix, feature_vector_types) -> list:
    """
    Retrieve selected feature vectors from a DL1 event.

    This function processes the DL1 event data to retrieve feature vectors
    based on the specified feature vector types and prefix. It returns a list
    of feature vectors for the selected types, which can be used as input features
    for the neural networks.

    Parameters
    ----------
    dl1_event : astropy.table.Table
        A table containing DL1 event data, including feature vectors for classification,
        energy regression, and geometry/direction regression.
    prefix : str
        A prefix for the feature vector group in the HDF5 file.
    feature_vector_types : list of str
        A list of feature vector types to be loaded from the DL1 data, such as
        ``classification``, ``energy``, and ``geometry``.

    Returns
    -------
    feature_vectors : list of np.ndarray
        A list of feature vectors for the selected types.
    """
    feature_vectors = []
    for feature_vector_type in feature_vector_types:
        feature_vectors.append(
            dl1_event[f"{prefix}_{feature_vector_type}_feature_vectors"]
        )
    return feature_vectors


class DLFeatureVectorReader(DLDataReader):
    """
    A data reader class for handling DL1 feature vector data.

    This class extends the ``DLDataReader`` to specifically handle the reading of
    DL1 feature vectors, obtained from a previous CTLearnModel. It supports the reading
    of both ``telescope``- and ``subarray``-level feature vectors. This reader class only
    supports the reading in stereo mode.
    """

    prefixes = List(
        trait=Unicode(),
        default_value=["CTLearn"],
        allow_none=False,
        help="List of prefixes for the feature vector group in the HDF5 file.",
    ).tag(config=True)

    feature_vector_types = List(
        trait=CaselessStrEnum(
            [
                "classification",
                "energy",
                "geometry",
            ]
        ),
        allow_none=False,
        help=(
            "Set the type of the feature vector to be loaded from the DL1 data. "
            "classification: load feature vectors used for particle classification, "
            "energy: load feature vectors used for energy regression, "
            "geometry: load feature vectors used for geometry/direction regression."
        ),
    ).tag(config=True)

    load_telescope_features = Bool(
        default_value=True,
        help="Set whether to load telescope feature vectors from the DL1 data.",
    ).tag(config=True)

    load_subarray_features = Bool(
        default_value=False,
        help="Set whether to load subarray feature vectors from the DL1 data.",
    ).tag(config=True)

    def __init__(
        self,
        input_url_signal,
        input_url_background=[],
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            input_url_signal=input_url_signal,
            input_url_background=input_url_background,
            config=config,
            parent=parent,
            **kwargs,
        )
        # Check that the mode is consistent with the feature reader.
        # The feature reader only supports stereo mode.
        if self.mode != "stereo":
            raise ValueError(
                f"'{self.__class__.__name__}' only supports 'stereo' mode. "
                "Please set the mode to 'stereo' or use one of the other subclasses."
            )
        # Check that at least one of the feature vector types is selected
        if not self.load_telescope_features and not self.load_subarray_features:
            raise ValueError(
                "No loading of feature vectors selected. Please set 'load_telescope_features' "
                "and/or 'load_subarray_features' to 'True'."
            )

    def _append_features(self, batch) -> Table:
        """
        Append previous obtained feature vectors to a given batch as features.

        This method processes a batch of events to append feature vectors as input features
        for the neural networks. It reads the feature vector data from the specified files
        and appends the feature vectors to the batch. The feature vectors can be loaded
        for both ``telescope``- and ``subarray``-level.

        Parameters
        ----------
        batch : astropy.table.Table
            A table containing information at minimum the following columns:
            - "file_index": List of indices corresponding to the files.
            - "table_index": List of indices corresponding to the event tables.
            - "tel_type_id": List of telescope type IDs.
            - "tel_id": List of telescope IDs.

        Returns
        -------
        batch : astropy.table.Table
            The input batch with the appended mono and stereo feature vectors.
        """
        mono_fvs, stereo_fvs = [], []
        for file_idx, table_idx, tel_type_id, tel_id in batch.iterrows(
            "file_index", "table_index", "tel_type_id", "tel_id"
        ):
            filename = list(self.files)[file_idx]
            if self.load_telescope_features:
                with lock:
                    mono_fvs_per_prefix = []
                    tel_table = f"tel_{tel_id:03d}"
                    for prefix in self.prefixes:
                        telescope_child = (
                            self.files[filename]
                            .root.dl1.event.telescope.features.__getitem__(prefix)
                            ._f_get_child(tel_table)
                        )
                        mono_fvs_per_prefix.append(
                            get_feature_vectors(
                                telescope_child[table_idx],
                                f"{prefix}_tel",
                                self.feature_vector_types,
                            )
                        )
                mono_fvs.append(mono_fvs_per_prefix)
            if self.load_subarray_features:
                with lock:
                    stereo_fvs_per_prefix = []
                    for prefix in self.prefixes:
                        subarray_child = self.files[
                            filename
                        ].root.dl1.event.subarray.features._f_get_child(prefix)
                        stereo_fvs_per_prefix.append(
                            get_feature_vectors(
                                subarray_child[table_idx],
                                prefix,
                                self.feature_vector_types,
                            )
                        )
                stereo_fvs.append(stereo_fvs_per_prefix)
        # Append the features to the batch
        if self.load_telescope_features:
            batch.add_column(np.array(mono_fvs), name="mono_feature_vectors", index=7)
        if self.load_subarray_features:
            batch.add_column(
                np.array(stereo_fvs), name="stereo_feature_vectors", index=7
            )
        return batch

    def _process_example_identifiers(self, example_identifiers) -> Table:
        """No additional labelling or selection of the example identifiers."""
        return example_identifiers
