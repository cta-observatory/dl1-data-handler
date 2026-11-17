"""
This module defines the ``DLDataReader`` subclass ``DLTriggerReader`` that hold the basic reading and processing functionality for trigger data.
"""

__all__ = [
    "DLTriggerReader",
    "apply_digital_sum",
    "quantised_per_flower",
    "get_window_start",
]

import os
import threading

import numpy as np
import tables
from astropy.table import (
    Table,
    join,
    vstack,
)
from ctapipe.core.traits import (
    Bool,
    CaselessStrEnum,
    Dict,
    Int,
    List,
)
from ctapipe.io import read_table
from ctapipe.io.datalevels import DataLevel

from dl1_data_handler.io import DLWaveformReader, get_unmapped_waveform

lock = threading.Lock()


def apply_digital_sum(waveform, mapper, l1_settings):
    """
    Apply the L1 digital sum to the raw waveform.

    The digital sum is computed per ``flower`` (7 pixels) or per ``superflower``
    (the flower and its neighbor flowers) and thresholded (``>= threshold``).

    Parameters
    ----------
    waveform : numpy.ndarray
        Raw waveform of the complete camera, with shape (n_pixels, time).
    mapper : HexagonalPatchMapper
        Mapper with the AdvCam trigger geometry.
    l1_settings : dict
        - ``eps`` : ``flower`` or ``superflower``.
        - ``threshold`` : threshold in ADC counts for the digital sum.

    Returns
    -------
    bin_flowers : numpy.ndarray
        Trigger mask of the thresholded digital sum, with shape (n_flowers, time).
    flower_sums : numpy.ndarray
        Digital sums, with shape (n_flowers, time).
    """
    if l1_settings["eps"] == "flower":
        flower_sums = waveform[mapper.fl_neighbor_array_l1].sum(axis=1)
    else:
        # Superflowers on the camera border have less than 7 flowers (padded with -1).
        flower_sums = (
            waveform[mapper.supfl_neighbor_array_l1]
            * mapper.supfl_neighbor_l1_mask[..., None]
        ).sum(axis=1)
    bin_flowers = (flower_sums >= l1_settings["threshold"]).astype(int)
    return bin_flowers, flower_sums


def quantised_per_flower(flower_sums, bin_flowers, l1_settings, quant_step):
    """
    Retrieve per FEB the number of triggered flowers and the quantised amplitude.

    For each FEB (7 flowers) this returns the number of flowers that passed the L1
    threshold (3 bits) and the summed amplitude above threshold of its flowers,
    quantised with ``quant_step`` and clipped to 7 bits.

    Parameters
    ----------
    flower_sums : numpy.ndarray
        Digital sums per flower, with shape (1141, time).
    bin_flowers : numpy.ndarray
        Thresholded digital sums per flower, with shape (1141, time).
    l1_settings : dict
        Dictionary with the L1 settings (``threshold`` in ADC counts).
    quant_step : int
        Quantisation step.

    Returns
    -------
    output : numpy.ndarray
        Number of triggered flowers and quantised amplitude, with shape (163, time, 2).
    """
    # Sum all the flowers of a FEB
    flower_sums_red = np.maximum(
        flower_sums.astype(np.int32) - l1_settings["threshold"], 0
    )
    feb_sums = flower_sums_red.reshape(163, 7, flower_sums.shape[-1])
    feb_summed = feb_sums.sum(axis=1)
    # Quantize the amplitudes
    vals_q = np.floor_divide(feb_summed, quant_step)
    vals_q = np.clip(vals_q, 0, 127).astype(np.uint16)
    # Retrieve number of triggered flowers per FEB
    mask_feb = bin_flowers.reshape(163, 7, flower_sums.shape[-1])
    mask_feb_summed = mask_feb.sum(axis=1)
    output = np.stack([mask_feb_summed, vals_q], axis=-1)
    return output


class DLTriggerReader(DLWaveformReader):
    """
    A data reader class for R0 raw waveforms of the Advanced SiPM camera (AdvCam)
    for trigger studies.

    Each example is one telescope event (``region="camera"``) or one trigger sector
    of a telescope event (``region="sector"``). Examples are labelled with the true
    number of Cherenkov p.e. (``example_class``: 1 if ``>= cpe_threshold``, 0 if no
    Cherenkov p.e.; examples in between are discarded).

    When trigger files are provided (see ``input_trigger_files``), only examples with
    a trigger are kept and a window of ``trigger_window`` samples is extracted with the
    first trigger in the second sample. In ``sector`` mode only the sectors that
    triggered inside that window are kept.
    """

    output_type = CaselessStrEnum(
        [
            "waveform",
            "masked_waveform",
            "stacked_waveform",
            "flower_mask",
            "flower_quantised",
        ],
        default_value="waveform",
        help=(
            "Set the content of each example. Shapes are given for the complete camera (sector in brackets). "
            "``waveform``: raw waveform per pixel, (7987, time) [(2989, time)]. "
            "``masked_waveform``: raw waveform multiplied by the trigger mask of its flower, (7987, time) [(2989, time)]. "
            "``stacked_waveform``: raw waveform and trigger mask as two channels, (7987, time, 2) [(2989, time, 2)]. "
            "``flower_mask``: binary trigger mask per flower, (1141, time) [(427, time)]. "
            "``flower_quantised``: number of triggered flowers and quantised amplitude per FEB, "
            "(163, time, 2) [(61, time, 2)]. "
            "Only the ``waveform`` outputs can be mapped to 2D images with other ImageMappers."
        ),
    ).tag(config=True)

    region = CaselessStrEnum(
        ["camera", "sector"],
        default_value="camera",
        help=(
            "``camera``: one example per telescope event with the complete camera. "
            "``sector``: one example per trigger sector (only AdvCam with the HexagonalPatchMapper)."
        ),
    ).tag(config=True)

    input_trigger_files = List(
        default_value=None,
        allow_none=True,
        help=(
            "One h5 trigger file per input file, with the obs_id, event_id, tel_id, table_index, "
            "trigger_per_sample, trigger_per_sector (if sector mode) and (optional) trigger_mask "
            "columns. These files can be generated with the code in https://github.com/jbuces/low_trigger. "
            "Requires ``trigger_window``."
        ),
    ).tag(config=True)

    trigger_window = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Number of samples of the window extracted around the trigger, with the first trigger in the "
            "second sample. The first trigger is searched in the 11 central samples for showers and in the "
            "complete readout for NSB events. Requires ``input_trigger_files``."
        ),
    ).tag(config=True)

    mask_from_file = Bool(
        default_value=False,
        help=(
            "Take the trigger mask from the ``trigger_mask`` column of the trigger files (e.g. TDSCAN) instead "
            "of computing the L1 mask on the fly. Only for ``masked_waveform``, ``stacked_waveform`` and ``flower_mask``."
        ),
    ).tag(config=True)

    l1_settings = Dict(
        default_value={"eps": "flower", "threshold": 2140},
        help=(
            "L1 trigger settings. "
            "``eps``: perform the ADC digital sum per ``flower`` or per ``superflower`` centered in each flower. "
            "``threshold``: threshold in ADC counts (``>=``) above which flowers are considered as triggered."
        ),
    ).tag(config=True)

    quantisation_step = Int(
        default_value=1,
        help="Quantisation step for the ``flower_quantised`` output.",
    ).tag(config=True)

    flatten_flower_quantised = Bool(
        default_value=False,
        help="Reshape the ``flower_quantised`` output from (n_febs, time, 2) to (n_febs, time*2) for 2D convolutions.",
    ).tag(config=True)

    cpe_threshold = Int(
        default_value=1,
        help=(
            "Minimum number of true Cherenkov p.e. (``>=``) of an example to be labelled as shower (class 1). "
            "Examples without Cherenkov p.e. are labelled as NSB (class 0) and the ones in between are discarded."
        ),
    ).tag(config=True)

    balance_classes = Bool(
        default_value=False,
        help="Randomly select from the most populated class the same number of examples as the other class.",
    ).tag(config=True)

    random_seed = Int(
        default_value=0,
        help="Seed for the random selection of ``balance_classes``.",
    ).tag(config=True)

    subtract_baseline = Int(
        default_value=None,
        allow_none=True,
        help="Value in ADC counts subtracted to each pixel for the ``waveform`` outputs.",
    ).tag(config=True)

    add_dummy_channel = Bool(
        default_value=False,
        help=(
            "Add an extra dimension of size 1 at the end to perform 3D convolutions. Not valid for the outputs "
            "that already have a channel dimension (``stacked_waveform`` and non flattened ``flower_quantised``)."
        ),
    ).tag(config=True)

    data_level = DataLevel.R0
    class_column = "example_class"

    def __init__(
        self,
        input_url_signal,
        input_url_background=[],
        config=None,
        parent=None,
        **kwargs,
    ):
        # The example identifiers (and the checks of the settings) are constructed in
        # the parent __init__, see '_process_example_identifiers'.
        super().__init__(
            input_url_signal=input_url_signal,
            input_url_background=input_url_background,
            config=config,
            parent=parent,
            **kwargs,
        )
        mapper = self.image_mappers[self.cam_name]

        # With a trigger window the complete readout is loaded and cropped afterwards.
        if self.trigger_window is not None:
            if self.trigger_window >= self.readout_length:
                raise ValueError(
                    f"'trigger_window' must be smaller than the readout length ({self.readout_length})."
                )
            self.sequence_length = self.readout_length
            self.waveform_settings["seq_length"] = self.readout_length
            time = self.trigger_window
        else:
            time = self.sequence_length

        # Input shape and neighbor array (for indexed convolutions) of each output
        sector = self.region == "sector"
        if self.output_type == "flower_quantised":
            self.neighbor_array = (
                mapper.sect0_feb_neighbors if sector else mapper.feb_neighbors
            )
            n_febs = len(self.neighbor_array)
            if self.flatten_flower_quantised:
                self.input_shape = (n_febs, time * 2)
            else:
                self.input_shape = (n_febs, time, 2)
        elif self.output_type == "flower_mask":
            self.neighbor_array = (
                mapper.sect0_flower_neighbors
                if sector
                else mapper.fl_neighbor_array_tdscan
            )
            self.input_shape = (len(self.neighbor_array), time)
        else:
            if sector:
                self.neighbor_array = mapper.sect0_neighbors
                self.input_shape = (len(self.neighbor_array), time)
            elif self._use_image_mapping():
                self.neighbor_array = None
                self.input_shape = (mapper.image_shape, mapper.image_shape, time)
            else:
                self.neighbor_array = mapper.cam_neighbor_array
                self.input_shape = (mapper.n_pixels, time)
            if self.output_type == "stacked_waveform":
                self.input_shape = self.input_shape + (2,)

        if self.add_dummy_channel:
            self.input_shape = self.input_shape + (1,)

    def _use_image_mapping(self):
        # Map the waveforms to 2D images when the mapper is not the 'HexagonalPatchMapper'
        # and not the old indexed convolution mapper.
        mapper = self.image_mappers[self.cam_name]
        return mapper.cam_neighbor_array is None and mapper.index_matrix is None

    def _check_settings(self):
        mapper = self.image_mappers[self.cam_name]
        has_sectors = getattr(mapper, "sector_mappings", None) is not None
        if self.mode != "mono":
            raise ValueError("DLTriggerReader only supports the mono mode.")
        if (
            self.output_type != "waveform" or self.region == "sector"
        ) and not has_sectors:
            raise ValueError(
                f"Output type '{self.output_type}' with region '{self.region}' is only available for "
                f"the AdvCamSiPM with the HexagonalPatchMapper (camera: '{self.cam_name}')."
            )
        if self.output_type != "waveform" and self.l1_settings["eps"] not in [
            "flower",
            "superflower",
        ]:
            raise ValueError(
                f"l1_settings eps must be 'flower' or 'superflower', got '{self.l1_settings['eps']}'."
            )
        if self.cpe_threshold < 1:
            raise ValueError(
                "'cpe_threshold' must be >= 1 (examples without Cherenkov p.e. are the NSB class)."
            )
        if bool(self.input_trigger_files) != (self.trigger_window is not None):
            raise ValueError(
                "'input_trigger_files' and 'trigger_window' must be set together."
            )
        if self.mask_from_file and (
            not self.input_trigger_files
            or self.output_type
            not in ["masked_waveform", "stacked_waveform", "flower_mask"]
        ):
            raise ValueError(
                "'mask_from_file' requires 'input_trigger_files' and output type 'masked_waveform', "
                "'stacked_waveform' or 'flower_mask'."
            )
        if self.add_dummy_channel and (
            self.output_type == "stacked_waveform"
            or (
                self.output_type == "flower_quantised"
                and not self.flatten_flower_quantised
            )
        ):
            raise ValueError(
                f"'add_dummy_channel' is not valid for output type '{self.output_type}'."
            )

    def _process_example_identifiers(self, example_identifiers):
        """
        Label the examples and apply the trigger selection.

        Steps:
        1. Add the trigger information from the trigger files (if provided).
        2. Read the true Cherenkov p.e. of each event and separate shower events
           (``>= cpe_threshold``) from NSB events (no Cherenkov p.e.).
        3. With trigger files, compute the start of the trigger window and keep only
           the events with a trigger in the allowed range.
        4. Create one example per event (camera) or per sector, with the Cherenkov p.e.
           and the class of the example. In sector mode with trigger files only the
           sectors that triggered inside the window are kept.
        5. Balance the classes if requested.
        """
        self._check_settings()

        # 1. Trigger information
        if self.input_trigger_files:
            example_identifiers = self._add_trigger_table(example_identifiers)

        # 2. True Cherenkov p.e. per event
        event_cpe = self._read_true_pe(example_identifiers, "true_image_sum")
        is_shower = event_cpe >= self.cpe_threshold
        is_nsb = event_cpe == 0

        # 3. Trigger window
        if self.input_trigger_files:
            trigger_per_sample = np.stack(
                example_identifiers["trigger_per_sample"]
            ).astype(bool)
            window_start = get_window_start(
                trigger_per_sample, is_shower, is_nsb, self.trigger_window
            )
            example_identifiers.add_column(window_start, name="window_start")
            keep = window_start >= 0
        else:
            keep = is_shower | is_nsb
        example_identifiers = example_identifiers[keep]
        event_cpe = event_cpe[keep]

        # 4. Examples and labels
        if self.region == "camera":
            cpe = event_cpe
            example_identifiers.add_column(cpe, name="cherenkov_pe")
        else:
            # True Cherenkov p.e. per sector, shape (n_events, n_sectors)
            sector_cpe = self._read_true_pe(example_identifiers, "true_image")
            n_events, n_sectors = sector_cpe.shape
            if self.input_trigger_files:
                sector_triggered = self._get_triggered_sectors(example_identifiers)
            else:
                sector_triggered = np.ones((n_events, n_sectors), dtype=bool)
            keep_sector = sector_triggered & (
                (sector_cpe >= self.cpe_threshold) | (sector_cpe == 0)
            )
            event_idx, sector_idx = np.nonzero(keep_sector)
            cpe = sector_cpe[event_idx, sector_idx]
            example_identifiers = example_identifiers[event_idx]
            example_identifiers.add_column(sector_idx, name="sector_index")
            example_identifiers.add_column(cpe, name="cherenkov_pe")
        example_identifiers.add_column(
            (cpe >= self.cpe_threshold).astype(np.int64), name=self.class_column
        )

        # The packed masks are only kept if they are needed to build the features.
        if "trigger_mask" in example_identifiers.colnames and not self.mask_from_file:
            example_identifiers.remove_column("trigger_mask")

        # 5. Balance the classes
        if self.balance_classes:
            example_identifiers = self._balance(example_identifiers)
        return example_identifiers

    def _add_trigger_table(self, example_identifiers):
        """
        Join the trigger information of the trigger files to the example identifiers.

        Each trigger file is paired with its input file with the ``input_file``
        attribute of the trigger table. If it is missing, the trigger files are
        paired with the (sorted) input files by sorted file name.
        """
        input_files = list(self.files)
        if len(self.input_trigger_files) != len(input_files):
            raise ValueError(
                f"Number of trigger files ({len(self.input_trigger_files)}) and input files ({len(input_files)}) differ."
            )
        need_mask = self.mask_from_file
        need_sectors = self.region == "sector"

        # Pair each input file with its trigger file
        trigger_file_of = {}
        for trigger_file in self.input_trigger_files:
            with tables.open_file(trigger_file, mode="r") as h5file:
                attrs = h5file.get_node("/table")._v_attrs
                input_name = attrs["input_file"] if "input_file" in attrs else None
                if need_mask:
                    if "trigger_mask_shape" not in attrs:
                        raise ValueError(
                            f"Trigger file '{trigger_file}' has no 'trigger_mask' column."
                        )
                    self._trigger_mask_shape = tuple(attrs["trigger_mask_shape"])
                    self._trigger_mask_bits = int(attrs["trigger_mask_bits"])
                    self._trigger_mask_bitorder = str(attrs["trigger_mask_bitorder"])
            if input_name is None:
                trigger_file_of = None
                break
            trigger_file_of[str(input_name)] = trigger_file
        if trigger_file_of is None:
            self.log.warning(
                "Trigger files without 'input_file' attribute: pairing trigger files and input files by sorted name."
            )
            trigger_file_of = dict(
                zip(
                    [os.path.basename(f) for f in input_files],
                    sorted(self.input_trigger_files),
                )
            )
        if need_mask and self._trigger_mask_shape[0] != len(
            self.image_mappers[self.cam_name].fl_neighbor_array_l1
        ):
            raise ValueError(
                f"The trigger masks must be per flower, got shape {self._trigger_mask_shape}."
            )

        keys = ["obs_id", "event_id", "tel_id", "table_index"]
        trigger_columns = (
            keys
            + ["trigger_per_sample"]
            + (["trigger_mask"] if need_mask else [])
            + (["trigger_per_sector"] if need_sectors else [])
        )
        tables_per_file = []
        for file_idx, filename in enumerate(input_files):
            trigger_file = trigger_file_of.get(os.path.basename(filename))
            if trigger_file is None:
                raise ValueError(f"No trigger file found for input file '{filename}'.")
            trigger_table = read_table(trigger_file, "/table")
            if need_sectors and "trigger_per_sector" not in trigger_table.colnames:
                raise ValueError(
                    f"Trigger file '{trigger_file}' has no 'trigger_per_sector' column (needed for region 'sector')."
                )
            trigger_table.keep_columns(trigger_columns)
            file_examples = example_identifiers[
                example_identifiers["file_index"] == file_idx
            ]
            merged = join(file_examples, trigger_table, keys=keys, join_type="inner")
            if len(merged) != len(file_examples):
                raise ValueError(
                    f"{len(file_examples) - len(merged)} events of '{filename}' are missing in '{trigger_file}'."
                )
            tables_per_file.append(merged)
        return vstack(tables_per_file)

    def _read_true_pe(self, example_identifiers, field, chunk_size=5000):
        """
        Read the true Cherenkov p.e. from the simulated images.

        ``field="true_image_sum"`` returns the p.e. per event, shape (n_examples,).
        ``field="true_image"`` returns the p.e. per sector, shape (n_examples, n_sectors).
        """
        mapper = self.image_mappers[self.cam_name]
        n_examples = len(example_identifiers)
        if field == "true_image_sum":
            cpe = np.zeros(n_examples, dtype=np.int64)
        else:
            sectors = mapper.sectors_bool.T.astype(
                np.int64
            )  # shape: (n_pixels, n_sectors)
            cpe = np.zeros((n_examples, sectors.shape[1]), dtype=np.int64)

        file_index = np.asarray(example_identifiers["file_index"])
        tel_ids = np.asarray(example_identifiers["tel_id"])
        table_index = np.asarray(example_identifiers["table_index"])
        filenames = list(self.files)
        for file_idx in np.unique(file_index):
            for tel_id in np.unique(tel_ids[file_index == file_idx]):
                rows = np.nonzero((file_index == file_idx) & (tel_ids == tel_id))[0]
                # Read the rows sorted by table index (faster), and store them back in place
                rows = rows[np.argsort(table_index[rows])]
                with lock:
                    node = self.files[
                        filenames[file_idx]
                    ].root.simulation.event.telescope.images._f_get_child(
                        f"tel_{tel_id:03d}"
                    )
                    for start in range(0, len(rows), chunk_size):
                        chunk = rows[start : start + chunk_size]
                        values = node.read_coordinates(table_index[chunk], field=field)
                        if field == "true_image_sum":
                            cpe[chunk] = values
                        else:
                            cpe[chunk] = values.astype(np.int64) @ sectors
        return cpe

    def _unpack_masks(self, packed):
        # Unpack the trigger masks written with np.packbits, shape (..., n_flowers, time)
        bits = np.unpackbits(packed, axis=-1, bitorder=self._trigger_mask_bitorder)[
            ..., : self._trigger_mask_bits
        ]
        return bits.reshape(packed.shape[:-1] + self._trigger_mask_shape)

    def _get_triggered_sectors(self, example_identifiers):
        """
        Find which sectors triggered inside the trigger window of each event, using the
        ``trigger_per_sector`` column of the trigger files. Returns a boolean array with
        shape (n_events, n_sectors).
        """
        trigger_per_sector = np.asarray(
            example_identifiers["trigger_per_sector"]
        ).astype(bool)  # (n, n_sectors, time)
        window_start = np.asarray(example_identifiers["window_start"])[:, None]
        time = np.arange(trigger_per_sector.shape[-1])[None, :]
        in_window = (time >= window_start) & (
            time < window_start + self.trigger_window
        )  # (n, time)
        return (trigger_per_sector & in_window[:, None, :]).any(axis=-1)

    def _balance(self, example_identifiers):
        # Keep all the examples of the smallest class and the same number of randomly
        # selected examples of the largest class.
        labels = np.asarray(example_identifiers[self.class_column])
        idx_0, idx_1 = np.nonzero(labels == 0)[0], np.nonzero(labels == 1)[0]
        n_min = min(len(idx_0), len(idx_1))
        if n_min == 0:
            raise ValueError(
                f"Cannot balance the classes: {len(idx_0)} NSB and {len(idx_1)} shower examples."
            )
        rng = np.random.default_rng(self.random_seed)
        idx_0 = rng.choice(idx_0, n_min, replace=False)
        idx_1 = rng.choice(idx_1, n_min, replace=False)
        return example_identifiers[np.sort(np.concatenate([idx_0, idx_1]))]

    def _build_features(self, waveform, file_mask, sector):
        """
        Build the features of one example from its raw waveform.

        Parameters
        ----------
        waveform : numpy.ndarray
            Raw waveform of the complete camera, shape (n_pixels, time).
        file_mask : numpy.ndarray or None
            Trigger mask per flower from the trigger file, shape (n_flowers, time).
        sector : int or None
            Sector index in ``sector`` mode.
        """
        mapper = self.image_mappers[self.cam_name]

        # L1 digital sum (always on the complete camera)
        if self.output_type != "waveform":
            l1_mask, flower_sums = apply_digital_sum(waveform, mapper, self.l1_settings)
            mask = file_mask if file_mask is not None else l1_mask

        if self.output_type == "flower_quantised":
            features = quantised_per_flower(
                flower_sums, l1_mask, self.l1_settings, self.quantisation_step
            )
            if sector is not None:
                features = mapper.get_sector(features, sector, "feb")
            if self.flatten_flower_quantised:
                features = features.reshape(features.shape[0], -1)
        elif self.output_type == "flower_mask":
            features = np.asarray(mask, dtype=np.int64)
            if sector is not None:
                features = mapper.get_sector(features, sector, "flower")
        else:
            if self.subtract_baseline:
                waveform = waveform - self.subtract_baseline
            if self.output_type != "waveform":
                pixel_mask = np.repeat(mask, 7, axis=0).astype(
                    np.int64
                )  # shape: (n_pixels, time)
            if sector is not None:
                waveform = mapper.get_sector(waveform, sector, "pixel")
                if self.output_type != "waveform":
                    pixel_mask = mapper.get_sector(pixel_mask, sector, "pixel")
            elif self._use_image_mapping():
                waveform = mapper.map_image(waveform).astype(np.int64)
                if self.output_type != "waveform":
                    pixel_mask = mapper.map_image(pixel_mask).astype(np.int64)

            if self.output_type == "waveform":
                features = waveform
            elif self.output_type == "masked_waveform":
                features = waveform * pixel_mask
            else:
                features = np.stack([waveform, pixel_mask], axis=-1)

        if self.add_dummy_channel:
            features = features[..., np.newaxis]
        return features

    def _append_features(self, batch) -> Table:
        """
        Append the features of each example to the batch.

        Reads the R0 waveform of each example, crops the trigger window (if
        ``trigger_window``) and builds the selected ``output_type``.

        Parameters
        ----------
        batch : astropy.table.Table
            Example identifiers of the batch with at least the ``file_index``,
            ``table_index`` and ``tel_id`` columns, plus ``window_start`` (with trigger
            files), ``trigger_mask`` (with ``mask_from_file``) and ``sector_index``
            (with ``region="sector"``).

        Returns
        -------
        batch : astropy.table.Table
            The input batch with the appended ``features`` column.
        """
        mapper = self.image_mappers[self.cam_name]
        filenames = list(self.files)
        features = []
        for i, (file_idx, table_idx, tel_id) in enumerate(
            batch.iterrows("file_index", "table_index", "tel_id")
        ):
            with lock:
                child = self.files[
                    filenames[file_idx]
                ].root.r0.event.telescope._f_get_child(f"tel_{tel_id:03d}")
                waveform = get_unmapped_waveform(
                    child[table_idx], self.waveform_settings, mapper.geometry
                )
            waveform = waveform.astype(np.int64)

            file_mask = None
            if self.trigger_window is not None:
                start = batch["window_start"][i]
                stop = start + self.trigger_window
                waveform = waveform[:, start:stop]
                if self.mask_from_file:
                    file_mask = self._unpack_masks(
                        np.asarray(batch["trigger_mask"][i])
                    )[:, start:stop]

            sector = batch["sector_index"][i] if self.region == "sector" else None
            features.append(self._build_features(waveform, file_mask, sector))

        batch.add_column(features, name="features", index=8)
        return batch


def get_window_start(trigger_per_sample, is_shower, is_nsb, window):
    """
    Compute the start of the trigger window of each event.

    The window starts one sample before the first trigger, so the first trigger is
    always in the second sample of the window. The first trigger is searched in the
    11 central samples for showers and in the complete readout for NSB events, and
    it must leave room for the complete window (samples 1 to n_samples - window + 1).

    Parameters
    ----------
    trigger_per_sample : numpy.ndarray
        Boolean array with shape (n_events, n_samples).
    is_shower, is_nsb : numpy.ndarray
        Boolean arrays with shape (n_events,).
    window : int
        Number of samples of the window.

    Returns
    -------
    window_start : numpy.ndarray
        Start sample of the window, -1 for events without a valid trigger.
    """
    n_samples = trigger_per_sample.shape[1]
    center = n_samples // 2
    allowed = np.zeros_like(trigger_per_sample, dtype=bool)
    allowed[is_shower, center - 5 : center + 6] = True
    allowed[is_nsb, :] = True
    # The first trigger must be in [1, n_samples - window + 1]
    allowed[:, 0] = False
    allowed[:, n_samples - window + 2 :] = False
    valid = trigger_per_sample & allowed
    has_trigger = valid.any(axis=1)
    first_trigger = valid.argmax(axis=1)
    return np.where(has_trigger, first_trigger - 1, -1)
