import numpy as np
import logging
import bisect

from scipy import spatial
from scipy.sparse import csr_matrix
from scipy.ndimage import rotate
from astropy import units as u
from collections import Counter

logger = logging.getLogger(__name__)


class ImageMapper:
    def __init__(
        self,
        camera_types=None,
        pixel_positions=None,
        mapping_method=None,
        padding=None,
        interpolation_image_shape=None,
        mask_interpolation=False,
    ):

        # image_shapes should be a non static field to prevent problems
        # when multiple instances of ImageMapper are created
        self.image_shapes = {
            "LSTCam": (110, 110, 1),
            "FlashCam": (112, 112, 1),
            "NectarCam": (110, 110, 1),
            "SCTCam": (120, 120, 1),
            "DigiCam": (96, 96, 1),
            "CHEC": (48, 48, 1),
            "ASTRICam": (56, 56, 1),
            "VERITAS": (54, 54, 1),
            "MAGICCam": (78, 78, 1),
            "FACT": (90, 90, 1),
            "HESS-I": (72, 72, 1),
            "HESS-II": (104, 104, 1),
        }

        # Camera types
        if camera_types:
            self.camera_types = []
            for camera_type in camera_types:
                if camera_type in self.image_shapes:
                    self.camera_types.append(camera_type)
                else:
                    logger.error("Camera type {} isn't supported.".format(camera_type))
        else:
            self.camera_types = [cam for cam in self.image_shapes]

        # Mapping method
        if mapping_method is None:
            mapping_method = {}
        self.mapping_method = {
            **{c: "oversampling" for c in self.camera_types},
            **mapping_method,
        }

        # Interpolation image shape
        if interpolation_image_shape is None:
            interpolation_image_shape = {}
        self.interpolation_image_shape = {
            **{c: self.image_shapes[c] for c in self.camera_types},
            **interpolation_image_shape,
        }

        # Padding
        if padding is None:
            padding = {}
        self.padding = {**{c: 0 for c in self.camera_types}, **padding}

        # Mask interpolation
        self.mask = True if mask_interpolation else False

        # Pixel positions, number of pixels, mapping tables and index matrixes initialization
        self.pixel_positions = {}
        self.num_pixels = {}
        self.mapping_tables = {}
        self.index_matrixes = {}

        for camtype in self.camera_types:
            # Get a corresponding pixel positions
            if pixel_positions is None:
                try:
                    from ctapipe.instrument.camera import CameraGeometry
                except ImportError:
                    raise ImportError(
                        "The `ctapipe.instrument.camera` python module is required, if pixel_positions is `None`."
                    )
                camgeo = CameraGeometry.from_name(camtype)
                self.num_pixels[camtype] = len(camgeo.pix_id)
                self.pixel_positions[camtype] = np.column_stack(
                    [camgeo.pix_x.value, camgeo.pix_y.value]
                ).T
                if camtype in ["LSTCam", "NectarCam", "MAGICCam"]:
                    rotation_angle = -camgeo.pix_rotation.value * np.pi / 180.0
                    rotation_matrix = np.matrix(
                        [
                            [np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [np.sin(rotation_angle), np.cos(rotation_angle)],
                        ],
                        dtype=float,
                    )
                    self.pixel_positions[camtype] = np.squeeze(
                        np.asarray(
                            np.dot(rotation_matrix, self.pixel_positions[camtype])
                        )
                    )
            else:
                self.pixel_positions[camtype] = pixel_positions[camtype]
                self.num_pixels[camtype] = pixel_positions[camtype].shape[1]

            map_method = self.mapping_method[camtype]
            if map_method not in [
                "oversampling",
                "rebinning",
                "nearest_interpolation",
                "bilinear_interpolation",
                "bicubic_interpolation",
                "image_shifting",
                "axial_addressing",
                "indexed_conv",
            ]:
                raise ValueError(
                    "Hex conversion algorithm {} is not implemented.".format(map_method)
                )
            elif map_method in [
                "image_shifting",
                "axial_addressing",
                "indexed_conv",
            ] and camtype in ["ASTRICam", "CHEC", "SCTCam"]:
                raise ValueError(
                    "{} (hexagonal convolution) is not available for square pixel cameras.".format(
                        map_method
                    )
                )

            if map_method in [
                "rebinning",
                "nearest_interpolation",
                "bilinear_interpolation",
                "bicubic_interpolation",
            ]:
                self.image_shapes[camtype] = self.interpolation_image_shape[camtype]

            # At the edges of the cameras the mapping methods run into issues.
            # Therefore, we are using a default padding to ensure that the camera pixels aren't affected.
            # The default padding is removed after the conversion is finished.
            if map_method in ["image_shifting", "axial_addressing", "indexed_conv"]:
                self.default_pad = 0
            elif map_method == "bicubic_interpolation":
                self.default_pad = 3
            else:
                self.default_pad = 2

            if map_method != "oversampling" or camtype in [
                "ASTRICam",
                "CHEC",
                "SCTCam",
            ]:
                self.image_shapes[camtype] = (
                    self.image_shapes[camtype][0] + self.default_pad * 2,
                    self.image_shapes[camtype][1] + self.default_pad * 2,
                    self.image_shapes[camtype][2],
                )
            else:
                self.image_shapes[camtype] = (
                    self.image_shapes[camtype][0] + self.default_pad * 4,
                    self.image_shapes[camtype][1] + self.default_pad * 4,
                    self.image_shapes[camtype][2],
                )

            # Initializing the indexed matrix
            self.index_matrixes[camtype] = None
            # Calculating the mapping tables for the selected camera types
            self.mapping_tables[camtype] = self.generate_table(camtype)

    def map_image(self, pixels, camera_type):
        """
        :param pixels: a numpy array of values for each pixel, in order of pixel index.
        :param camera_type: a string specifying the telescope type.
        :return: a numpy array of shape [img_width, img_length, N_channels]

        Usage:

        >>> IM = dl1_data_handler.image_mapper.ImageMapper(camera_types=['LSTCam'])
        >>> one_channel = np.expand_dims(np.arange(1855), axis=1)
        >>> # Use the ImageMapper with one channel (charge or peak position):
        >>> image = IM.map_image(one_channel, 'LSTCam')
        >>> # Use the ImageMapper with two channels (charge and peak position):
        >>> two_channels = np.concatenate((one_channel, one_channel[::-1]),axis=1)
        >>> images = IM.map_image(two_channels, 'LSTCam')
        """

        # Get relevant parameters
        map_tab = self.mapping_tables[camera_type]
        n_channels = pixels.shape[1]
        if n_channels != self.image_shapes[camera_type][2]:
            self.image_shapes[camera_type] = (
                self.image_shapes[camera_type][0],
                self.image_shapes[camera_type][1],
                n_channels,  # number of channels
            )

        # We reshape each channel and then stack the result
        result = []
        for channel in range(n_channels):
            vector = pixels[:, channel]
            image_2d = (vector.T @ map_tab).reshape(
                self.image_shapes[camera_type][0], self.image_shapes[camera_type][1], 1
            )
            result.append(image_2d)
        telescope_image = np.concatenate(result, axis=-1)
        return telescope_image

    def get_indexmatrix(self, camera_type):
        """
        :param camera_type: a string specifying the telescope type.
        :return: a 2D numpy array [img_width,img_length]
        """
        # Check if axial addressing is selected in the image_mapper
        if self.index_matrixes[camera_type] is None:
            raise ValueError(
                "The function get_indexmatrix() can only be called, when 'indexed_conv' is selected in the ImageMapper."
            )
        # Return the index matrix, which has been calculated in 'generate_table()'
        return self.index_matrixes[camera_type]

    def generate_table(self, camera_type):
        # Get relevant parameters
        output_dim = self.image_shapes[camera_type][0]
        num_pixels = self.num_pixels[camera_type]
        # Get telescope pixel positions and padding for the given tel type
        pos = self.pixel_positions[camera_type]
        pad = self.padding[camera_type]
        default_pad = self.default_pad
        map_method = self.mapping_method[camera_type]

        # Creating the hexagonal and the output grid for the conversion methods.
        grid_size_factor = 1
        if map_method == "rebinning":
            grid_size_factor = 10
        hex_grid, table_grid = self.get_grids(pos, camera_type, grid_size_factor)
        # Updating output_dim, since it could be modified in self.get_grid()
        output_dim = self.image_shapes[camera_type][0]

        # Oversampling and nearest interpolation
        if map_method in [
            "oversampling",
            "nearest_interpolation",
            "image_shifting",
            "axial_addressing",
            "indexed_conv",
        ]:
            # Finding the nearest point in the hexagonal grid for each point in the square grid
            tree = spatial.cKDTree(hex_grid)
            nn_index = np.reshape(tree.query(table_grid)[1], (output_dim, output_dim))
            # Store the nn_index array in the index_matrix. Replace virtual pixel indexes with -1.
            if map_method == "indexed_conv":
                index_matrix = nn_index
                index_matrix[index_matrix >= num_pixels] = -1
                index_matrix = np.flip(index_matrix, axis=0)
                self.index_matrixes[camera_type] = index_matrix
            if map_method == "oversampling" and camera_type not in [
                "ASTRICam",
                "CHEC",
                "SCTCam",
            ]:
                pixel_weight = 1 / 4
            else:
                pixel_weight = 1
            mapping_matrix3d = np.zeros(
                (hex_grid.shape[0], output_dim + pad * 2, output_dim + pad * 2),
                dtype=np.float32,
            )
            for y_grid in np.arange(output_dim):
                for x_grid in np.arange(output_dim):
                    mapping_matrix3d[nn_index[y_grid][x_grid]][y_grid + pad][
                        x_grid + pad
                    ] = pixel_weight

        # Rebinning (approximation)
        elif map_method == "rebinning":
            # Finding the nearest point in the hexagonal grid for each point in the square grid
            tree = spatial.cKDTree(hex_grid)
            nn_index = np.reshape(
                tree.query(table_grid)[1],
                (output_dim * grid_size_factor, output_dim * grid_size_factor),
            )

            # Calculating the overlapping area/weights for each square pixel
            mapping_matrix3d = np.zeros(
                (hex_grid.shape[0], output_dim + pad * 2, output_dim + pad * 2),
                dtype=np.float32,
            )
            for y_grid in np.arange(0, output_dim * grid_size_factor, grid_size_factor):
                for x_grid in np.arange(
                    0, output_dim * grid_size_factor, grid_size_factor
                ):
                    counter = Counter(
                        np.reshape(
                            nn_index[
                                y_grid : y_grid + grid_size_factor,
                                x_grid : x_grid + grid_size_factor,
                            ],
                            -1,
                        )
                    )
                    pixel_index = np.array(list(counter.keys()))
                    weights = list(counter.values()) / np.sum(list(counter.values()))
                    for key in np.arange(len(pixel_index)):
                        mapping_matrix3d[pixel_index[key]][
                            int(y_grid / grid_size_factor) + pad
                        ][int(x_grid / grid_size_factor) + pad] = weights[key]

        # Bilinear interpolation
        elif map_method == "bilinear_interpolation":
            # Finding the nearest point in the hexagonal grid for each point in the square grid
            tree = spatial.cKDTree(hex_grid)
            nn_index = np.reshape(tree.query(table_grid)[1], (output_dim, output_dim))

            if camera_type in ["ASTRICam", "CHEC", "SCTCam"]:
                hex_grid_transpose = hex_grid.T
                x_ticks = np.unique(hex_grid_transpose[0]).tolist()
                y_ticks = np.unique(hex_grid_transpose[1]).tolist()

                dict_hex_grid = {tuple(coord): i for i, coord in enumerate(hex_grid)}

                dict_corner_indexes = {}
                dict_corner_points = {}
                for i, x_val in enumerate(x_ticks):
                    for j, y_val in enumerate(y_ticks):
                        if i == len(x_ticks) - 1 and j < len(y_ticks) - 1:
                            square_points = [[x_ticks[i - 1], y_val]]
                            square_points.append([x_ticks[i - 1], y_ticks[j + 1]])
                            square_points.append([x_val, y_val])
                            square_points.append([x_val, y_ticks[j + 1]])
                        elif j == len(y_ticks) - 1 and i < len(x_ticks) - 1:
                            square_points = [[x_val, y_ticks[j - 1]]]
                            square_points.append([x_val, y_val])
                            square_points.append([x_ticks[i + 1], y_ticks[j - 1]])
                            square_points.append([x_ticks[i + 1], y_val])
                        elif i == len(x_ticks) - 1 and j == len(y_ticks) - 1:
                            square_points = [[x_ticks[i - 1], y_ticks[j - 1]]]
                            square_points.append([x_ticks[i - 1], y_val])
                            square_points.append([x_val, y_ticks[j - 1]])
                            square_points.append([x_val, y_val])
                        else:
                            square_points = [[x_val, y_val]]
                            square_points.append([x_val, y_ticks[j + 1]])
                            square_points.append([x_ticks[i + 1], y_val])
                            square_points.append([x_ticks[i + 1], y_ticks[j + 1]])

                        square_points = np.array(square_points)
                        square_indexes = []
                        for k in np.arange(square_points.shape[0]):
                            square_indexes.append(
                                dict_hex_grid[
                                    (square_points[k][0], square_points[k][1])
                                ]
                            )
                        square_indexes = np.array(square_indexes)
                        dict_corner_points[(i, j)] = square_points
                        dict_corner_indexes[(i, j)] = square_indexes

                corner_points = []
                corner_indexes = []  # index in hexgrid
                for i in np.arange(table_grid.shape[0]):
                    x_index = bisect.bisect_left(x_ticks, table_grid[i][0])
                    y_index = bisect.bisect_left(y_ticks, table_grid[i][1])
                    if x_index != 0:
                        x_index = x_index - 1
                    if y_index != 0:
                        y_index = y_index - 1

                    corner_points.append(dict_corner_points[(x_index, y_index)])
                    corner_indexes.append(dict_corner_indexes[(x_index, y_index)])

                corner_points = np.array(corner_points)
                corner_indexes = np.array(corner_indexes)

            else:
                # Drawing Delaunay triangulation on the hex grid
                tri = spatial.Delaunay(hex_grid)

                corner_indexes = tri.simplices[tri.find_simplex(table_grid)]
                corner_points = hex_grid[corner_indexes]

            weights = self.get_weights(corner_points, table_grid)
            weights = np.reshape(weights, (output_dim, output_dim, weights.shape[1]))
            corner_indexes = np.reshape(
                corner_indexes, (output_dim, output_dim, corner_indexes.shape[1])
            )

            mapping_matrix3d = np.zeros(
                (hex_grid.shape[0], output_dim + pad * 2, output_dim + pad * 2),
                dtype=np.float32,
            )
            for i in np.arange(output_dim):
                for j in np.arange(output_dim):
                    for k in np.arange(corner_indexes.shape[2]):
                        mapping_matrix3d[corner_indexes[j][i][k]][j + pad][
                            i + pad
                        ] = weights[j][i][k]

        # Bicubic interpolation
        elif map_method == "bicubic_interpolation":
            # Finding the nearest point in the hexagonal grid for each point in the square grid
            tree = spatial.cKDTree(hex_grid)
            nn_index = np.reshape(tree.query(table_grid)[1], (output_dim, output_dim))

            if camera_type in ["ASTRICam", "CHEC", "SCTCam"]:
                # Drawing four bigger squares (*,+,-,~) around the target point (.)
                # and then calculate the weights.
                #
                #       +____~____+____~
                #       |    |    |    |
                #       *____-____*____-
                #       |    |  . |    |
                #       +____~____+____~
                #       |    |    |    |
                #       *____-____*____-
                #
                hex_grid_transpose = hex_grid.T
                x_ticks = np.unique(hex_grid_transpose[0]).tolist()
                y_ticks = np.unique(hex_grid_transpose[1]).tolist()
                dict_hex_grid = {tuple(coord): i for i, coord in enumerate(hex_grid)}

                dict_corner_indexes = {}
                dict_corner_points = {}
                invalid_x_val = x_ticks[0] - 1
                invalid_y_val = y_ticks[0] - 1
                for i, x_val in enumerate(x_ticks):
                    for j, y_val in enumerate(y_ticks):
                        square_points = []
                        if (
                            i == 0
                            or j == 0
                            or i >= len(x_ticks) - 2
                            or j >= len(y_ticks) - 2
                        ):
                            for k in np.arange(16):
                                square_points.append([invalid_x_val, invalid_y_val])
                        else:
                            # The square marked as '*' in the drawing above.
                            square_points.append([x_ticks[i - 1], y_ticks[j - 1]])
                            square_points.append([x_ticks[i - 1], y_ticks[j + 1]])
                            square_points.append([x_ticks[i + 1], y_ticks[j - 1]])
                            square_points.append([x_ticks[i + 1], y_ticks[j + 1]])
                            # The square marked as '+' in the drawing above.
                            square_points.append([x_ticks[i - 1], y_val])
                            square_points.append([x_ticks[i - 1], y_ticks[j + 2]])
                            square_points.append([x_ticks[i + 1], y_val])
                            square_points.append([x_ticks[i + 1], y_ticks[j + 2]])
                            # The square marked as '-' in the drawing above.
                            square_points.append([x_val, y_ticks[j - 1]])
                            square_points.append([x_val, y_ticks[j + 1]])
                            square_points.append([x_ticks[i + 2], y_ticks[j - 1]])
                            square_points.append([x_ticks[i + 2], y_ticks[j + 1]])
                            # The square marked as '~' in the drawing above.
                            square_points.append([x_val, y_val])
                            square_points.append([x_val, y_ticks[j + 2]])
                            square_points.append([x_ticks[i + 2], y_val])
                            square_points.append([x_ticks[i + 2], y_ticks[j + 2]])

                        square_points = np.array(square_points)
                        square_indexes = []
                        for k in np.arange(square_points.shape[0]):
                            if square_points[k][0] == invalid_x_val:
                                square_indexes.append(-1)
                            else:
                                square_indexes.append(
                                    dict_hex_grid[
                                        (square_points[k][0], square_points[k][1])
                                    ]
                                )
                        square_indexes = np.array(square_indexes)
                        # reshape square_points and square_indexes
                        square_indexes = np.reshape(square_indexes, (4, 4))
                        square_points = np.reshape(
                            square_points, (4, 4, square_points.shape[1])
                        )

                        dict_corner_points[(i, j)] = square_points
                        dict_corner_indexes[(i, j)] = square_indexes

                weights = []
                corner_indexes = []  # index in hexgrid
                for i in np.arange(table_grid.shape[0]):
                    x_index = bisect.bisect_left(x_ticks, table_grid[i][0])
                    y_index = bisect.bisect_left(y_ticks, table_grid[i][1])
                    if x_index != 0:
                        x_index = x_index - 1
                    if y_index != 0:
                        y_index = y_index - 1

                    corner_points = dict_corner_points[(x_index, y_index)]
                    target = table_grid[i]
                    target = np.expand_dims(target, axis=0)
                    weights_temp = []
                    for j in np.arange(0, corner_points.shape[0], 1):
                        if corner_points[j][0][0] == invalid_x_val:
                            w = np.array([0, 0, 0, 0])
                        else:
                            cp = np.expand_dims(corner_points[j], axis=0)
                            w = self.get_weights(cp, target)
                            w = np.squeeze(w, axis=0)
                        weights_temp.append(w)

                    weights_temp = np.array(weights_temp)
                    weights.append(weights_temp)
                    corner_indexes.append(dict_corner_indexes[(x_index, y_index)])

                weights = np.array(weights)
                corner_indexes = np.array(corner_indexes)
                weights = np.reshape(
                    weights,
                    (output_dim, output_dim, weights.shape[1], weights.shape[2]),
                )
                corner_indexes = np.reshape(
                    corner_indexes,
                    (
                        output_dim,
                        output_dim,
                        corner_indexes.shape[1],
                        corner_indexes.shape[2],
                    ),
                )

            else:
                #
                #                 /\        /\
                #                /  \      /  \
                #               /    \    /    \
                #              / 2NN  \  / 2NN  \
                #             /________\/________\
                #            /\        /\        /\
                #           /  \  NN  /  \  NN  /  \
                #          /    \    /    \    /    \
                #         / 2NN  \  /  .   \  /  2NN \
                #        /________\/________\/________\
                #                 /\        /\
                #                /  \  NN  /  \
                #               /    \    /    \
                #              / 2NN  \  / 2NN  \
                #             /________\/________\
                #

                tri = spatial.Delaunay(hex_grid)

                # Get all relevant simplex indices
                simplex_index = tri.find_simplex(table_grid)
                simplex_index_NN = tri.neighbors[simplex_index]
                simplex_index_2NN = tri.neighbors[simplex_index_NN]

                table_simplex = tri.simplices[simplex_index]
                table_simplex_points = hex_grid[table_simplex]

                # NN
                weights_NN = []
                simplexes_NN = []
                for i in np.arange(simplex_index.shape[0]):
                    if -1 in simplex_index_NN[i] or all(
                        ind >= num_pixels for ind in table_simplex[i]
                    ):
                        w = np.array([0, 0, 0])
                        weights_NN.append(w)
                        corner_simplexes_2NN = np.array([-1, -1, -1])
                        simplexes_NN.append(corner_simplexes_2NN)
                    else:
                        corner_points_NN, corner_simplexes_NN = self.get_triangle(
                            tri, hex_grid, simplex_index_NN[i], table_simplex[i]
                        )
                        target = table_grid[i]
                        target = np.expand_dims(target, axis=0)
                        w = self.get_weights(corner_points_NN, target)
                        w = np.squeeze(w, axis=0)
                        weights_NN.append(w)
                        simplexes_NN.append(corner_simplexes_NN)

                weights_NN = np.array(weights_NN)
                simplexes_NN = np.array(simplexes_NN)

                # 2NN
                weights_2NN = []
                simplexes_2NN = []
                for i in np.arange(3):
                    weights = []
                    simplexes = []
                    for j in np.arange(simplex_index.shape[0]):
                        table_simplex_NN = tri.simplices[simplex_index_NN[j][i]]
                        if (
                            -1 in simplex_index_2NN[j][i]
                            or -1 in simplex_index_NN[j]
                            or all(ind >= num_pixels for ind in table_simplex_NN)
                        ):
                            w = np.array([0, 0, 0])
                            weights.append(w)
                            corner_simplexes_2NN = np.array([-1, -1, -1])
                            simplexes.append(corner_simplexes_2NN)
                        else:
                            corner_points_2NN, corner_simplexes_2NN = self.get_triangle(
                                tri, hex_grid, simplex_index_2NN[j][i], table_simplex_NN
                            )
                            target = table_grid[j]
                            target = np.expand_dims(target, axis=0)
                            w = self.get_weights(corner_points_2NN, target)
                            w = np.squeeze(w, axis=0)
                            weights.append(w)
                            simplexes.append(corner_simplexes_2NN)

                    weights = np.array(weights)
                    simplexes = np.array(simplexes)
                    weights_2NN.append(weights)
                    simplexes_2NN.append(simplexes)

                weights_2NN.append(weights_NN)
                simplexes_2NN.append(simplexes_NN)
                weights_2NN = np.array(weights_2NN)
                simplexes_2NN = np.array(simplexes_2NN)
                weights = np.reshape(
                    weights_2NN,
                    (
                        weights_2NN.shape[0],
                        output_dim,
                        output_dim,
                        weights_2NN.shape[2],
                    ),
                )
                corner_indexes = np.reshape(
                    simplexes_2NN,
                    (
                        simplexes_2NN.shape[0],
                        output_dim,
                        output_dim,
                        simplexes_2NN.shape[2],
                    ),
                )

            mapping_matrix3d = np.zeros(
                (hex_grid.shape[0], output_dim + pad * 2, output_dim + pad * 2),
                dtype=np.float32,
            )
            for i in np.arange(4):
                for j in np.arange(output_dim):
                    for k in np.arange(output_dim):
                        for l in np.arange(weights.shape[3]):
                            if weights.shape[3] == 3:
                                mapping_matrix3d[corner_indexes[i][k][j][l]][k + pad][
                                    j + pad
                                ] = (weights[i][k][j][l] / 4)
                            elif weights.shape[3] == 4:
                                mapping_matrix3d[corner_indexes[k][j][i][l]][k + pad][
                                    j + pad
                                ] = (weights[k][j][i][l] / 4)

        # Cutting the mapping table after num_pixels, since the virtual pixels have intensity zero.
        mapping_matrix3d = mapping_matrix3d[:num_pixels]
        # Mask interpolation
        if self.mask and map_method in [
            "bilinear_interpolation",
            "bicubic_interpolation",
        ]:
            mapping_matrix3d = self.apply_mask_interpolation(
                mapping_matrix3d, nn_index, num_pixels, pad
            )
        # Normalization (approximation) of the mapping table
        if map_method in [
            "rebinning",
            "nearest_interpolation",
            "bilinear_interpolation",
            "bicubic_interpolation",
        ]:
            mapping_matrix3d = self.normalize_mapping_matrix(
                mapping_matrix3d, num_pixels
            )

        if (pad + default_pad) != 0:
            if map_method != "oversampling" or camera_type in [
                "ASTRICam",
                "CHEC",
                "SCTCam",
            ]:
                map_mat = np.zeros(
                    (
                        mapping_matrix3d.shape[0],
                        output_dim + (pad - default_pad) * 2,
                        output_dim + (pad - default_pad) * 2,
                    ),
                    dtype=np.float32,
                )
                for i in np.arange(mapping_matrix3d.shape[0]):
                    map_mat[i] = mapping_matrix3d[i][
                        default_pad : output_dim + pad * 2 - default_pad,
                        default_pad : output_dim + pad * 2 - default_pad,
                    ]
                self.image_shapes[camera_type] = (
                    self.image_shapes[camera_type][0] + (pad - default_pad) * 2,
                    self.image_shapes[camera_type][1] + (pad - default_pad) * 2,
                    self.image_shapes[camera_type][2],
                )
            else:
                map_mat = np.zeros(
                    (
                        mapping_matrix3d.shape[0],
                        output_dim + pad * 2 - default_pad * 4,
                        output_dim + pad * 2 - default_pad * 4,
                    ),
                    dtype=np.float32,
                )
                for i in np.arange(mapping_matrix3d.shape[0]):
                    map_mat[i] = mapping_matrix3d[i][
                        default_pad * 2 : output_dim + pad * 2 - default_pad * 2,
                        default_pad * 2 : output_dim + pad * 2 - default_pad * 2,
                    ]
                self.image_shapes[camera_type] = (
                    self.image_shapes[camera_type][0] + pad * 2 - default_pad * 4,
                    self.image_shapes[camera_type][1] + pad * 2 - default_pad * 4,
                    self.image_shapes[camera_type][2],
                )
        else:
            map_mat = mapping_matrix3d

        # Applying a flip to all mapping tables that the image indexing starts from the top left corner.
        for i in np.arange(map_mat.shape[0]):
            map_mat[i] = np.flip(map_mat[i], axis=0)

        sparse_map_mat = csr_matrix(
            map_mat.reshape(
                map_mat.shape[0],
                self.image_shapes[camera_type][0] * self.image_shapes[camera_type][1],
            ),
            dtype=np.float32,
        )

        return sparse_map_mat

    def get_triangle(self, tri, hex_grid, simplex_index_NN, table_simplex):
        """
        :param tri: a Delaunay triangulation.
        :param hex_grid: a 2D numpy array (hexagonal grid).
        :param simplex_index_NN: a numpy array containing the indexes of the three neighboring simplexes.
        :param table_simplex: a numpy array containing the three indexes (hexaganol grid) of the target simplex.
        :return: two numpy array containing the three corner points and simplexes.
        """
        # This function is calculating the corner points (marked as 'X') and simplexes
        # for the nearest neighbor (NN) triangles. The function returns a bigger triangle,
        # which contains four Delaunay triangles.
        #
        #            X--------------------X
        #             \        /\        /
        #              \  NN  /  \  NN  /
        #               \    /    \    /
        #                \  /  .   \  /
        #                 \/________\/
        #                  \        /
        #                   \  NN  /
        #                    \    /
        #                     \  /
        #                      X
        #

        corner_points = []
        corner_simplexes = []
        for neighbors in np.arange(3):
            table_simplex_NN = tri.simplices[simplex_index_NN[neighbors]]
            simplex = np.array(list(set(table_simplex_NN) - set(table_simplex)))
            simplex = np.squeeze(simplex, axis=0)
            corner_simplexes.append(simplex)
            corner_points.append(hex_grid[simplex])
        corner_points = np.array(corner_points)
        corner_simplexes = np.array(corner_simplexes)
        corner_points = np.expand_dims(corner_points, axis=0)
        return corner_points, corner_simplexes

    def get_weights(self, p, target):
        """
        :param p: a numpy array of shape (i,3 or 4,2) for three or four 2D points (one triangual or square). The index i means that one can calculate the weights for multiply trianguals or squares with one function call.
        :param target: a numpy array of shape (i,2) for one target 2D point.
        :return: a numpy array of shape (i,3) containing the three or four weights.
        """
        weights = []
        if p.shape[1] == 3:
            #
            #       Barycentric coordinates:
            #                 (x3,y3)
            #                   .
            #                  / \
            #                 /   \
            #                /     \
            #               /       \
            #              /         \
            #             /        .  \
            #            /       (x,y) \
            #    (x1,y1)._______________.(x2,y2)
            #
            #       x = w1*x1 + w2*x2 + w3*x3
            #       y = w1*y1 + w2*y2 + w3*y3
            #       1 = w1 + w2 + w3
            #
            #       -> Rearranging:
            #              (y2-y3)*(x-x3)+(x3-x2)*(y-y3)
            #       w1 = ---------------------------------
            #             (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
            #
            #              (y3-y1)*(x-x3)+(x1-x3)*(y-y3)
            #       w2 = ---------------------------------
            #             (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
            #
            #       w3 = 1 - w1 - w2
            #
            for i in np.arange(p.shape[0]):
                w = [0, 0, 0]
                divisor = float(
                    (
                        (p[i][1][1] - p[i][2][1]) * (p[i][0][0] - p[i][2][0])
                        + (p[i][2][0] - p[i][1][0]) * (p[i][0][1] - p[i][2][1])
                    )
                )
                w[0] = (
                    float(
                        (
                            (p[i][1][1] - p[i][2][1]) * (target[i][0] - p[i][2][0])
                            + (p[i][2][0] - p[i][1][0]) * (target[i][1] - p[i][2][1])
                        )
                    )
                    / divisor
                )
                w[1] = (
                    float(
                        (
                            (p[i][2][1] - p[i][0][1]) * (target[i][0] - p[i][2][0])
                            + (p[i][0][0] - p[i][2][0]) * (target[i][1] - p[i][2][1])
                        )
                    )
                    / divisor
                )
                w[2] = 1 - w[0] - w[1]
                weights.append(w)

        elif p.shape[1] == 4:
            #
            #        (x1,y2)          (x2,y2)
            #         w2._______________.w4
            #           |               |
            #           |               |
            #           |               |
            #           |         .     |
            #           |     (x,y)     |
            #         w1._______________.w3
            #       (x1,y1)             (x2,y1)
            #
            #              (x2-x)*(y2-y)
            #       w1 = -----------------
            #             (x2-x1)*(y2-y1)
            #
            #              (x2-x)*(y-y1)
            #       w2 = -----------------
            #             (x2-x1)*(y2-y1)
            #
            #              (x-x1)*(y2-y)
            #       w3 = -----------------
            #             (x2-x1)*(y2-y1)
            #
            #              (x-x1)*(y-y1)
            #       w4 = -----------------
            #             (x2-x1)*(y2-y1)
            #
            for i in np.arange(p.shape[0]):
                w = [0, 0, 0, 0]
                divisor = float((p[i][3][0] - p[i][0][0]) * (p[i][3][1] - p[i][0][1]))
                w[0] = (
                    float((p[i][3][0] - target[i][0]) * (p[i][3][1] - target[i][1]))
                    / divisor
                )
                w[1] = (
                    float((p[i][3][0] - target[i][0]) * (target[i][1] - p[i][0][1]))
                    / divisor
                )
                w[2] = (
                    float((target[i][0] - p[i][0][0]) * (p[i][3][1] - target[i][1]))
                    / divisor
                )
                w[3] = (
                    float((target[i][0] - p[i][0][0]) * (target[i][1] - p[i][0][1]))
                    / divisor
                )
                weights.append(w)

        return np.array(weights, dtype=np.float32)

    def get_grids(self, pos, camera_type, grid_size_factor):
        """
        :param pos: a 2D numpy array of pixel positions, which were taken from the CTApipe.
        :param camera_type: a string specifying the camera type
        :param grid_size_factor: a number specifying the grid size of the output grid. Only if 'rebinning' is selected, this factor differs from 1.
        :return: two 2D numpy arrays (hexagonal grid and squared output grid)
        """

        # Get relevant parameters
        output_dim = self.image_shapes[camera_type][0]
        default_pad = self.default_pad
        map_method = self.mapping_method[camera_type]

        x = np.around(pos[0], decimals=3)
        y = np.around(pos[1], decimals=3)

        x_ticks = np.unique(x).tolist()
        y_ticks = np.unique(y).tolist()

        if camera_type in ["CHEC", "ASTRICam", "SCTCam"]:

            if camera_type == "CHEC":
                # The algorithm doesn't work with the CHEC camera. Additional smoothing
                # for the 'x_ticks' and 'y_ticks' array for CHEC pixel positions.
                num_x_ticks = len(x_ticks)
                remove_x_val = []
                change_x_val = []
                for i in np.arange(num_x_ticks - 1):
                    if np.abs(x_ticks[i] - x_ticks[i + 1]) <= 0.002:
                        remove_x_val.append(x_ticks[i])
                        change_x_val.append(x_ticks[i + 1])
                for j in np.arange(len(remove_x_val)):
                    x_ticks.remove(remove_x_val[j])
                    for k in np.arange(len(x)):
                        if x[k] == remove_x_val[j]:
                            x[k] = change_x_val[j]

                num_y_ticks = len(y_ticks)
                remove_y_val = []
                change_y_val = []
                for i in np.arange(num_y_ticks - 1):
                    if np.abs(y_ticks[i] - y_ticks[i + 1]) <= 0.002:
                        remove_y_val.append(y_ticks[i])
                        change_y_val.append(y_ticks[i + 1])

                for j in np.arange(len(remove_y_val)):
                    y_ticks.remove(remove_y_val[j])
                    for k in np.arange(len(y)):
                        if y[k] == remove_y_val[j]:
                            y[k] = change_y_val[j]

            x_dist = np.around(abs(x_ticks[0] - x_ticks[1]), decimals=3)
            y_dist = np.around(abs(y_ticks[0] - y_ticks[1]), decimals=3)
            for i in np.arange(default_pad):
                x_ticks.append(np.around(x_ticks[-1] + x_dist, decimals=3))
                x_ticks.insert(0, np.around(x_ticks[0] - x_dist, decimals=3))
                y_ticks.append(np.around(y_ticks[-1] + y_dist, decimals=3))
                y_ticks.insert(0, np.around(y_ticks[0] - y_dist, decimals=3))

            virtual_pixels = self.get_virtual_pixels(x_ticks, y_ticks, x, y)
            x = np.concatenate((x, virtual_pixels[:, 0]))
            y = np.concatenate((y, virtual_pixels[:, 1]))
            hex_grid = np.column_stack([x, y])

            xx = np.linspace(
                np.min(x), np.max(x), num=output_dim * grid_size_factor, endpoint=True
            )
            yy = np.linspace(
                np.min(y), np.max(y), num=output_dim * grid_size_factor, endpoint=True
            )
            x_grid, y_grid = np.meshgrid(xx, yy)
            x_grid = np.reshape(x_grid, -1)
            y_grid = np.reshape(y_grid, -1)
            output_grid = np.column_stack([x_grid, y_grid])

        else:
            if len(x_ticks) < len(y_ticks):
                first_ticks = x_ticks
                first_pos = x
                second_ticks = y_ticks
                second_pos = y
            else:
                first_ticks = y_ticks
                first_pos = y
                second_ticks = x_ticks
                second_pos = x

            dist_first = np.around(abs(first_ticks[0] - first_ticks[1]), decimals=3)
            dist_second = np.around(abs(second_ticks[0] - second_ticks[1]), decimals=3)

            if map_method in ["oversampling", "image_shifting"]:
                tick_diff = len(first_ticks) * 2 - len(second_ticks)
                tick_diff_each_side = np.array(int(tick_diff / 2))
            else:
                tick_diff = 0
                tick_diff_each_side = 0
            for i in np.arange(tick_diff_each_side + default_pad * 2):
                second_ticks.append(
                    np.around(second_ticks[-1] + dist_second, decimals=3)
                )
                second_ticks.insert(
                    0, np.around(second_ticks[0] - dist_second, decimals=3)
                )
            for i in np.arange(default_pad):
                first_ticks.append(np.around(first_ticks[-1] + dist_first, decimals=3))
                first_ticks.insert(
                    0, np.around(first_ticks[0] - dist_first, decimals=3)
                )

            if tick_diff % 2 != 0:
                second_ticks.insert(
                    0, np.around(second_ticks[0] - dist_second, decimals=3)
                )

            # Create the virtual pixels outside of the camera
            if map_method not in ["axial_addressing", "indexed_conv"]:
                virtual_pixels = []
                for i in np.arange(2):
                    vp1 = self.get_virtual_pixels(
                        first_ticks[i::2], second_ticks[0::2], first_pos, second_pos
                    )
                    vp2 = self.get_virtual_pixels(
                        first_ticks[i::2], second_ticks[1::2], first_pos, second_pos
                    )
                    virtual_pixels.append(vp1) if vp1.shape[0] < vp2.shape[
                        0
                    ] else virtual_pixels.append(vp2)

                virtual_pixels = np.concatenate(virtual_pixels)

                first_pos = np.concatenate((first_pos, np.array(virtual_pixels[:, 0])))
                second_pos = np.concatenate(
                    (second_pos, np.array(virtual_pixels[:, 1]))
                )

            if map_method == "oversampling":
                grid_first = []
                for i in first_ticks:
                    grid_first.append(i - dist_first / 4.0)
                    grid_first.append(i + dist_first / 4.0)
                grid_second = []
                for j in second_ticks:
                    grid_second.append(j + dist_second / 2.0)

            elif map_method == "image_shifting":
                for i in np.arange(len(second_pos)):
                    if second_pos[i] in second_ticks[::2]:
                        second_pos[i] = second_ticks[
                            second_ticks.index(second_pos[i]) + 1
                        ]

                grid_first = np.unique(first_pos).tolist()
                grid_second = np.unique(second_pos).tolist()
                self.image_shapes[camera_type] = (
                    len(grid_first),
                    len(grid_second),
                    self.image_shapes[camera_type][2],
                )

            elif map_method in ["axial_addressing", "indexed_conv"]:
                virtual_pixels = []
                # manipulate y ticks with extra ticks
                num_extra_ticks = len(y_ticks)
                for i in np.arange(num_extra_ticks):
                    second_ticks.append(
                        np.around(second_ticks[-1] + dist_second, decimals=3)
                    )
                first_ticks = reversed(first_ticks)
                for shift, ticks in enumerate(first_ticks):
                    for i in np.arange(len(second_pos)):
                        if first_pos[i] == ticks and second_pos[i] in second_ticks:
                            second_pos[i] = second_ticks[
                                second_ticks.index(second_pos[i]) + shift
                            ]

                grid_first = np.unique(first_pos).tolist()
                grid_second = np.unique(second_pos).tolist()

                # Squaring the output image if grid axes have not the same length.
                if len(grid_first) > len(grid_second):
                    for i in np.arange(len(grid_first) - len(grid_second)):
                        grid_second.append(
                            np.around(grid_second[-1] + dist_second, decimals=3)
                        )
                elif len(grid_first) < len(grid_second):
                    for i in np.arange(len(grid_second) - len(grid_first)):
                        grid_first.append(
                            np.around(grid_first[-1] + dist_first, decimals=3)
                        )

                # Creating the virtual pixels outside of the camera.
                virtual_pixels.append(
                    self.get_virtual_pixels(
                        grid_first, grid_second, first_pos, second_pos
                    )
                )
                virtual_pixels = np.concatenate(virtual_pixels)

                first_pos = np.concatenate((first_pos, np.array(virtual_pixels[:, 0])))
                second_pos = np.concatenate(
                    (second_pos, np.array(virtual_pixels[:, 1]))
                )
                self.image_shapes[camera_type] = (
                    len(grid_first),
                    len(grid_second),
                    self.image_shapes[camera_type][2],
                )

            else:
                # Add corner
                minimum = min([np.min(first_pos), np.min(second_pos)])
                maximum = max([np.max(first_pos), np.max(second_pos)])

                first_pos = np.concatenate((first_pos, [minimum]))
                second_pos = np.concatenate((second_pos, [minimum]))
                first_pos = np.concatenate((first_pos, [minimum]))
                second_pos = np.concatenate((second_pos, [maximum]))
                first_pos = np.concatenate((first_pos, [maximum]))
                second_pos = np.concatenate((second_pos, [minimum]))
                first_pos = np.concatenate((first_pos, [maximum]))
                second_pos = np.concatenate((second_pos, [maximum]))

                grid_first = grid_second = np.linspace(
                    minimum, maximum, num=output_dim * grid_size_factor, endpoint=True
                )

            if len(x_ticks) < len(y_ticks):
                hex_grid = np.column_stack([first_pos, second_pos])
                x_grid, y_grid = np.meshgrid(grid_first, grid_second)
            else:
                hex_grid = np.column_stack([second_pos, first_pos])
                x_grid, y_grid = np.meshgrid(grid_second, grid_first)
            x_grid = np.reshape(x_grid, -1)
            y_grid = np.reshape(y_grid, -1)
            output_grid = np.column_stack([x_grid, y_grid])

        return hex_grid, output_grid

    @staticmethod
    def get_virtual_pixels(x_ticks, y_ticks, x, y):
        gridpoints = np.array(np.meshgrid(x_ticks, y_ticks)).T.reshape(-1, 2)
        gridpoints = [tuple(l) for l in gridpoints.tolist()]
        virtual_pixels = set(gridpoints) - set(zip(x, y))
        virtual_pixels = np.array(list(virtual_pixels))
        return virtual_pixels

    @staticmethod
    def normalize_mapping_matrix(mapping_matrix3d, num_pixels):
        norm_factor = np.sum(mapping_matrix3d) / float(num_pixels)
        mapping_matrix3d /= norm_factor
        return mapping_matrix3d

    @staticmethod
    def apply_mask_interpolation(mapping_matrix3d, nn_index, num_pixels, pad):
        mask = np.zeros(
            (nn_index.shape[0] + pad * 2, nn_index.shape[1] + pad * 2), dtype=np.float32
        )
        for i in range(nn_index.shape[0]):
            for j in range(nn_index.shape[1]):
                if nn_index[j][i] < num_pixels:
                    mask[j + pad][i + pad] = 1.0
        for i in range(1, mapping_matrix3d.shape[0]):
            mapping_matrix3d[i] *= mask
        return mapping_matrix3d
