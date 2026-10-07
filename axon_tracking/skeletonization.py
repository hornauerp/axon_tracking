import os
import warnings

import kimimaro
import numpy as np
import osteoid
import scipy.ndimage as nd
import scipy.stats as stats
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from skimage.feature import peak_local_max
from skimage.morphology import ball

from axon_tracking import utils as ut


def full_skeletonization(root_path, template_id, params, skel_params, qc_params):
    """
    Runs the full skeletonization pipeline on a single template: preprocessing, thresholding,
    skeletonization and two rounds of path QC.

    Args:
        root_path (str): Folder containing the template files (<template_id>.npy).
        template_id (int): Unit/template ID.
        params (dict): Preprocessing/thresholding parameters (see notebooks/part3_skeletonization.ipynb).
            If params["max_noise_level"] is set, templates whose mean noise (MAD) exceeds it are skipped.
        skel_params (dict): Parameters passed to `skeletonize`.
        qc_params (dict): Parameters passed to `perform_path_qc`.

    Returns:
        qc_skeleton (osteoid.Skeleton): Skeleton after QC (coordinates in um, z in samples), or [] if skipped.
        scaled_qc_list (list): Paths that passed the QC, or [] if skipped.
    """
    template, _ = load_template_file(root_path, template_id)
    interp_temp, interp_noise, ais = preprocess_template(template, params)
    max_noise_level = params.get("max_noise_level")
    if max_noise_level is not None and np.mean(interp_noise) > max_noise_level:
        return [], []

    th_template = threshold_template(interp_temp, interp_noise, ais, params)
    skels = skeletonize(th_template.astype("bool"), **skel_params)
    if 1 not in skels:
        return [], []
    skeleton = kimimaro.join_close_components(skels[1], radius=5)
    skeleton = kimimaro.postprocess(skeleton, tick_threshold=5, dust_threshold=10)

    all_branches = branches_from_paths(skeleton)
    scaled_qc_list, _, _, _ = perform_path_qc(all_branches, params, **qc_params)
    if len(scaled_qc_list) == 0:
        return [], []

    qc_skeleton = paths_to_skeleton(scaled_qc_list)
    qc_skeleton = kimimaro.postprocess(
        qc_skeleton, dust_threshold=10, tick_threshold=10
    )
    qc_skeleton = kimimaro.join_close_components(qc_skeleton, radius=200)
    qc_skeleton = kimimaro.postprocess(qc_skeleton, dust_threshold=0, tick_threshold=0)

    all_branches = branches_from_paths(qc_skeleton)
    scaled_qc_list, _, _, _ = perform_path_qc(all_branches, params, **qc_params)

    return qc_skeleton, scaled_qc_list


def load_template_file(root_path, template_id):
    """
    Load a template file.
    This function loads a template file from the specified root path and template ID.
    The template file is expected to be in NumPy (.npy) format.
    Args:
        root_path (str): The root directory path where the template file is located.
        template_id (int): The identifier for the template file. The file is expected
                            to be named as '<template_id>.npy'.
    Returns:
        tuple: A tuple containing:
            - template (numpy.ndarray): The loaded template as a NumPy array of type float64.
            - template_save_file (str): The full path to the loaded template file.
    """

    template_save_file = os.path.join(root_path, str(template_id) + ".npy")
    template = np.load(template_save_file).astype("float64")

    return template, template_save_file


def preprocess_template(template, params):
    """
    Preprocess a template by localizing the AIS and thresholding the template.
    This function preprocesses a given template by localizing the AIS and thresholding
    the template based on the absolute signal amplitude, the maximum velocity, and the
    noise of the electrode.
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        params (dict): A dictionary containing the parameters for preprocessing.
    Returns:
        numpy.ndarray: The interpolated template.
        numpy.ndarray: The noise matrix.
        numpy.ndarray: The AIS coordinates (empty if params["ais_detection"] is None).
    """
    # Calculate the noise matrix based on the template
    noise = generate_noise_matrix(template, mode="mad")
    # Large median filter to remove noise
    med_filt = nd.median_filter(template, footprint=ball(2))
    # First derivative to eliminate slow drift
    temp_diff = np.diff(med_filt)
    # Median filter to remove noise from the derivative
    tmp_filt = nd.median_filter(temp_diff, footprint=ball(1))

    # Interpolate the template in x, y, and z (time)
    interp_temp = interpolate_template(tmp_filt, spacing=params["upsample"])

    # Localize neurons based on the derivative
    if params["ais_detection"] is not None:
        interp_temp, ais = localize_ais(interp_temp, params)
    else:
        ais = np.array([])

    # Generate a noise matrix based on the template
    interp_noise = interpolate_template(noise, spacing=params["upsample"])

    return interp_temp, interp_noise, ais


def localize_ais(input_mat, params):
    """Localize the AIS in a given template.
    This function localizes AIS in a given input matrix using the `peak_local_max`
    function from the `skimage` library to detect the peaks.
    Args:
        input_mat (numpy.ndarray): The input matrix as a NumPy array.
        params (dict): A dictionary containing the parameters for localization.
            ais_detection (str): "dev" (peak closest to the expected peak time params["ms_cutout"][0]),
                "time" (first peak) or "amp" (highest peak).
            buffer_frames (int): Frames kept before the AIS peak.
    Returns:
        numpy.ndarray: The capped matrix containing the localized neurons.
        numpy.ndarray: The coordinates of the AIS peak.
    """
    peak_idx = params["ms_cutout"][0] * params["sampling_rate"] / 1000
    # Peaks on the spatial border are kept (neurons at the edge of the array), peaks at the
    # first/last frames are not
    local_max = peak_local_max(
        np.abs(input_mat), min_distance=3, threshold_rel=0.1, num_peaks=10, exclude_border=(0, 0, 3)
    )
    if len(local_max) == 0:
        raise ValueError("No peak found to localize the AIS")

    if (
        params["ais_detection"] == "dev"
    ):  # Search for the minimum deviation from expected peak time
        ais = local_max[np.argmin(np.abs(local_max[:, 2] - peak_idx)), :]
    elif params["ais_detection"] == "time":  # Search for first peak
        ais = local_max[np.argmin(local_max[:, 2]), :]
    elif params["ais_detection"] == "amp":  # Search for highest peak
        ais = local_max[0, :]
    else:
        raise ValueError("Invalid search mode")

    # Cap the matrix at the AIS peak time and add a buffer
    ais_buffer = np.min([ais[2], params["buffer_frames"]])  # Prevent negative indices
    ais_peak = int(ais[2] - ais_buffer)
    capped_matrix = input_mat[:, :, ais_peak:]
    ais[2] = ais_buffer
    return capped_matrix, ais


def generate_noise_matrix(template, mode="mad"):
    """Generate a noise matrix based on the template.
    This function generates a noise matrix based on the template. The noise matrix is
    calculated as the median absolute deviation (MAD) or standard deviation (SD) of the
    template along the third axis.
    Args:
        template (numpy.ndarray): The template as a NumPy array
        mode (str): The mode to use for noise calculation. Can be either 'mad' for median
                    absolute deviation or 'sd' for standard deviation.
    Returns:
        numpy.ndarray: The generated noise matrix.
    """

    if mode == "mad":
        noise = stats.median_abs_deviation(template, axis=2)
    elif mode == "sd":
        noise = np.std(template, axis=2)
    else:
        raise ValueError("Invalid mode. Must be 'mad' or 'sd'.")

    noise_matrix = noise[:, :, np.newaxis]  # For compatibility with the template shape

    return noise_matrix


def threshold_template(template, noise, target_coor, params):
    """Generate a thresholded template based on the absolute signal amplitude, the
    maximum velocity, and the noise of the electrode (taken from noise matrix).
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        noise (numpy.ndarray): The noise matrix as a NumPy array.
        target_coor (list): The target coordinates for the template.
        params (dict): A dictionary containing the parameters for thresholding.
    Returns:
        numpy.ndarray: The boolean thresholded template.
    """

    if params["noise_threshold"] is not None:
        noise_th = template < (
            params["noise_threshold"] * noise[:, :, : template.shape[2]]
        )
    else:
        noise_th = np.full_like(template, True, dtype=bool)
    abs_th = template < params["abs_threshold"]
    velocity_th = valid_latency_map(template, target_coor, params)
    th_template = noise_th & abs_th & velocity_th
    return th_template


def valid_latency_map(template, start, params):
    """Generate a boolean matrix indicating the valid latency map based on the maximum
    velocity.
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        start (list): The starting coordinates (AIS) for velocity calculations.
        params (dict): A dictionary containing the parameters for thresholding.
    Returns:
        numpy.ndarray: The boolean matrix indicating the valid latency map.
    """
    n_x, n_y, n_z = template.shape
    x = np.arange(n_x)[:, None, None]
    y = np.arange(n_y)[None, :, None]
    z = np.arange(n_z)[None, None, :]

    # Distance from the start point in m, shape (x, y, 1)
    distances = (
        np.sqrt(
            ((x - start[0]) * params["el_spacing"]) ** 2
            + ((y - start[1]) * params["el_spacing"]) ** 2
        )
        / 1000000
    )
    # Delay from the start point in s, shape (1, 1, z); non-zero at the starting point
    delay = np.abs(z - start[2]) / params["sampling_rate"]
    delay[:, :, start[2]] = 1 / params["sampling_rate"]

    # Check which velocity is within the maximum velocity
    passed = (distances / delay) < params["max_velocity"]  # Boolean matrix
    return passed


def restore_sparse_template(template, spacing=(0.5, 0.5, 0.5)):
    """Remove empty electrodes and interpolate the sparse template to fill in the gaps.
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        spacing (tuple): The spacing for interpolation.
    Returns:
        numpy.ndarray: The interpolated template.
    """

    # Find where the actual template begins (first non-zero electrode)
    x, y, _ = np.nonzero(template)
    # Restrict template to the smallest bounding box
    true_template = template[np.min(x) :, np.min(y) :, :]
    # Remove empty electrodes (due to sparseness)
    del_x = np.delete(
        true_template, np.nonzero(np.sum(true_template, axis=(0, 2)) == 0), axis=1
    )
    del_y = np.delete(del_x, np.nonzero(np.sum(del_x, axis=(1, 2)) == 0), axis=0)
    # Interpolate the template to fill in the gaps
    interp_tmp = interpolate_template(del_y, spacing)
    return interp_tmp


def interp_max(x, spacing):
    """Calculate the maximum value for interpolation.
    Args:
        x (numpy.ndarray): The input array.
        spacing (int): The spacing for interpolation.
    Returns:
        int: The maximum value for interpolation.
    """

    if len(x) == 1:
        interp_max = spacing
    elif spacing == 1:
        interp_max = x[-1] + 1
    else:
        interp_max = x[-1]
    return interp_max


def interpolate_template(template, spacing=(1, 1, 0.2), template_path=None, overwrite=False):
    """Generate an interpolated template based on the input template and spacing.
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        spacing (list): The spacing for interpolation.
        template_path (str): The path to save the interpolated file to (optional).
        overwrite (bool): Whether to overwrite the existing interpolated file.
    Returns:
        numpy.ndarray: The interpolated template.
    """

    if template_path:
        split_path = template_path.split(sep="/")
        split_path[-1] = "interp_" + split_path[-1]
        interp_tmp_path = "/".join(split_path)
    else:
        interp_tmp_path = ""

    if os.path.exists(interp_tmp_path) and not overwrite:
        interp_template = np.load(interp_tmp_path).astype("float64")

    else:
        x, y, z = [np.arange(template.shape[k]) for k in range(3)]
        f = RegularGridInterpolator((x, y, z), template)
        new_grid = np.mgrid[
            0 : interp_max(x, spacing[0]) : spacing[0],
            0 : interp_max(y, spacing[1]) : spacing[1],
            0 : interp_max(z, spacing[2]) : spacing[2],
        ]

        new_grid = np.moveaxis(
            new_grid, (0, 1, 2, 3), (3, 0, 1, 2)
        )  # reorder axes for evaluation
        interp_template = f(new_grid)

        if template_path:
            np.save(interp_tmp_path, interp_template.astype("float32"))

    return interp_template


def cone(matrix_shape, r, apex=None):
    if apex is None:
        apex = np.empty(3)
        apex[0], apex[1], apex[2] = (
            np.floor(matrix_shape[0] / 2),
            np.floor(matrix_shape[1] / 2),
            0,
        )

    x, y, z = np.ogrid[: matrix_shape[0], : matrix_shape[1], : matrix_shape[2]]
    cone_equation = (x - apex[0]) ** 2 + (y - apex[1]) ** 2 <= r**2 * (
        1 - (z - apex[2]) / matrix_shape[2]
    ) ** 2
    cone_matrix = np.zeros(matrix_shape, dtype=bool)
    cone_matrix[cone_equation] = True
    return cone_matrix


def generate_dilation_structure(max_t, params):
    """Generate a cone-shaped dilation structure based on the maximum time and parameters.
    Args:
        max_t (int): The maximum time.
        params (dict): A dictionary containing the parameters for dilation.
    Returns:
        numpy.ndarray: The generated dilation structure.

    """
    frame_time = 1000000 / params["sampling_rate"]  # in us
    t = np.ceil(max_t / frame_time).astype("int16")  # in samples
    max_r = (
        params["max_velocity"] * max_t
    )  # Infer the maximum radius from the maximum velocity
    r = np.ceil(max_r / params["el_spacing"]).astype("int16")  # in grid units
    d = (2 * r + 1).astype("int16")  # maximum diameter of the cone

    cone_matrix = cone((d, d, t), r)  # Generate the cone matrix

    structure = cone_matrix[:, :, ::-1]
    structure_base = np.full(
        (structure.shape[0], structure.shape[1], structure.shape[2]), False
    )
    structure_init = np.full((structure.shape[0], structure.shape[1], 1), False)
    structure_init[r, r, 0] = True
    full_structure = np.concatenate(
        (structure_base, structure_init, structure), axis=2
    )  # Ensure that dilation only occurs in the future

    return full_structure


def skeletonize(
    input_matrix,
    scale=2,
    const=50,
    pdrf_exponent=4,
    pdrf_scale=10000,
    dust_threshold=0,
    anisotropy=(1, 1, 1),
    tick_threshold=None,
    n_jobs=16,
    ais=None,
):
    """Skeletonize a thresholded (boolean) template with kimimaro (TEASAR).

    Args:
        input_matrix (numpy.ndarray): Boolean template (x, y, t).
        scale, const, pdrf_exponent, pdrf_scale: TEASAR parameters; `const` is in physical units,
            i.e. voxels scaled by `anisotropy`.
        dust_threshold (int): Skip connected components with fewer voxels.
        anisotropy (tuple): Physical size of a voxel along (x, y, t). With the default (1, 1, 1),
            TEASAR parameters are in voxels.
        tick_threshold: Has no effect here; apply it with kimimaro.postprocess().
        n_jobs (int): Number of parallel processes.
        ais (array-like, optional): AIS voxel (x, y, t), e.g. from `localize_ais`. If it lies inside the
            thresholded template, tracing is started from it (kimimaro extra_targets_before).

    Returns:
        dict: {label: osteoid.Skeleton} with vertices in voxel coordinates.
    """
    if tick_threshold is not None:
        warnings.warn(
            "tick_threshold has no effect in skeletonize(); apply it with kimimaro.postprocess().",
            stacklevel=2,
        )

    extra_targets_before = []
    if ais is not None and len(ais) >= 3:
        ais_voxel = tuple(int(a) for a in ais[:3])
        inside = all(0 <= a < s for a, s in zip(ais_voxel, input_matrix.shape))
        if inside and input_matrix[ais_voxel]:
            extra_targets_before = [ais_voxel]

    skels = kimimaro.skeletonize(
        input_matrix,
        teasar_params={
            "scale": scale,
            "const": const,  # physical units
            "pdrf_exponent": pdrf_exponent,
            "pdrf_scale": pdrf_scale,
            "soma_acceptance_threshold": 35,  # physical units
            "soma_detection_threshold": 20,  # physical units
            "soma_invalidation_const": 30,  # physical units
            "soma_invalidation_scale": 2,
        },
        anisotropy=anisotropy,
        dust_threshold=dust_threshold,
        parallel=n_jobs,
        extra_targets_before=extra_targets_before,
    )

    # Downstream code works in voxel coordinates
    return {label: skel.voxel_space() for label, skel in skels.items()}


def perform_path_qc(
    paths,
    params,
    window_size=7,
    max_duplicate_ratio=0.5,
    min_r2=0.9,
    vel_range=(0.4, 1),
    min_length=10,
):
    """Splits paths at changes of their temporal direction and keeps paths with a linear
    distance-time relationship (R2 > min_r2) and a velocity within vel_range [m/s]. Paths too
    short to fit a velocity (<= 3 points) are kept.

    Returns:
        qc_list (list): Paths (in um, float32 as required by kimimaro) that passed the QC.
        r2s, vels, lengths (list): R2 (NaN for short paths), velocity and number of points of each kept path.
    """
    if len(paths) == 0:
        return [], [], [], []

    scaled_paths = ut.convert_coor_scale(paths, params, scale="um")
    good_path_list, r2s, vels, lengths = [], [], [], []
    for path in scaled_paths:
        if len(path) > window_size:
            path, _ = split_path(
                path, window_size=window_size, max_duplicate_ratio=max_duplicate_ratio
            )
            path = [p for p in path if len(p) > min_length]
        if len(path) < 1:
            continue
        if not isinstance(path, list):
            path = [path]
        good_path_list = good_path_list + path

    for path in good_path_list:
        if path.shape[0] > 3:
            vel, r2 = calculate_path_velocity(path, params)
        else:
            vel, r2 = np.nan, np.nan
        r2s.append(r2)
        vels.append(vel)
        lengths.append(len(path))

    r2s, vels = np.array(r2s, dtype=float), np.array(vels, dtype=float)
    good_vel = (
        (r2s > min_r2) & (vels < vel_range[1]) & (vels > vel_range[0])
    ) | np.isnan(r2s)
    keep_idx = np.flatnonzero(good_vel)

    qc_list = [good_path_list[x].astype(np.float32) for x in keep_idx]
    r2s = [r2s[x] for x in keep_idx]
    vels = [vels[x] for x in keep_idx]
    lengths = [lengths[x] for x in keep_idx]
    return qc_list, r2s, vels, lengths


def split_path(path, window_size=7, max_duplicate_ratio=0.3):
    smoothed_data = np.convolve(
        path[:, 2], np.ones(window_size) / window_size, mode="valid"
    )
    gradient = np.gradient(smoothed_data)
    inflection_points = np.where(np.diff(np.sign(gradient)))[0]
    inflection_points = (inflection_points + np.floor(window_size / 2)).astype("int")
    inflection_points = np.delete(
        inflection_points, np.flatnonzero(np.abs(np.diff(inflection_points)) < 2)
    )
    split_points = [0] + list(inflection_points) + [path.shape[0]]
    split_paths = [
        path[split_points[s] : split_points[s + 1], :]
        for s in range(len(split_points) - 1)
    ]
    qc_paths = remove_circulating_paths(
        split_paths, max_duplicate_ratio=max_duplicate_ratio
    )

    return qc_paths, inflection_points


def branches_from_paths(skeleton):
    """Splits a skeleton into its branches, i.e. the paths between adjacent critical points
    (terminals and branch points). Each branch includes both of its end points.

    Args:
        skeleton (osteoid.Skeleton): Skeleton to split.

    Returns:
        list of np.ndarray: Vertex coordinates of each branch (with more than one vertex).
    """
    return [branch for branch in skeleton.interjoint_paths() if branch.shape[0] > 1]


def paths_to_skeleton(paths):
    """Merges paths (arrays of vertex coordinates) into one skeleton. Vertices are cast to float32,
    which kimimaro.postprocess/join_close_components require (e.g. after average_smoothing, which
    returns float64 vertices)."""
    return osteoid.Skeleton.simple_merge(
        [osteoid.Skeleton.from_path(np.asarray(path, dtype=np.float32)) for path in paths]
    )


def remove_circulating_paths(path_list, max_duplicate_ratio=0.3):
    """
    Removes paths that contain too many duplicate x,y coordinates, expressed as ratio.
    """
    indices_to_remove = []
    for p, path in enumerate(path_list):
        points = path[:, :2]
        duplicate_ratio = 1 - (len(np.unique(points, axis=0)) / len(points))

        if duplicate_ratio > max_duplicate_ratio:
            indices_to_remove.append(p)

    clean_path_list = [
        path_list[i] for i in range(len(path_list)) if i not in indices_to_remove
    ]
    return clean_path_list


def calculate_path_velocity(path, params):
    path_diff = np.diff(path, axis=0)
    dist = np.sqrt(path_diff[:, 0] ** 2 + path_diff[:, 1] ** 2) / 1000
    time = (np.cumsum(np.abs(path_diff[:, 2])) / params["sampling_rate"]) * 1000
    regressor = LinearRegression(fit_intercept=True)
    vel_y = np.cumsum(dist).reshape(-1, 1)
    vel_x = time.reshape(-1, 1)
    regressor.fit(vel_x, vel_y)
    y_pred = regressor.predict(vel_x)
    r2 = r2_score(vel_y, y_pred)
    return regressor.coef_[0][0], r2


def path_to_vertices(path_list, params, unscale=True):
    vertices = np.concatenate(path_list)
    sorted_indices = np.argsort(vertices[:, 2])
    sorted_vertices = vertices[sorted_indices, :]
    return sorted_vertices
