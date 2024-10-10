import os, kimimaro, sklearn
import numpy as np
import cloudvolume as cv
import scipy.ndimage as nd
from skimage.morphology import disk, ball
from skimage.feature import peak_local_max
from scipy.interpolate import RegularGridInterpolator
import scipy.stats as stats
from scipy.spatial.distance import pdist, cdist


def full_skeletonization(
    root_path, stream_id, template_id, params, skel_params, qc_params
):
    template, template_save_file, noise = load_template_file(
        root_path, stream_id, template_id
    )
    if np.mean(noise) > params["max_noise_level"]:
        return [], []

    temp_diff = np.diff(template)
    capped_template, target_coor = localize_neurons(
        temp_diff, ms_cutout=params["ms_cutout"]
    )
    tmp_filt = nd.gaussian_filter(capped_template, sigma=1)
    interp_temp = interpolate_template(tmp_filt, spacing=params["upsample"])
    th_template = threshold_template(interp_temp, noise, target_coor, params)

    t_cap = [0, th_template.shape[2]]  # in samples
    skels = skeletonize(
        th_template[:, :, t_cap[0] : t_cap[1]].astype("bool"), **skel_params
    )
    skeleton = kimimaro.join_close_components(skels[1], radius=5)
    skeleton = kimimaro.postprocess(skeleton, tick_threshold=5, dust_threshold=10)

    all_branches = branches_from_paths(skeleton)
    scaled_qc_list, r2s, vels, lengths = perform_path_qc(
        all_branches, params, **qc_params
    )

    qc_skel_list = [cv.Skeleton.from_path(x) for x in scaled_qc_list]
    qc_skeleton = cv.Skeleton.simple_merge(qc_skel_list)
    qc_skeleton = kimimaro.postprocess(
        qc_skeleton, dust_threshold=10, tick_threshold=10
    )
    qc_skeleton = kimimaro.join_close_components(qc_skeleton, radius=200)
    qc_skeleton = kimimaro.postprocess(qc_skeleton, dust_threshold=0, tick_threshold=0)

    all_branches = branches_from_paths(qc_skeleton)

    scaled_qc_list, r2s, full_vels, lengths = perform_path_qc(
        all_branches, params, **qc_params
    )

    return qc_skeleton, scaled_qc_list


def load_template_file(root_path, stream_id, template_id):
    template_save_file = os.path.join(
        root_path, stream_id, "sorter_output", "templates", str(template_id) + ".npy"
    )  #'sorter_output',
    noise_save_file = os.path.join(
        root_path, stream_id, "templates", str(template_id) + "_noise.npy"
    )
    template = np.load(template_save_file).astype("float64")
    if False:  # os.path.exists(noise_save_file):
        noise = np.load(noise_save_file).astype("float64")
    else:
        # print('No noise file found, inferring from template')
        noise = generate_noise_matrix(template)

    return template, template_save_file, noise


def localize_neurons(
    input_mat,
    ms_cutout,
    min_distance=5,
    threshold_rel=0.1,
    num_peaks=3,
    buffer_frames=2,
    ms_peak_cutout=0.5,
):

    # local_max = peak_local_max(np.abs(input_mat), min_distance=min_distance, threshold_rel=threshold_rel, num_peaks=num_peaks)

    # cutout_ratio = (ms_cutout[0]/np.sum(ms_cutout))
    # peak_range = [(cutout_ratio-(ms_peak_cutout/np.sum(ms_cutout))),(cutout_ratio+(ms_peak_cutout/np.sum(ms_cutout)))]
    # peak_range = np.round(np.array(peak_range)*input_mat.shape[2])

    # target_coor = local_max[(local_max[:,2] >= peak_range[0]) & (local_max[:,2] <= peak_range[1]),:].astype("int16")

    # if len(target_coor) > 0:
    target_coor = list(np.unravel_index(np.argmax(-input_mat), input_mat.shape))
    capped_matrix = input_mat[:, :, (target_coor[2] - buffer_frames) :]
    target_coor[2] = buffer_frames
    # else:
    #    capped_matrix = input_mat
    #    target_coor=[[0, 0, 0]]

    # post_coor = local_max[local_max[:,2] > peak_range[1],:].astype("int16")
    # if len(post_coor) > 0: #Check if postsynaptic target was detected
    #    post_coor[0][2] = post_coor[0][2] - target_coor[0][2]

    return capped_matrix, target_coor  # , post_coor


def generate_noise_matrix(template, noise=[], mode="mad"):
    if not noise:
        if mode == "mad":
            noise = stats.median_abs_deviation(template, axis=2)
        elif mode == "sd":
            noise = np.std(template, axis=2)

    noise_matrix = noise[:, :, np.newaxis]

    return noise_matrix


def threshold_template(template, noise, target_coor, params):
    if params["noise_threshold"]:
        noise_th = template < (
            params["noise_threshold"] * noise[:, :, : template.shape[2]]
        )
    else:
        noise_th = np.full_like(template, True)
    abs_th = template < params["abs_threshold"]

    # r = int((((template.shape[2] / params['sampling_rate']) * params['max_velocity']) * 1000000) / 17.5)
    # velocity_th = cone(template.shape, r, apex=tuple(target_coor[0]))
    velocity_th = valid_latency_map(template, target_coor, params)
    th_template = noise_th * abs_th * velocity_th
    return th_template


def interp_max(x, spacing):
    if len(x) == 1:
        interp_max = spacing
    elif spacing == 1:
        interp_max = x[-1] + 1
    else:
        interp_max = x[-1]
    return interp_max


def interpolate_template(
    template, spacing=[1, 1, 0.2], template_path=[], overwrite=False
):
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
        # new_grid = np.mgrid[0:x[-1]:spacing[0], 0:y[-1]:spacing[1], 0:z[-1]+1:spacing[2]]
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


def valid_latency_map(template, start, params):
    indices_array = np.indices(template.shape) * params["el_spacing"]  # convert to (um)
    distances = (
        np.sqrt(
            (indices_array[0] - start[0] * params["el_spacing"]) ** 2
            + (indices_array[1] - start[1] * params["el_spacing"]) ** 2
        )
        / 1000000
    )  # convert to (s)
    th_mat = np.zeros(distances.shape)
    for z in range(distances.shape[2]):
        th_mat[:, :, z] = (params["max_velocity"] / params["sampling_rate"]) * (z + 2)

    passed = distances <= th_mat
    return passed


def cone(matrix_shape, r, apex=[]):
    if not apex:
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


def generate_dilation_structure(max_t, max_r, spacing=1 / 3, sampling_rate=20000):
    """
    max_t: numeric
        Maximum time [us] to detect a peak from a previous peak
    max_r: numeric
        Maximum deviation [um] from the peak within max_t to detect the next peak
    spacing: numeric
        Spacing of the interpolation (if performed before the dilation)
    """
    el_dist = params["el_spacing"]
    frame_time = (1000000 / sampling_rate) * spacing  # Assumes 20k sampling rate
    t = np.ceil(max_t / frame_time).astype("int16")
    r = np.ceil(max_r / el_dist).astype("int16")
    d = (2 * r + 1).astype("int16")

    cone_matrix = cone((d, d, t), r)
    # x, y, z = np.ogrid[:d, :d, :t]
    # cone_equation = (x - (r))**2 + (y - (r))**2 <= r**2 * (1 - (z - 0)/t)**2
    # cone_matrix = np.zeros((d,d,t), dtype=bool)
    # cone_matrix[cone_equation] = True

    structure = cone_matrix[:, :, ::-1]
    structure_base = np.full(
        (structure.shape[0], structure.shape[1], structure.shape[2]), False
    )
    structure_init = np.full((structure.shape[0], structure.shape[1], 1), False)
    structure_init[r, r, 0] = True
    full_structure = np.concatenate((structure_base, structure_init, structure), axis=2)

    return full_structure


def iterative_dilation(
    template,
    r_dilation=2,
    init_th=-10,
    min_th=-1,
    filter_footprint=(3, 3, 3),
    use_derivative=True,
):
    if use_derivative:
        template = np.diff(template)

    structure = generate_dilation_structure(r_dilation)
    m_init = template < init_th  # Detection of initial seeds/definitive peaks
    mask = template < min_th  # Mask indicating potential peak locations
    dilated = nd.binary_dilation(m_init, structure=structure, iterations=0, mask=mask)
    # filtered = nd.median_filter(dilated,size=filter_size)
    # filtered = nd.median_filter(dilated,footprint=ball(1))
    if filter_footprint is not None:
        dilated = nd.median_filter(dilated, footprint=filter_footprint)

    return dilated


def skeletonize(
    input_matrix,
    scale=2,
    const=50,
    pdrf_exponent=4,
    pdrf_scale=10000,
    dust_threshold=0,
    anisotropy=(17.5, 17.5, 50.0),
    tick_threshold=10,
    n_jobs=16,
    ais=[],
):
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
        dust_threshold=dust_threshold,
        parallel=n_jobs,
    )
    # extra_targets_before=ais

    return skels


def perform_path_qc(
    paths,
    params,
    window_size=7,
    max_duplicate_ratio=0.5,
    min_r2=0.9,
    vel_range=[0.4, 1],
    min_length=10,
):
    if np.max(np.concatenate(paths)) < 220:
        scaled_paths = scale_path_coordinates(paths, params)
    else:
        scaled_paths = paths
    good_path_list, r2s, vels, lengths = [], [], [], []
    path_list = []
    for path in scaled_paths:
        if len(path) > window_size:
            path, inflection_points = split_path(
                path, window_size=window_size, max_duplicate_ratio=max_duplicate_ratio
            )
            path = [p for p in path if len(p) > min_length]
        if len(path) < 1:
            continue
        if type(path) is not list:
            path = [path]
        good_path_list = good_path_list + path
    # good_path_list.append(path)
    # print(good_path_list)
    for path in good_path_list:
        if path.shape[0] > 3:
            vel, r2 = calculate_path_velocity(path, params)
        else:
            vel, r2 = np.nan, np.nan
        r2s.append(r2)
        vels.append(vel)
        lengths.append(len(path))

    good_vel = np.where(
        (np.array(r2s) > min_r2)
        & (np.array(vels) < vel_range[1])
        & (np.array(vels) > vel_range[0])
    )
    good_vel = good_vel | np.isnan(r2)
    qc_list = [good_path_list[x] for x in good_vel[0]]
    r2s = [r2s[x] for x in good_vel[0]]
    vels = [vels[x] for x in good_vel[0]]
    lengths = [lengths[x] for x in good_vel[0]]
    return qc_list, r2s, vels, lengths


def split_path(path, window_size=7, max_duplicate_ratio=0.3):
    smoothed_data = np.convolve(
        path[:, 2], np.ones(window_size) / window_size, mode="valid"
    )
    gradient = np.gradient(smoothed_data)
    inflection_points = np.where(np.diff(np.sign(gradient)))[0]
    inflection_points = (inflection_points + np.floor(window_size / 2)).astype("int")
    inflection_points = np.delete(
        inflection_points, [np.where(np.abs(np.diff(inflection_points)) < 2)]
    )
    split_points = [0] + list(inflection_points) + [path.shape[0]]
    # print(split_points)
    split_paths = [
        path[split_points[s] : split_points[s + 1], :]
        for s in range(len(split_points) - 1)
    ]
    qc_paths = remove_circulating_paths(
        split_paths, max_duplicate_ratio=max_duplicate_ratio
    )

    return qc_paths, inflection_points


def branches_from_paths(skeleton):
    branch_points = skeleton.branches()
    path_list = skeleton.paths()
    all_branches = []
    for path in path_list:
        path_idx, _ = np.where(
            (path[:, None] == skeleton.vertices[branch_points]).all(axis=2)
        )
        branch_idx = []
        for p in path_idx:
            # check_idx = range(p-1,p+2)
            dists = pdist(path[p - 1 : p + 2, :])
            branch_idx = [0]
            if dists[0] > dists[2]:
                branch_idx.append(p)
            else:
                branch_idx.append(p + 1)
        if len(branch_idx) > 0:
            branch_idx.append(path.shape[0])
            new_branches = [
                path[branch_idx[x] : branch_idx[x + 1], :]
                for x in range(len(branch_idx) - 1)
            ]
            all_branches = all_branches + new_branches

    unique_branches = []
    for arr in all_branches:
        # Check if the array is not already in unique_arrays
        branch_list = []
        if not any(np.array_equal(arr, unique_arr) for unique_arr in unique_branches):
            # unique_branches.append(arr)
            branch_list.append(arr)
        unique_branches = unique_branches + branch_list
        unique_branches = [x for x in unique_branches if x.shape[0] > 1]
    return unique_branches


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
    # print(len(path_list) - len(clean_path_list))
    return clean_path_list


def calculate_path_velocity(path, params):
    path_diff = np.diff(path, axis=0)
    # print(path_diff.shape)
    dist = np.sqrt(path_diff[:, 0] ** 2 + path_diff[:, 1] ** 2) / 1000
    time = (np.cumsum(np.abs(path_diff[:, 2])) / params["sampling_rate"]) * 1000
    regressor = sklearn.linear_model.LinearRegression(fit_intercept=False)
    vel_y = np.cumsum(dist).reshape(-1, 1)
    vel_x = time.reshape(-1, 1)
    regressor.fit(vel_x, vel_y)
    y_pred = regressor.predict(vel_x)
    r2 = sklearn.metrics.r2_score(vel_y, y_pred)
    return regressor.coef_[0][0], r2


def scale_path_coordinates(path_list, params):
    scaled_paths = [
        np.concatenate((path[:, :2] * params["el_spacing"], path[:, 2:]), axis=1)
        for path in path_list
    ]
    return scaled_paths


def unscale_path_coordinates(path_list, params):
    unscaled_paths = [
        np.concatenate((path[:, :2] / params["el_spacing"], path[:, 2:]), axis=1)
        for path in path_list
    ]
    return unscaled_paths


def path_to_vertices(path_list, params, unscale=True):
    if unscale:
        path_list = unscale_path_coordinates(path_list, params)

    vertices = np.concatenate(path_list)
    sorted_indices = np.argsort(vertices[:, 2])
    sorted_vertices = vertices[sorted_indices, :]
    return sorted_vertices
