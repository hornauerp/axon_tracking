import os

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy.ndimage as nd
from IPython.display import Image, display
from matplotlib.collections import LineCollection
from skimage.morphology import ball

from axon_tracking import skeletonization as skel
from axon_tracking import utils as ut


def plot_velocity_qc(vels, r2s, fig_size=(6, 2)):
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    plt.subplot(121)
    plt.hist(vels, 100, range=(0, 2))
    plt.title("Velocity")

    plt.subplot(122)
    plt.hist(r2s, 100, range=(0, 1))
    plt.title("R2")
    plt.show()


def plot_bare_skeleton(
    path_list, params, save_path=[], figsize=4, linewidth=2, cmap="copper"
):
    sorted_vertices = skel.path_to_vertices(path_list, params)
    c_max = 1000 * (sorted_vertices[-1][2]) / params["sampling_rate"]

    fig, ax = plt.subplots(
        figsize=(22 * figsize, 12 * figsize), constrained_layout=True
    )
    for path in path_list:
        x = path[:, 0]
        y = path[:, 1]
        cols = (path[:, 2]) / params["sampling_rate"] * 1000
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, linewidths=linewidth)
        lc.set_array(cols)
        lc.set_clim((0, np.ceil(c_max)))

        line = ax.add_collection(lc)
    ax.set_xlim([0, 440])
    ax.set_ylim([240, 0])
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True)
        plt.close()
    # plt.show()
    return fig, ax


def plot_delay_skeleton(
    path_list,
    params,
    skel_params,
    figsize=4,
    ais=None,
    plot_ais_connection=True,
    linewidth=2,
    font_size=24,
):

    # path_list = skel.scale_path_coordinates(path_list)
    sorted_vertices = skel.path_to_vertices(path_list, params, unscale=False)
    c_max = 1000 * (sorted_vertices[-1][2]) / params["sampling_rate"]
    fig_ratio = (
        np.ptp(sorted_vertices, axis=0)[0] / np.ptp(sorted_vertices, axis=0)[1] + 0.7
    )

    fig, ax = plt.subplots(
        constrained_layout=True #figsize=(22 * figsize, 12 * figsize), 
    )
    for path in path_list:
        x = path[:, 0]
        y = path[:, 1]
        cols = (path[:, 2]) / params["sampling_rate"] * 1000
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="copper", linewidths=linewidth)
        lc.set_array(cols)
        lc.set_clim((0, np.ceil(c_max)))

        line = ax.add_collection(lc)

    clb = fig.colorbar(
        line, ax=ax, ticks=[0, np.ceil(c_max)], shrink=0.3
    )  # ,format=mticker.FixedFormatter(clb_ticks)
    clb.set_label(label="Delay (ms)", size=font_size * figsize)
    clb.ax.tick_params(labelsize=font_size * figsize, length=0)
    if ais is not None:
        plt.scatter(
            ais[0],
            ais[1],
            s=50,
            color="k",
            zorder=10,
        )

    if plot_ais_connection and ais is not None:
        plt.plot(
            [ais[0], sorted_vertices[0][0]], [ais[1], sorted_vertices[0][1]], c="k"
        )
        print(sorted_vertices[0][0])

    ax.autoscale_view()

    # ax.set_xlim([0, 440])
    # ax.set_ylim([0, 240])
    ax.set_ylim(ax.get_ylim()[::-1])
    # ax.set_xlabel("(μm)")
    # ax.set_ylabel("(μm)")
    # plt.show()
    return fig, ax


def plot_conduction_velocity(path_list, params, fig_size=(3, 3)):

    fig, axes = plt.subplots(figsize=fig_size)
    for path in path_list:
        path_diff = np.diff(path, axis=0)
        dist = np.sqrt(path_diff[:, 0] ** 2 + path_diff[:, 1] ** 2) / 1000
        time = (np.cumsum(np.abs(path_diff[:, 2])) / params["sampling_rate"]) * 1000
        plt.scatter(time, np.cumsum(dist), s=1)
    plt.ylabel("Distance (mm)")
    plt.xlabel("Time (ms)")
    plt.show()


def plot_template_and_noise(template, noise, th_template, vrange=[-3, 3]):
    max_template = ut.generate_max_template(template, peak="both", absolute_value=False)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    plt.subplot(131)
    plt.imshow(max_template.T, vmin=vrange[0], vmax=vrange[1], cmap="coolwarm")
    plt.colorbar(shrink=0.3)

    plt.subplot(132)
    plt.imshow(np.squeeze(noise).T)
    plt.colorbar(shrink=0.3)

    plt.subplot(133)
    plt.imshow(np.sum(th_template, axis=2).T, vmin=0, vmax=20)
    plt.colorbar(shrink=0.3)

    plt.show()


def plot_template_overview(
    root_path,
    n_cols=3,
    vmin=-10,
    vmax=0,
    filename="overview",
    unit_ids=None,
    overwrite=False,
):
    full_filename = os.path.join(root_path, "templates", filename + ".png")
    if os.path.exists(full_filename) and not overwrite:
        display(Image(filename=full_filename))
    else:
        template_folder = os.path.join(root_path, "templates")
        files = os.listdir(template_folder)
        template_files = [
            f for f in files if "_" not in f if "overview" not in f if ".npy" in f
        ]
        ids = [float(t.split(".")[0]) for t in template_files]
        if unit_ids is not None:
            sort_idx = [ids.index(x) for x in unit_ids]
        else:
            sort_idx = np.argsort(ids)
        template_files = [template_files[i] for i in sort_idx]

        n_rows = int(np.ceil(len(template_files) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
        for i, template_file in enumerate(template_files):
            template_path = os.path.join(template_folder, template_file)
            template = np.load(template_path)
            temp_diff = np.diff(template)
            tmp_filt = nd.gaussian_filter(temp_diff, sigma=1)
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(
                np.min(tmp_filt, axis=2).T, vmin=vmin, vmax=vmax, cmap="gist_heat"
            )
            plt.title(template_file)

        plt.savefig(full_filename)


def plot_waveforms(template):
    flat_tmp = np.reshape(template.transpose((2, 0, 1)), [template.shape[2], -1])
    plt.plot(flat_tmp)
    plt.show()


def plot_skeleton(skeleton, x_lim=[], y_lim=[], fig_size=5, marker_size=10, ais=[]):
    x, y, z = [skeleton.vertices[:, x] for x in range(3)]
    x_scaling = np.abs(
        ((np.max(x) - np.min(x)) / (np.max(y) - np.min(y))) * 1.15
    )  # [0]
    clb_ticks = ["0", str(np.round(max(z) / 20, decimals=1))]

    fig, axes = plt.subplots(figsize=(fig_size * x_scaling, fig_size))

    im = plt.scatter(x, y, c=z, s=marker_size, cmap="coolwarm", marker="o", vmax=1)

    clb = plt.colorbar(
        ticks=[0, max(z)], format=mticker.FixedFormatter(clb_ticks), shrink=0.5
    )
    clb.set_label(label="Latency (ms)", size=6)
    clb.ax.tick_params(labelsize=6, length=0)
    # axes.set_axis_off()
    if len(x_lim) > 1 & len(y_lim) > 1:
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    else:
        axes.autoscale_view()
        axes.set_ylim(axes.get_ylim()[::-1])

    plt.clim([0, max(z)])
    if len(ais) > 0:
        plt.scatter(ais[0], ais[1], s=marker_size * 20, color="k")
    plt.show()
    return x, y, z


def generate_propagation_gif(
    template,
    params,
    cumulative=False,
    vertices=[],
    downsample=2,
    clim=[-10, 0],
    cmap="Greys",
    spacing=1,
    marker_size=10,
):
    el_offset = params["el_spacing"]
    interp_offset = el_offset * spacing
    xticks = np.arange(0, 3850, 500)
    yticks = np.arange(0, 2100, 500)
    conv_xticks = xticks / interp_offset
    conv_yticks = yticks / interp_offset
    if len(vertices) > 0:
        x, y, z = [vertices[:, x] for x in range(3)]

        clb_ticks = [
            "0",
            str(np.round(max(z) / (params["sampling_rate"] / 1000), decimals=1)),
        ]
    ims = []

    fig = plt.figure()  # added
    for i in range(1, template.shape[2], downsample):
        ax = plt.axes()
        if cumulative:
            plt_data = np.min(template[:, :, 0:i].T, axis=0)
        else:
            plt_data = template[:, :, i].T
        # im = ax.imshow(plt_data, animated=True,vmin=clim[0],vmax=clim[1],cmap=cmap)
        ax.imshow(
            plt_data, animated=True, vmin=clim[0], vmax=clim[1], cmap=cmap
        )  # added
        if len(vertices) > 0:
            xi, yi, zi = x[z <= i], y[z <= i], z[z <= i]
            scat = ax.scatter(
                xi,
                yi,
                c=zi,
                s=marker_size,
                cmap="coolwarm",
                vmin=np.min(z),
                vmax=np.max(z),
            )  # ,alpha=0.5)
        # if i == 1:
        #    clb = plt.colorbar(scat, ticks=[min(z), max(z)],format=mticker.FixedFormatter(clb_ticks),shrink= 0.5)
        #    clb.set_label(label="Latency (ms)",size=6)
        #    clb.ax.tick_params(labelsize=6,length=0)
        ax.set_xticks(conv_xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel("\u03bcm")

        ax.set_yticks(conv_yticks)
        ax.set_yticklabels(yticks)
        ax.set_ylabel("\u03bcm")
        ims.append([ax])
    ani = anim.ArtistAnimation(fig, ims, interval=100)
    return ani


def plot_filled_contour(
    interp_tmp, skeleton, params, radius=5, save_path=[], fig_size=1, font_size=24
):
    #interp_tmp = skel.interpolate_template(capped_template, spacing=params["upsample"])
   # interp_tmp = nd.gaussian_filter(interp_tmp, sigma=0.8)
    skel_paths = ut.convert_coor_scale(skeleton.paths(), params, "el")
    sorted_vertices = skel.path_to_vertices(skel_paths, params)
    skel_mat = np.zeros(interp_tmp.shape)
    skel_mat[tuple(sorted_vertices.astype("int").T)] = True
    dil_mat = nd.binary_dilation(skel_mat, structure=ball(radius))
    th_data = interp_tmp * dil_mat  # [:,:,t_cap[0]:t_cap[1]]
    contour_data = np.abs(np.min(th_data, axis=2).T)
    #contourf_lines = np.linspace(-5, -0.1, 15) #np.append(np.floor(-np.max(contour_data)),  )
    contourf_lines = np.geomspace(-np.max(contour_data),-0.1,10)
    # contourf_lines = np.append(np.geomspace(-np.max(contour_data),-3,10),np.linspace(-3, -0.1,15))
    fig, ax = plt.subplots(
        figsize=(22 * fig_size, 12 * fig_size), constrained_layout=True
    )
    plt.contourf(
        -contour_data, levels=contourf_lines, cmap="inferno", vmin=np.max([-5, -np.max(contour_data)]), vmax=-0.1
    )  # ,linewidths = 0.2,vmax=20,vmin=2)hatches =[':'],
    clb = plt.colorbar(
        ticks=[-np.max(contour_data), -0.1],
        #format=mticker.FixedFormatter(["-100", "-2"]),
        shrink=0.3,
    )
    clb.set_label(label="\u03bcV/ms", size=font_size)
    clb.ax.tick_params(labelsize=font_size, length=0)
    ax.autoscale_view()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()


def plot_delay_contour(
    interp_tmp, skeleton, params, skel_params, radius=5, save_path=[], linewidth=4, n_contours=15
):
    #interp_tmp = skel.interpolate_template(capped_template, spacing=params["upsample"])
    #interp_tmp = nd.gaussian_filter(interp_tmp, sigma=0.8)
    skel_paths = ut.convert_coor_scale(skeleton.paths(), params, "el")
    sorted_vertices = skel.path_to_vertices(skel_paths, params)
    #sorted_vertices = skel.path_to_vertices(skeleton.paths(), params)
    skel_mat = np.zeros(interp_tmp.shape)
    skel_mat[tuple(sorted_vertices.astype("int").T)] = True
    dil_mat = nd.binary_dilation(skel_mat, structure=ball(radius))
    th_data = interp_tmp * dil_mat  # [:,:,t_cap[0]:t_cap[1]]
    contour_data = np.abs(np.min(th_data, axis=2).T)
    # contour_lines = np.append(
    #     np.linspace(0.1, 2, 15), np.linspace(2.5, np.max(contour_data), 20)
    # )
    contour_lines = np.geomspace(0.1,np.max(contour_data),n_contours)

    fig, ax = plot_delay_skeleton(
        skel_paths,
        params,
        skel_params,
        figsize=1,
        ais=None,
        plot_ais_connection=False,
        linewidth=linewidth,
    )
    plt.contour(
        contour_data,
        levels=contour_lines,
        colors="k",
        linewidths=0.2,
        alpha=0.8,
        zorder=0,
    )  # ,vmax=20,vmin=2)hatches =[':'],
    ax.autoscale_view()
    #ax.set_ylim([0, 120])
    #ax.set_ylim(ax.get_ylim()[::-1])
    #ax.set_xlim([0, 220])
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()
