import os, kimimaro
import numpy as np
import scipy.ndimage as nd
from skimage.morphology import disk, ball
from skimage.feature import peak_local_max
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
import matplotlib.animation as anim
#from IPython.display import HTML

def load_template_file(root_path, stream_id, template_id):
    template_save_file = os.path.join(root_path, stream_id, 'sorter_output', 'templates', str(template_id) + '.npy')
    template = np.load(template_save_file).astype('float64')

    return template, template_save_file

def interpolate_template(template, spacing=0.5, template_path = []):
    split_path = template_path.split(sep='/')
    split_path[-1] = 'interp_' + split_path[-1]
    interp_tmp_path = '/'.join(split_path)
    if os.path.exists(interp_tmp_path):
        interp_template = np.load(interp_tmp_path).astype('float64')

    else:
        x, y, z = [np.arange(template.shape[k]) for k in range(3)]
        
        f = RegularGridInterpolator((x, y, z), template)
        new_grid = np.mgrid[0:x[-1]:spacing, 0:y[-1]:spacing, 0:z[-1]:spacing]
        new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
        interp_template = f(new_grid)

        if template_path:
            np.save(interp_tmp_path, interp_template.astype('float32'))
            
    return interp_template

def localize_neurons(input_mat, ms_cutout, min_distance=5, threshold_rel=0.5, num_peaks=3):

    local_max = peak_local_max(np.abs(input_mat), min_distance=min_distance, threshold_rel=threshold_rel, num_peaks=num_peaks)
    #print(local_max)
    ms_peak_cutout = 0.5

    cutout_ratio = (ms_cutout[0]/np.sum(ms_cutout))

    peak_range = [(cutout_ratio-(ms_peak_cutout/np.sum(ms_cutout))),(cutout_ratio+(ms_peak_cutout/np.sum(ms_cutout)))]
    peak_range = np.round(np.array(peak_range)*input_mat.shape[2])

    #pre_coor = local_max[local_max[:,2] < peak_range[0],:].astype("int16")
    target_coor = local_max[(local_max[:,2] >= peak_range[0]) & (local_max[:,2] <= peak_range[1]),:].astype("int16")
    buffer_frames = 5
    capped_matrix = input_mat[:,:,(target_coor[:,2][0] - buffer_frames):]
    target_coor[0][2] = buffer_frames

    post_coor = local_max[local_max[:,2] > peak_range[1],:].astype("int16")
    if post_coor: #Check if postsynaptic target was detected
        post_coor[0][2] = post_coor[0][2] - target_coor[0][2]
    
    
    return capped_matrix, target_coor, post_coor

def generate_dilation_structure(r):
    """
    r: distance in electrodes from the detected signal to be detectable in the next/previous frame
    """
    d = (2*r) + 1 #Convert to diameter for mask setup
    structure = np.full((3,d,d),False)
    structure[0,:,:] = disk(r).astype(bool)
    structure[1,r,r] = True
    structure[2,:,:] = disk(r).astype(bool)

    return structure

def iterative_dilation(template, r_dilation=2, init_th=-10, min_th=-1, filter_footprint =(3,3,3), use_derivative=True):
    if use_derivative:
        template = np.diff(template)

    structure = generate_dilation_structure(r_dilation)
    m_init = template < init_th #Detection of initial seeds/definitive peaks
    mask = template < min_th #Mask indicating potential peak locations
    dilated = nd.binary_dilation(m_init, structure=structure, iterations=0, mask=mask)
    #filtered = nd.median_filter(dilated,size=filter_size)
    #filtered = nd.median_filter(dilated,footprint=ball(1))
    if filter_footprint is not None:
        dilated = nd.median_filter(dilated,footprint=filter_footprint)

    return dilated

def skeletonize(input_matrix, scale=2, const=50, pdrf_exponent=4, pdrf_scale=10000, dust_threshold=0, anisotropy=(17.5,17.5,50.0), tick_threshold=10):
    skels = kimimaro.skeletonize(
    input_matrix, 
    teasar_params={
    'scale': scale,
    'const': const, # physical units
    'pdrf_exponent': pdrf_exponent,
    'pdrf_scale': pdrf_scale,
    "soma_acceptance_threshold": 35, # physical units
    "soma_detection_threshold": 20, # physical units
    "soma_invalidation_const": 30, # physical units
    "soma_invalidation_scale": 2,
    },
    dust_threshold = dust_threshold,
    parallel = 12)

    return skels

def generate_propagation_gif(template, cumulative=True, skeleton = [], downsample=2, clim=[-10, 0], cmap="Greys"):
    spacing = 1/3
    el_offset = 17.5
    interp_offset = el_offset*spacing
    xticks = np.arange(0,3850,500)
    yticks = np.arange(0,2200,500)
    conv_xticks = xticks/interp_offset
    conv_yticks = yticks/interp_offset
    if skeleton:
        x,y,z = [skeleton.vertices[:,x] for x in range(3)]
    
    ims = []
    #fig, ax = plt.subplots()
    fig = plt.figure() #added
    for i in range(1,template.shape[2],downsample):
        ax = plt.axes()
        if cumulative:
            plt_data = np.min(template[:,:,0:i].T, axis=0)
        else:
            plt_data = template[:,:,i].T
        #im = ax.imshow(plt_data, animated=True,vmin=clim[0],vmax=clim[1],cmap=cmap)
        ax.imshow(plt_data, animated=True,vmin=clim[0],vmax=clim[1],cmap=cmap) #added
        if skeleton:
            xi,yi,zi = x[z<=i], y[z<=i], z[z<=i]
            ax.scatter(xi,yi,c=zi,s=0.1,cmap="coolwarm",vmin=np.min(z),vmax=np.max(z),alpha=0.5)
        #clb = plt.colorbar()
        ax.set_xticks(conv_xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel(u"\u03bcm")
        
        ax.set_yticks(conv_yticks)
        ax.set_yticklabels(yticks)
        ax.set_ylabel(u"\u03bcm")
        #ims.append([im])
        ims.append([ax])
    ani = anim.ArtistAnimation(fig, ims,interval=100)
    return ani
