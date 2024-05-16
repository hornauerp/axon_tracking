import os, h5py
import spikeinterface.full as si
import numpy as np
import scipy as sp
import sklearn as sk
from tqdm import tqdm

from axon_tracking import spike_sorting as ss

def extract_templates_from_sorting_dict(sorting_dict, qc_params={}, te_params={}):
    rec_list = list(sorting_dict.keys())

    for rec_path in rec_list:
        sorting_list = sorting_dict[rec_path]
        sorting_list.sort()

        for sorting_path in sorting_list:
            sorting = si.KiloSortSortingExtractor(sorting_path)
            stream_id = [p for p in sorting_path.split('/') if p.startswith('well')][0] #Find out which well this belongs to
            print(stream_id)
            #rec_names, common_el, pos = ss.find_common_electrodes(rec_path, stream_id)
            multirecording, common_el, pos = ss.concatenate_recording_slices(rec_path, stream_id)
            cleaned_sorting = select_good_units(sorting, **qc_params)
            cleaned_sorting = si.remove_excess_spikes(cleaned_sorting, multirecording) #Relevant if last spike time == recording_length
            cleaned_sorting.register_recording(multirecording)
            segment_sorting = si.SplitSegmentSorting(cleaned_sorting, multirecording)
            extract_all_templates(stream_id, segment_sorting, sorting_path, pos, te_params)

def extract_templates_from_concatenated_recording(root_path, stream_id, qc_params={}, te_params={}):

    sorting_path = os.path.join(root_path, stream_id)
    seg_sorting, concat_rec = ss.split_concatenated_sorting(sorting_path)
        
    # Split axon tracking
    #ax_sorting = si.select_segment_sorting(seg_sorting,0)
    #ax_rec_path = ss.get_recording_path(ax_sorting)
    #ax_recording, common_el, pos = ss.concatenate_recording_slices(ax_rec_path, stream_id, center=False)
    cleaned_sorting = select_good_units(seg_sorting, **qc_params)
    #ax_sorting = si.select_segment_sorting(cleaned_sorting,0)
    #ax_sorting = si.remove_excess_spikes(ax_sorting, ax_recording)
    #ax_sorting.register_recording(ax_recording)
    #ax_split_sorting = si.SplitSegmentSorting(ax_sorting, ax_recording)

    # Split network recordings
    nw_sorting = si.select_segment_sorting(cleaned_sorting,1)
    nw_recording = concat_rec._kwargs['recording_list'][1]._kwargs['recording']._kwargs['parent_recording']
    nw_sorting = si.remove_excess_spikes(nw_sorting, nw_recording)
    nw_sorting.register_recording(nw_recording)
    nw_split_sorting = si.SplitSegmentSorting(nw_sorting, nw_recording)

    # Save split sortings
    ss.save_split_sorting(nw_split_sorting)
    
    # Extract templates
    #extract_all_templates(stream_id, ax_split_sorting, sorting_path, pos, te_params)

def get_assay_information(rec_path):
    h5 = h5py.File(rec_path)
    pre, post, well_id = -1, -1, 0
    while pre <= 0 or post <= 0: #some failed axon trackings give negative trigger_post values, so we try different wells
        well_name = list(h5['wells'].keys())[well_id]
        rec_name = list(h5['wells'][well_name].keys())[well_id]
        pre = h5['wells'][well_name][rec_name]['groups']['routed']['trigger_pre'][0]
        post = h5['wells'][well_name][rec_name]['groups']['routed']['trigger_post'][0]
        well_id += 1
        
    return [pre, post]

def find_files(save_root, file_name="templates.npy", folder_name="sorter_output"):
    file_list = [root
                 for root, dirs, files in os.walk(save_root)
                 for dir in dirs
                 if dir == folder_name and os.path.exists(os.path.join(root,folder_name,file_name))]
    #file_list = [os.path.join(file, folder_name) for file in file_list] 
    return file_list

def find_successful_sortings(path_list, save_path_changes):

    sorting_dict = dict()
    for rec_path in path_list:
        save_root = ss.convert_rec_path_to_save_path(rec_path, save_path_changes)
        
        #Takes into account different sorting folder names, subfolder depth, well IDs etc.
        sorting_files = [root
                         for root, dirs, files in os.walk(save_root)
                         for name in files
                         if name == "templates.npy"]
        sorting_dict[rec_path] = sorting_files
        
    return sorting_dict

            
def postprocess_sorting():
    #Maybe we will do some postprocessing before we use them
    return

def select_good_units(sorting, min_n_spikes=1500, exclude_mua=True, use_bc=True):
    if exclude_mua:
        ks_idx = sorting.get_property('KSLabel') == 'good'
    else:
        ks_idx = np.full((sorting.get_num_units(),), True, dtype='bool')

    if use_bc and sorting.get_property('bc_unitType'):
        bc_idx = sorting.get_property('bc_unitType') == 'GOOD'
    else:
        #print('No bombcell output found')
        bc_idx = np.full((sorting.get_num_units(),), True, dtype='bool')
            
    n_spikes = [len(sorting.get_unit_spike_train(x,segment_index=0)) for x in sorting.get_unit_ids()]

    
    good_n_spikes_idx = np.array(n_spikes) > min_n_spikes
    good_idx = ks_idx & bc_idx & good_n_spikes_idx
    good_ids = sorting.get_unit_ids()[good_idx]
    cleaned_sorting = sorting.select_units(good_ids)
    
    return cleaned_sorting



def extract_waveforms(segment_sorting, stream_id, save_root, n_jobs, overwrite_wf):
    full_path = ss.get_recording_path(segment_sorting)
        
    cutout = [x / (segment_sorting.get_sampling_frequency()/1000) for x in get_assay_information(full_path)] #convert cutout to ms
    h5 = h5py.File(full_path)
    rec_names = list(h5['wells'][stream_id].keys())
    
    for sel_idx, rec_name in enumerate(rec_names):
        wf_path = os.path.join(save_root, 'waveforms', 'seg' + str(sel_idx))
        if not os.path.exists(wf_path) or overwrite_wf:
            rec = si.MaxwellRecordingExtractor(full_path,stream_id=stream_id,rec_name=rec_name)
            chunk_size = np.min([10000, rec.get_num_samples()]) - 100 #Fallback for ultra short recordings (too little activity)
            rec_centered = si.center(rec, chunk_size=chunk_size)
            
            seg_sort = si.SelectSegmentSorting(segment_sorting, sel_idx)
            seg_sort = si.remove_excess_spikes(seg_sort, rec_centered)
            seg_sort.register_recording(rec_centered)
                    
            seg_we = si.WaveformExtractor.create(rec_centered, seg_sort,
                                                 wf_path, 
                                                 allow_unfiltered=True,
                                                 remove_if_exists=True)
            seg_we.set_params(ms_before=cutout[0], ms_after=cutout[1], return_scaled = True)
            seg_we.run_extract_waveforms(n_jobs=n_jobs)


def align_waveforms(seg_we, sel_unit_id, cutout, ms_peak_cutout, upsample, align_cutout, rm_outliers, n_jobs, n_neighbors):
    
    sample_peak_cutout = ms_peak_cutout * upsample
    peak_idx = cutout[0] * upsample
    peak_cutout = range(np.int16(peak_idx - sample_peak_cutout), np.int16(peak_idx + sample_peak_cutout))
    wfs = seg_we.get_waveforms(sel_unit_id)
    interp_wfs = sp.interpolate.pchip_interpolate(list(range(wfs.shape[1])), wfs, np.linspace(0,wfs.shape[1], num = wfs.shape[1]*upsample), axis=1)
    interp_wfs = interp_wfs - np.median(interp_wfs, axis=1)[:,np.newaxis,:]

    if align_cutout:
        peak_el = [np.where(interp_wfs[w,peak_cutout,:] == np.nanmin(interp_wfs[w,peak_cutout,:]))[1][0] for w in range(interp_wfs.shape[0])]
        ref_el, count = sp.stats.mode(peak_el, keepdims=False)
        peak_shift = [np.where(interp_wfs[w,peak_cutout,ref_el] == np.nanmin(interp_wfs[w,peak_cutout,ref_el]))[0][0] for w in range(interp_wfs.shape[0])]
        aligned_length = interp_wfs.shape[1] - 2*sample_peak_cutout
        aligned_wfs = np.full([interp_wfs.shape[0], np.int16(aligned_length), interp_wfs.shape[2]], np.nan)
        for w in range(interp_wfs.shape[0]):
            aligned_wfs[w,:,:] = interp_wfs[w,peak_shift[w]:np.int16(peak_shift[w]+aligned_length),:]
    else:
        aligned_wfs = interp_wfs
    

    if rm_outliers:
        peak_el = [np.where(interp_wfs[w,peak_cutout,:] == np.nanmin(interp_wfs[w,peak_cutout,:]))[1][0] for w in range(interp_wfs.shape[0])]
        ref_el, count = sp.stats.mode(peak_el, keepdims=False)
        aligned_wfs = remove_wf_outliers(aligned_wfs, ref_el, n_jobs, n_neighbors)

    aligned_template = np.median(aligned_wfs, axis=0)
    
    return aligned_template


def remove_wf_outliers(aligned_wfs, ref_el, n_jobs, n_neighbors):
    clf = sk.neighbors.LocalOutlierFactor(n_jobs=n_jobs, n_neighbors=n_neighbors)
    outlier_idx = clf.fit_predict(aligned_wfs[:,:,ref_el])
    #print(f'Detected {sum(outlier_idx==-1)} outliers')
    outlier_rm = np.delete(aligned_wfs, outlier_idx==-1, axis=0)
    
    return outlier_rm

def combine_templates(stream_id, segment_sorting, sel_unit_id, save_root, peak_cutout=2, align_cutout=True, upsample=2, rm_outliers=True, n_jobs=16, n_neighbors=10, overwrite_wf=False, overwrite_tmp = True):
    full_path = ss.get_recording_path(segment_sorting)
        
    cutout = get_assay_information(full_path)
    if align_cutout:
        wf_length = np.int16((sum(cutout) - 2*peak_cutout) * upsample) #length of waveforms after adjusting for potential peak alignments
        
    else:
        wf_length = np.int16(sum(cutout) * upsample)
        
    template_matrix = np.full([wf_length, 26400], np.nan)
    noise_levels = np.full([1,26400], np.nan)
        
    extract_waveforms(segment_sorting, stream_id, save_root, n_jobs, overwrite_wf)

    h5 = h5py.File(full_path)
    rec_names = list(h5['wells'][stream_id].keys())
    
    for sel_idx, rec_name in enumerate(rec_names):
        rec = si.MaxwellRecordingExtractor(full_path,stream_id=stream_id,rec_name=rec_name)
        els = rec.get_property("contact_vector")["electrode"]
        seg_sort = si.SelectSegmentSorting(segment_sorting, sel_idx)
        seg_we = si.load_waveforms(os.path.join(save_root, 'waveforms', 'seg' + str(sel_idx)), sorting = seg_sort)
        aligned_wfs = align_waveforms(seg_we, sel_unit_id, cutout, peak_cutout, upsample, align_cutout, rm_outliers, n_jobs, n_neighbors)
        template_matrix[:,els] = aligned_wfs #find way to average common electrodes 
        noise_levels[:,els] = si.compute_noise_levels(seg_we)
    
    return template_matrix, noise_levels

def convert_to_grid(template_matrix, pos):
    clean_template = np.delete(template_matrix, np.isnan(pos['x']), axis = 1)
    clean_x = pos['x'][~np.isnan(pos['x'])]
    clean_y = pos['y'][~np.isnan(pos['y'])]
    x_idx = np.int16(clean_x / 17.5)
    y_idx = np.int16(clean_y / 17.5)
    grid = np.full([np.max(x_idx) + 1, np.max(y_idx) + 1, clean_template.shape[0]],0).astype('float32')
    for i in range(len(y_idx)):
        grid[x_idx[i],y_idx[i],:] = clean_template[:,i]
    
    return grid


def extract_all_templates(stream_id, segment_sorting, save_root, pos, te_params):
    sel_unit_ids = segment_sorting.get_unit_ids()
    template_save_path = os.path.join(save_root, 'templates')
    if not os.path.exists(template_save_path):
        os.makedirs(template_save_path)
        
    for sel_unit_id in tqdm(sel_unit_ids): 
        template_save_file = os.path.join(template_save_path, str(sel_unit_id) + '.npy')
        noise_save_file = os.path.join(template_save_path, str(sel_unit_id) + '_noise.npy')
        
        if not os.path.isfile(template_save_file) or te_params['overwrite_tmp']:
            try:
                template_matrix, noise_levels = combine_templates(stream_id, segment_sorting, sel_unit_id, save_root, **te_params)
                grid = convert_to_grid(template_matrix, pos)
                np.save(template_save_file, grid)
                noise_levels = convert_to_grid(noise_levels, pos)
                np.save(noise_save_file, noise_levels)
                
            except Exception as e:
                print(f'Unit {sel_unit_id} encountered the following error:\n {e}')

