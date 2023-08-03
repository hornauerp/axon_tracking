import os
import spikeinterface.full as si
import numpy as np

def postprocess_sorting():

    return


def select_units():

    return


def 


def extract_waveforms(concatenated_recording, segment_sorting, stream_name, ms_cutout, n_jobs):
    for sel_idx in range(len(concatenated_recording.recording_list)):
        rec_name = 'rec' + '%0*d' % (4, sel_idx)
        rec = si.MaxwellRecordingExtractor(full_path,stream_name=stream_name,rec_name=rec_name)
        ss = si.SelectSegmentSorting(split_sorting, sel_idx)
        ss.register_recording(rec)
        wf_path = wf_folder + '_seg' + str(sel_idx)
    
        seg_we = si.WaveformExtractor.create(rec, ss,
                                         wf_path, 
                                         allow_unfiltered=True,
                                         remove_if_exists=True)
        seg_we.set_params(ms_before=ms_cutout[0], ms_after=ms_cutout[1], return_scaled = True)
        seg_we.run_extract_waveforms(n_jobs=n_jobs)


def align_waveforms(seg_we, sel_unit_id, ms_peak_cutout, upsample, rm_outliers):
    ms_conv = seg_we.recording.get_sampling_frequency() / 1000
    sample_peak_cutout = ms_peak_cutout * ms_conv * upsample
    peak_idx = ms_cutout[0] * ms_conv * upsample
    peak_cutout = range(np.int16(peak_idx - sample_peak_cutout), np.int16(peak_idx + sample_peak_cutout))
    wfs = seg_we.get_waveforms(sel_unit_id)
    interp_wfs = sp.interpolate.pchip_interpolate(list(range(wfs.shape[1])), wfs, np.linspace(0,wfs.shape[1], num = wfs.shape[1]*upsample), axis=1)
    interp_wfs = interp_wfs - np.median(interp_wfs, axis=1)[:,np.newaxis,:]
    
    peak_el = [np.where(interp_wfs[w,peak_cutout,:] == np.nanmin(interp_wfs[w,peak_cutout,:]))[1][0] for w in range(interp_wfs.shape[0])]
    ref_el, count = sp.stats.mode(peak_el,keepdims=False)
    peak_shift = [np.where(interp_wfs[w,peak_cutout,ref_el] == np.nanmin(interp_wfs[w,peak_cutout,ref_el]))[0][0] for w in range(interp_wfs.shape[0])]
    aligned_length = interp_wfs.shape[1] - 2*sample_peak_cutout
    aligned_wfs = np.full([interp_wfs.shape[0], np.int16(aligned_length), interp_wfs.shape[2]], np.nan)
    for w in range(interp_wfs.shape[0]):
        aligned_wfs[w,:,:] = interp_wfs[w,peak_shift[w]:np.int16(peak_shift[w]+aligned_length),:]

    if rm_outliers:
        aligned_wfs = remove_wf_outliers(aligned_wfs, ref_el, n_jobs, n_neighbors)

    aligned_template = np.median(aligned_wfs, axis=0)
    
    return aligned_template


def remove_wf_outliers(aligned_wfs, ref_el, n_jobs, n_neighbors):
    clf = sk.neighbors.LocalOutlierFactor(n_jobs=n_jobs, n_neighbors=n_neighbors)
    outlier_idx = clf.fit_predict(aligned_wfs[:,:,ref_el])
    #print(f'Detected {sum(outlier_idx==-1)} outliers')
    outlier_rm = np.delete(aligned_wfs, outlier_idx==-1, axis=0)
    
    return outlier_rm


def extract_all_templates(segment_sorting):
    sel_unit_ids = segment_sorting.get_unit_ids()
    
    for id in tqdm(range(len(sel_unit_ids))):
        sel_unit_id = sel_unit_ids[id]
        template_save_path = os.path.join(save_root, 'template_' + str(sel_unit_id))
        os.makedirs(template_save_path, exist_ok=True)
        template_save_file = os.path.join(template_save_path,'template.npy')
        
        if not os.path.isfile(template_save_file):
            try:
                template_matrix = combine_templates(full_path, stream_name, segment_sorting, sel_unit_id, ms_peak_cutout, upsample, rm_outliers, n_jobs, n_neighbors)
                np.save(os.path.join(template_save_path,'template.npy'), template_matrix)
            except Exception as e:
                print(f'Unit {sel_unit_id} encountered the following error')
                print(e)