import os, time, h5py
import spikeinterface.full as si
import numpy as np
from pathlib import Path

def sort_recording_list(path_list, save_path_changes, sorter, sorter_params = dict(), clear_files=True, verbose=True):
    """
    Function that iterates over a list of axon scans, finds common electrodes, concatenates and spike sorts the recording slices. 

    Arguments
    ----------
    path_list: list
        List of string referring to the paths of axon scan recording.
    save_path_changes: dict
        Dictionary containing keys 'pos' and 'vals' that indicate the changes to be made to the rec_path.
        Refer to the inidices after splitting the path by '/'.
    sorter: str
        Name of the sorter, as per the spikeinterface convention (e.g. 'kilosort2_5')
    sorter_params: dict
        Dictionary containing parameters for the spike sorter. If left empty, will default to default parameters.
    clear_files: bool
        Flag if large temporary files should be deleted after the sorting. Default: True
    verbose: bool
        Default: True

    Returns
    ----------
    sorting_list: list of sorting objects 
        Specific type depends on the sorter.
    
    """
    
    sorting_list = []
    
    for rec_path in path_list:
        h5 = h5py.File(rec_path)
        #Check that all wells are recorded throughout all recordings (should not fail)
        recs = h5['recordings'].keys()
        assert(h5['recordings'][list(recs)[0]].keys() == h5['recordings'][list(recs)[-1]].keys())

        save_root = convert_rec_path_to_save_path(rec_path, save_path_changes)
        
        for stream_name in list(h5['recordings'][list(recs)[0]]):
            multirecording =  concatenate_recording_slices(rec_path, stream_name)
            sorting = clean_sorting(multirecording, save_root, stream_name, sorter, sorter_params, clear_files=clear_files, verbose=verbose)
            sorting_list.append(sorting)
            
    return sorting_list



def convert_rec_path_to_save_path(rec_path, save_path_changes):
    """
    Function that converts a recording path to the corresponding save path for a spike sorting.

    Arguments
    ----------
    rec_path: str
        Path to the axon scan file.
    save_path_changes: dict
        Dictionary containing keys 'pos' and 'vals' that indicate the changes to be made to the rec_path.
        Refer to the inidices after splitting the path by '/'.
        
    Returns
    ----------
    save_path: str
        Root save path. Well ID will be appended during the sorting.
    """

    path_parts = rec_path.split('/')
    for x,y in zip(save_path_changes['pos'], save_path_changes['vals']):
        path_parts[x] = y
        
    save_path = os.path.join(*path_parts)
    
    return save_path

def find_common_electrodes(rec_path, stream_name):
    """
    Function that returns the common electrodes of the successive axon scan recordings.
    
    Arguments
    ----------
    rec_path: str
        Path to the axon scan file.
    stream_name: str
        Well ID in the format "well***"; Well 1 would be "well001", Well 20 would be "well020"
        
    Returns
    ----------
    common_el: list
        List of electrodes that are present in all axon scan recordings.
    """
    
    assert(os.path.exists(rec_path))
    
    h5 = h5py.File(rec_path)
    n_recs = len(h5['recordings'].keys())

    for rec_id in range(n_recs):
        rec_name = 'rec' + '%0*d' % (4, rec_id)
        rec = si.MaxwellRecordingExtractor(rec_path, stream_name=stream_name, rec_name=rec_name)
        rec_el = rec.get_property("contact_vector")["electrode"]
        if rec_id == 0:
            common_el = rec_el
        else:
            common_el = list(set(common_el).intersection(rec_el))
            
    return n_recs, common_el



def concatenate_recording_slices(rec_path, stream_name):
    """
    Function that centers and concatenates the recordings of an axon scan for all common electrodes. 

    Arguments
    ----------
    rec_path: str
        Path to the axon scan file.
    stream_name: str
        Well ID in the format "well***"; Well 1 would be "well001", Well 20 would be "well020"
        
    Returns
    ----------
    multirecording: ConcatenatedRecordingSlice
        Concatenated recording across common electrodes (spikeinterface object)
    """

    n_recs, common_el = find_common_electrodes(rec_path, stream_name)
    
    rec_list = []
    for r in range(2):#(n_recs): 
        rec_name = 'rec' + '%0*d' % (4, r)
        rec = si.MaxwellRecordingExtractor(rec_path, stream_name=stream_name, rec_name=rec_name)
                
        ch_id = rec.get_property("contact_vector")['device_channel_indices']
        rec_el = rec.get_property("contact_vector")["electrode"]
        
        chan_idx = [np.where(rec_el == el)[0][0] for el in common_el]
        sel_channels = rec.get_channel_ids()[chan_idx]
        chunk_size = np.min([10000, rec.get_num_samples()]) - 100 #Fallback for ultra short recordings (too little activity?)
        rec_centered = si.center(rec,chunk_size=chunk_size)
        rec_list.append(rec_centered.channel_slice(sel_channels, renamed_channel_ids=list(range(len(chan_idx)))))
    
    multirecording = si.concatenate_recordings(rec_list)
    
    return multirecording
    


def clean_sorting(rec, save_root, stream_name, sorter, sorter_params = dict(), clear_files=True, verbose=True):
    """
    Function that creates output folder if it does not exist, sorts the recording using the specified sorter
    and clears up large files afterwards. 

    Arguments
    ----------
    rec: MaxwellRecordingExtractor
        Recording to be sorted.
    save_root: str
        Root path where the sorted data will be stored. Stream name (i.e. well ID) will be appended.
    stream_name: str
        Well ID in the format "well***"; Well 1 would be "well001", Well 20 would be "well020"
    sorter: str
        Name of the sorter, as per the spikeinterface convention (e.g. 'kilosort2_5')
    sorter_params: dict
        Dictionary containing parameters for the spike sorter. If left empty, will default to default parameters.
    clear_files: bool
        Flag if large temporary files should be deleted after the sorting. Default: True
    verbose: bool
        Default: True

    Returns
    ----------
    sorting: Sorting object 
        Specific type depends on the sorter.
    """
    
    output_folder = Path(os.path.join(save_root, stream_name, 'sorted'))

    # Creates output folder if sorting has not yet been done
    if not os.path.exists(os.path.join(output_folder,'amplitudes.npy')):
        output_folder.mkdir(parents=True, exist_ok=True)
        raw_file = os.path.join(output_folder, 'sorter_output', 'recording.dat')
        wh_file = os.path.join(output_folder, 'sorter_output', 'temp_wh.dat')

        if verbose:
            print(f"DURATION: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
                    f"NUM. CHANNELS: {rec.get_num_channels()}")

        # We use try/catch to not break loops when iterating over several sortings (e.g. when not all wells were recorded)
        try:
            t_start_sort = time.time()
            sorting = si.run_sorter(sorter, rec, output_folder=output_folder, verbose=verbose,
                                    **sorter_params)
            if verbose:
                print(f"\n\nSpike sorting elapsed time {time.time() - t_start_sort} s")
            
            #Making sure we clean up the largest temporary files
            if clear_files & os.path.exists(wh_file):
                os.remove(wh_file)
            if clear_files & os.path.exists(raw_file):
                os.remove(raw_file)
        except Exception as e:
            sorting = []
            print(e)
            if clear_files & os.path.exists(wh_file):
                os.remove(wh_file)
            if clear_files & os.path.exists(raw_file):
                os.remove(raw_file)
                
    return sorting

