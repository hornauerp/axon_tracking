def clean_sorting(rec, save_root, stream_name, clear_files=True, verbose=True):
    """
    Function that creates output folder if it does not exist, sorts the recording using the specified sorter
    and clears up large files afterwards. 

    Parameters
    ----------
    rec: spikeinterface.
    ----------
    """
    # Creates output folder, sorts and clears large temporary files
    save_path = os.path.join(save_root,stream_name)
    output_folder = Path(os.path.join(save_path, 'sorted'))
    
    if not os.path.exists(os.path.join(output_folder,'amplitudes.npy')):
        output_folder.mkdir(parents=True, exist_ok=True)
        raw_file = os.path.join(output_folder, 'sorter_output', 'recording.dat')
        wh_file = os.path.join(output_folder, 'sorter_output', 'temp_wh.dat')

        if verbose:
            print(f"DURATION: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
                    f"NUM. CHANNELS: {rec.get_num_channels()}")

        # We use try/catch to not break loops when iterating over several sortings
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