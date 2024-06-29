import os
import pickle


NAMES = ['timestamp', 'bandwidth']


def load_traces(cooked_trace_folder, load_mahimahi_ptrs=True):
    print("Loading traces from " + cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    all_mahimahi_ptrs = []
    for subdir ,dirs ,files in os.walk(cooked_trace_folder):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        files.sort()
        for file in files:
            file_path = subdir + os.sep + file
            if file_path.endswith('.pkl'):
                if load_mahimahi_ptrs:
                    all_mahimahi_ptrs = pickle.load(open(file_path, 'rb'))
                continue

            val_folder_name = os.path.basename( os.path.normpath( subdir ) )
            cooked_time = []
            cooked_bw = []
            with open(file_path, 'rb') as phile:
                for line in phile:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(val_folder_name + '_' + file)
    return all_cooked_time, all_cooked_bw, all_file_names, all_mahimahi_ptrs


def compute_cdf(data):
    """ Return the cdf of input data.

    Args
        data(list): a list of numbers.

    Return
        sorted_data(list): sorted list of numbers.

    """
    length = len(data)
    sorted_data = sorted(data)
    cdf = [i / length for i, val in enumerate(sorted_data)]
    return sorted_data, cdf
