from math import floor
import pandas as pd
import numpy as np
import h5py
import matplotlib.pylab as plt
import os
import librosa
import librosa.display
import pickle
import ctypes

#FUNCTIONS

def hethsorter_main():
    global letter_names
    global time_converter
    letter_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    time_converter = 0.023219814
    params = pd.read_csv("hethsorter_params.csv")
    wavename = params.iloc[0]['wavename'].split('.')[0]
    
    if os.path.exists(wavename + '.wav'):
        if os.path.exists(wavename + '.h5'):
            with h5py.File(wavename + '.h5', 'r') as hf:
                data = hf[wavename + '_dataset'][:]
        else:
            data = set_up(wavename)
    else:
        print(f'{wavename} does not exist.') 
    
    template_wave_name = params.iloc[0]['template_wave']
    template_table_name = params.iloc[0]['template_table']
    if isinstance(template_wave_name, str) and isinstance(template_table_name, str):
        print('found template names')
        template_wave_name = template_wave_name.split('.')[0]
        template_table_name = template_table_name.split('.')[0]
        if os.path.exists(template_wave_name + '.wav') and os.path.exists(template_table_name + '.txt'):
            
        #create a labelled raven file from templates, either by creating a template pickle file or by using an existing one 
            if os.path.exists('templates_for_' + template_wave_name + '.pkl'):
                with open('templates_for_' + template_wave_name + '.pkl', 'rb') as f:
                    template_dict = pickle.load(f)
                print('template found and loaded', np.shape(template_dict['A']))
            else:
                template_table = pd.read_csv(template_table_name + '.txt', sep='\t')
                if not os.path.exists(template_wave_name + '.h5'):
                    set_up(template_wave_name)
                with h5py.File(template_wave_name + '.h5', 'r') as hf:
                    temp_data = hf[template_wave_name + '_dataset'][:]
                print('template exists')
                template_dict = new_create_template(temp_data, template_table)

                with open('templates_for_' + template_wave_name + '.pkl', 'wb') as f:
                    pickle.dump(template_dict, f)
                print('template created', np.shape(template_dict['A']))

            unlabelled_table_name = params.iloc[0]['tablename'].split('.')[0]
            unlabelled_table = pd.read_csv(unlabelled_table_name + '.txt', sep='\t')
            for index, row in unlabelled_table.iterrows():
                pool = post_pool(data, int(floor(row['Begin Time (s)']/time_converter)), len(data), len(data[0]), 15, 10)
                filter_one_pool(pool)
                normalize_one_pool(pool)
                lowest_score = 1000
                lowest_template = ''
                for template in template_dict.keys():
                    score = score_two_pools(template_dict[template], pool)
                    if score < lowest_score:
                        lowest_score = score
                        lowest_template = template
                unlabelled_table.loc[index, 'ID'] = lowest_template
                print('added: ', lowest_template) 
            unlabelled_table.to_csv('hs_wt_' + wavename + '.txt', sep='\t', index=False)
            Mbox('HethSorter Results', f'Used template to create hs_wt_{wavename}.txt', 0x00001000)
        else:
            print('did not find template files', template_wave_name, template_table_name)
        
    #create a labelled raven file without templates
    else:
        print('no template files found')
        completion_strings = []
        for index, row in params.iterrows():
            completion_strings.append(hethsorter_nt_one_file(row['wavename'], row['tablename']))
            print(f'COMPLETED {index + 1}.')

        mbox_string = ''
        for comp_string in completion_strings:
            mbox_string += comp_string + '\n\n'
        Mbox('HethSorter Results', mbox_string, 0x00001000)
                            
def hethsorter_nt_one_file(wavename, tablename):
    wavename = wavename.split('.')[0]
    global pool_dict
    raven_table, pool_dict, compare_dict = compare_no_template(tablename, wavename)
    new_table_name = label_no_template(raven_table, wavename, pool_dict, compare_dict, .3)
    return f"created new table without a template. {new_table_name} has song-type labels added."
                                  
                                  
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)                          

#get the spectrogram and raven table for the templates
#check if appropriately named template file exists
def new_create_template(data, template_table):
    all_labels = template_table['ID'].unique()
    print(all_labels)
    template_dict = {}
    for label in all_labels:
        current_pool_list = []
        for index, row in template_table.iterrows():
            if row['ID'] == label:
                pool = post_pool(data, int(floor(row['Begin Time (s)']/time_converter)), len(data), len(data[0]), 15, 10)
                filter_one_pool(pool)
                normalize_one_pool(pool)
                current_pool_list.append(pool)
        mean_pool = np.mean(current_pool_list, axis=0)
        print(label)
        template_dict[label] = mean_pool
    return template_dict

#store the dictionary as a pickle file with a name starting with 'templates_for_filename'

def fourier(filename, wave_start = 0, wave_end = 0):
    y, sr = librosa.load(filename + '.wav')
    start_int = int(wave_start * sr)
    end_int = int(wave_end * sr)
    duration = librosa.get_duration(y=y, sr=sr)
    if wave_start > duration:
        return 'fail'
    
    if end_int > 0 and wave_end < duration:
        D = librosa.stft(y[start_int:end_int])
    else:
        D = librosa.stft(y[start_int:])
    D = D[120:743,:]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

def set_up(filename):
    data = fourier(filename)
    with h5py.File(filename + '.h5', 'w') as hf:
        hf.create_dataset(filename + "_dataset", data=data)
    return data

#creates a post introductory pooled spectrogram
def post_pool(array, column, rows, columns, cell_height, cell_length):
    column_limit = column + 70
    row_index = 0
    post_note_array = []
    if column + 70 > columns:
        column_limit = columns
    while row_index < rows - cell_height:
        col_index = 0
        row_values = []
        while column + col_index <= column_limit - cell_length:
            row_values.append(np.mean(array[row_index:row_index + cell_height,column + col_index:column + col_index + cell_length]))
            col_index += cell_length
        post_note_array.append(row_values)
        row_index += cell_height
    return np.array(post_note_array)

def display_pool(start_col, intro_freq):
    with h5py.File(filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    sample_pool = post_pool(data, start_col, len(data), len(data[0]), 15, 10)
    cut_pool = cut_top_third(sample_pool, floor(intro_freq / 10))
    fig, ax = plt.subplots(figsize=(15, 7))
    # img = librosa.display.specshow(array[1], x_axis='time', y_axis=None, sr=22050, ax=ax)
    img = librosa.display.specshow(cut_pool, x_axis=None, y_axis=None, sr=22050, ax=ax)
    ax.set_title('Spectrogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 24, 1))
    fig.gca().set_xticks(range(0, 9, 1))
    fig.gca().set_ylabel("Row")
    plt.show()
    
def cut_top_third(pool, intro_freq):
    if intro_freq < 2:
        sweet_bot = 0
    else:
        sweet_bot = intro_freq - 2
    if intro_freq > (len(pool) / 2):
        high_song = True
    if intro_freq + 23 > len(pool):
        sweet_top = len(pool)
        sweet_bot = len(pool) - 25
    else:
        sweet_top = intro_freq + 23
    sweet_spot_array = pool[sweet_bot: sweet_top, 2:11]

    top_third_val = np.percentile(sweet_spot_array, 70)
    cut_sweet = np.where(sweet_spot_array > top_third_val, sweet_spot_array, -80)
    return cut_sweet

def display_spect(array):
    fig, ax = plt.subplots(figsize=(15, 7))
    # img = librosa.display.specshow(array[1], x_axis='time', y_axis=None, sr=22050, ax=ax)
    img = librosa.display.specshow(array, x_axis=None, y_axis=None, sr=22050, ax=ax)
    ax.set_title('Spectrogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    #fig.gca().set_yticks(range(0, 743-120, 25))
    fig.gca().set_ylabel("Row")
    plt.show()

def score_two_pools(pool1, pool2):
    max_possible = abs(np.sum(pool1) + np.sum(pool2))
    score = (sum(abs(pool1.flatten() - pool2.flatten())))/max_possible
    return score

def count_two_pools(pool1, pool2):
    score = 0
    count1 = count_active_cells(pool1)
    count2 = count_active_cells(pool2)
    total_possible = count1 + count2
    for i in range(len(pool1)):
        for j in range(len(pool1[0])):
            if pool1[i, j] > -80 and pool2[i,j] > -80:
                score += 1
    return 1 - score/total_possible
            


def make_compare_dict(pool_dict):
    compare_dict = {}
    for thisKey in pool_dict:
        for thatKey in pool_dict:
            if f'{thisKey}:{thatKey}' in compare_dict or f'{thatKey}:{thisKey}' in compare_dict or thisKey == thatKey:
                continue
            score = score_two_pools(pool_dict[thisKey], pool_dict[thatKey])
            compare_dict[f'{thisKey}:{thatKey}'] = score
    return compare_dict

def make_small_pool_dict_from_selection_list(pool_dict, selection_list):
    small_pool_dict = {}
    for thisKey in pool_dict:
        if thisKey in selection_list:
            small_pool_dict[thisKey] = pool_dict[thisKey]
    return small_pool_dict
    

def filter_pools(pool_dict):        
    for thisKey in pool_dict:
        thisPool = pool_dict[thisKey]
        
        #look for stacks of 3 cells that are similar to each other and remove them
        indexes = []
        for i in range(len(thisPool)-2):
            for j in range(len(thisPool[0])):
                diff = abs(thisPool[i+1, j] - thisPool[i, j])
                diff += abs(thisPool[i+2, j] - thisPool[i+1, j])
                if diff < 7:
                    indexes.append([i+1, j])
        for index in indexes:
            thisPool[index[0], index[1]] = -80
        
        #remove anything quieter than -60
        for i in range(len(thisPool)):
            for j in range(len(thisPool[0])):
                if thisPool[i,j] < -65:
                    thisPool[i,j] = -80
        
        #find the hottest zone and remove anything outside of it
        #this filter is not necessarily great
        best_score = 0
        best_row = 0
        for i in range(len(thisPool) - 9):
            score = count_active_cells(thisPool[i:i+9, 1:5])
            if score > best_score:
                best_score = score
                best_row = i
        for i in range(len(thisPool)):
            for j in range(len(thisPool[0])):
                if j > 0 and (i < best_row - 5 or i > best_row + 14) :
                    thisPool[i, j] = -80

def filter_one_pool(pool):
    #look for stacks of 3 cells that are similar to each other and remove them
    indexes = []
    for i in range(len(pool)-2):
        for j in range(len(pool[0])):
            diff = abs(pool[i+1, j] - pool[i, j])
            diff += abs(pool[i+2, j] - pool[i+1, j])
            if diff < 7:
                indexes.append([i+1, j])
    for index in indexes:
        pool[index[0], index[1]] = -80

    #remove anything quieter than -60
    for i in range(len(pool)):
        for j in range(len(pool[0])):
            if pool[i,j] < -65:
                pool[i,j] = -80

    #find the hottest zone and remove anything outside of it
    #this filter is not necessarily great
    best_score = 0
    best_row = 0
    for i in range(len(pool) - 9):
        score = count_active_cells(pool[i:i+9, 1:5])
        if score > best_score:
            best_score = score
            best_row = i
    for i in range(len(pool)):
        for j in range(len(pool[0])):
            if j > 0 and (i < best_row - 5 or i > best_row + 14) :
                pool[i, j] = -80
                    
def normalize_pools(pool_dict):
    for thisKey in pool_dict:
        thisPool = pool_dict[thisKey]
        
        max_decibel = np.max(thisPool) + 80
        
        for i in range(len(thisPool)):
            for j in range(len(thisPool[0])):
                thisPool[i, j] += 80
                thisPool[i, j] /= max_decibel
                
def normalize_one_pool(pool):
    max_decibel = np.max(pool) + 80
    for i in range(len(pool)):
        for j in range(len(pool[0])):
            pool[i, j] += 80
            pool[i, j] /= max_decibel

def top_twenty_cells(pool_dict):
    for thisKey in pool_dict:
        thisPool = pool_dict[thisKey]
        percentile_val = np.percentile(thisPool, 93)
        for i in range(len(thisPool)):
            for j in range(len(thisPool[0])):
                
                if thisPool[i, j] < percentile_val:
                    thisPool[i, j] = 0
                
def count_active_cells(array):
    count = 0
    for row in array:
        for cell in row:
            if cell > -80:
                count += 1
    return count
                    
def make_pool_dict(data, raven_table):
    pool_dict = {}
    rows = len(data)
    columns = len(data[0])
    for index, row in raven_table.iterrows():
        #edit this so that excludes post-intro notes in an intelligent way??
        pool_dict[row['Selection']] = post_pool(data, int(floor(row['Begin Time (s)']/time_converter)), rows, columns, 15, 10)

    return pool_dict

def label_table(raven_table, labelled_groups):
    new_table = raven_table.copy()
    for thisKey in labelled_groups.keys():
        for index in labelled_groups[thisKey]:
            new_table.loc[new_table['Selection'] == index, 'ID'] = thisKey
    return new_table

def make_groupings(pool_dict, compare_dict, threshold):
    compare_dict_copy = compare_dict.copy()
    pool_keys = list(pool_dict.keys())
    groupings = []
    loops = 0
    min_val = 0
    while len(pool_keys) > 0 and min_val < threshold:
        
        min_val = min(compare_dict_copy.values())
        min_key = min(zip(compare_dict_copy.values(), compare_dict_copy.keys()))[1]
        compare_dict_copy.pop(min_key)
        key1 = int(min_key.split(':')[0])
        key2 = int(min_key.split(':')[1])
        key1_group = 1000
        key2_group = 1000
        
        #check if either keys are already in a group and which group it is
        for i in range(len(groupings)):
            if key1 in groupings[i]:
                key1_group = i
            if key2 in groupings[i]:
                key2_group = i
                
        #group things according to whether the keys are already sorted
        if key1_group < 1000 and key2_group < 1000 and key1_group != key2_group:
            groupings[key1_group] = groupings[key1_group] + groupings[key2_group]
            groupings.pop(key2_group)
        elif key1_group < 1000 and key2_group == 1000:
            groupings[key1_group].append(key2)
            pool_keys.remove(key2)
        elif key1_group == 1000 and key2_group < 1000:
            groupings[key2_group].append(key1)
            pool_keys.remove(key1)
        elif key1_group == 1000 and key2_group == 1000:
            groupings.append([key1, key2])
            pool_keys.remove(key1)
            pool_keys.remove(key2)
        loops += 1
    return groupings, pool_keys

def label_groupings_by_intro_freq(groupings, raven_table):
    groupings_dict = {}
    for group in groupings:
        intro_freqs = []
        for selection in group:
            row_index = raven_table.index[(raven_table['Selection'] == selection)]
            row = raven_table.iloc[row_index]
            intro_freq = np.mean([row['Low Freq (Hz)'], row['High Freq (Hz)']])
            intro_freqs.append(intro_freq)
        median_freq = np.median(intro_freqs) + selection * .00001
        groupings_dict[median_freq] = group
    return groupings_dict

def relabel_groupings_by_letter(sorted_dict):
    labelled_groups = {}
    letterIndex = 0
    for thisKey in sorted_dict.keys():
        labelled_groups[letter_names[letterIndex]] = sorted_dict[thisKey]
        letterIndex += 1
    return labelled_groups

def group_to_mean_dist(pool_dict, selection_list):
    group_pool_dict = make_small_pool_dict_from_selection_list(pool_dict, selection_list)
    
    group_pool_list = list(group_pool_dict.values())
    mean_pool = np.mean(group_pool_list, axis=0)
    scores = []
    for pool in group_pool_list:
        scores.append(score_two_pools(pool, mean_pool))
    return np.mean(scores)

def make_mean_pool(pool_dict, selection_list):
    group_pool_dict = make_small_pool_dict_from_selection_list(pool_dict, selection_list)
    group_pool_list = list(group_pool_dict.values())
    mean_pool = np.mean(group_pool_list, axis=0)
    return mean_pool

def compare_two_groups(pool_dict, group1_selections, group2_selections, threshold):
    av_group1_dist = group_to_mean_dist(pool_dict, group1_selections)
    av_group2_dist = group_to_mean_dist(pool_dict, group2_selections)
    
    group1_pool_dict = make_small_pool_dict_from_selection_list(pool_dict, group1_selections)
    group1_pools = list(group1_pool_dict.values())
    group2_pool_dict = make_small_pool_dict_from_selection_list(pool_dict, group2_selections)
    group2_pools = list(group2_pool_dict.values())
    
    mean_group1 = np.mean(group1_pools, axis=0)
    mean_group2 = np.mean(group2_pools, axis=0)
    group1_to_group2_dist = score_two_pools(mean_group1, mean_group2)
    collapse = False
    if av_group1_dist + av_group2_dist > group1_to_group2_dist*threshold:
        collapse = True
    
    return {'collapse': collapse, 'average_group1_dist': av_group1_dist, 'average_group2_dist': av_group2_dist, 'group1_to_group2': group1_to_group2_dist}

def collapse_groups(groupings_dict, threshold):
    collapsed_a_group = True
    while collapsed_a_group:
        collapsed_a_group = False
        collapsed_groups = groupings_dict.copy()
        compared_list = []
        for groupKey1 in groupings_dict:
            if groupKey1 not in compared_list:
                compared_list.append(groupKey1)
                for groupKey2 in groupings_dict:
                    if groupKey2 not in compared_list:
                        group_report = compare_two_groups(pool_dict, groupings_dict[groupKey1], groupings_dict[groupKey2], threshold)
                        if group_report['collapse']:
                            collapsed_groups[groupKey1] += collapsed_groups[groupKey2]
                            collapsed_groups.pop(groupKey2)
                            compared_list.append(groupKey2)
                            collapsed_a_group = True
        groupings_dict = collapsed_groups.copy()
    return collapsed_groups

def make_songtype_templates(pool_dict, selection_table):
    letter_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    #make a dictionary where keys are song-type labels and values are averaged pool arrays
    mean_pool_dict = {}
    for letter in letter_names:
        row_indexes = selection_table.index[(selection_table['ID'] == letter) & (selection_table['View'] == 'Spectrogram 1')].tolist()
        if row_indexes:
            selection_list = []
            for index in row_indexes:
                selection_list.append(selection_table.loc[index]['Selection'])
            mean_pool = make_mean_pool(pool_dict, selection_list)
            mean_pool_dict[letter] = mean_pool
    return mean_pool_dict

def add_misfits(pool_dict, labelled_groups, misfits):
    new_labelled_groups = labelled_groups.copy()
    for misfit in misfits:
        misfit_pool = pool_dict[misfit]
        best_score = 1000
        best_label = ''
        for groupKey in new_labelled_groups:
            group_pool_dict = make_small_pool_dict_from_selection_list(pool_dict, new_labelled_groups[groupKey])
            group_pool_list = list(group_pool_dict.values())
            mean_pool = np.mean(group_pool_list, axis=0)
            score = score_two_pools(misfit_pool, mean_pool)
            if score < best_score:
                best_score = score
                best_label = groupKey
        new_labelled_groups[best_label].append(misfit)
    return new_labelled_groups

def categorize_by_templates(pool_dict, mean_pool_dict):
    labelled_groups = {}
    for meanKey in mean_pool_dict:
        labelled_groups[meanKey] = []
    for thisKey in pool_dict:
        best_score = 1000
        best_category = 'Z'
        for meanKey in mean_pool_dict:
            score = score_two_pools(mean_pool_dict[meanKey], pool_dict[thisKey])
            if score < best_score:
                best_score = score
                best_category = meanKey
        labelled_groups[best_category].append(thisKey)
    return labelled_groups

def compare_no_template(raven_name, filename):
    raven_table = pd.read_csv(raven_name + '.txt', sep='\t')
    if os.path.exists(filename + '.h5'):
        with h5py.File(filename + '.h5', 'r') as hf:
            data = hf[filename + '_dataset'][:]
    else:
        data = set_up(filename)
    pool_dict = make_pool_dict(data, raven_table)
    filter_pools(pool_dict)
    normalize_pools(pool_dict)
    compare_dict = make_compare_dict(pool_dict)
    # plt.hist(compare_dict.values(), density=False, bins=40)  # density=False would make counts
    # plt.ylabel('Counts')
    # plt.xlabel('Data')
    return raven_table, pool_dict, compare_dict
    
def label_no_template(raven_table, filename, pool_dict, compare_dict, threshold):
    filter_pools(pool_dict)
    normalize_pools(pool_dict)
    
    #creates initial groupings
    groupings, misfits = make_groupings(pool_dict, compare_dict, threshold)
    groupings_dict = label_groupings_by_intro_freq(groupings, raven_table)
    collapsed_groups = collapse_groups(groupings_dict, 100)
    print('length of groupings: ', len(groupings_dict))
    print('length of collapsed groupings: ', len(collapsed_groups))
    sorted_dict = dict(sorted(collapsed_groups.items()))
    labelled_groups = relabel_groupings_by_letter(sorted_dict)
    new_labelled_groups = add_misfits(pool_dict, labelled_groups, misfits)
    new_table = label_table(raven_table, new_labelled_groups)
    new_table.to_csv('hs_nt_' + filename + '.txt', sep='\t', index=False)
    return 'hs_nt_' + filename + '.txt'

def make_template_file(pool_dict):
    final_table = pd.read_csv('final_labels_' + filename + '.txt', sep='\t')
    mean_pool_dict = make_songtype_templates(pool_dict, final_table)
    with open('templates' + filename + '.pkl', 'wb') as f:
        pickle.dump(mean_pool_dict, f)
        
def label_with_template():
    #prepare data
    with h5py.File(filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    with open('templates' + filename + '.pkl', 'rb') as f:
        mean_pool_dict = pickle.load(f)
    raven_table = pd.read_csv('raven' + filename + '.txt', sep='\t')
    pool_dict = make_pool_dict(data, raven_table)
    filter_pools(pool_dict)
    normalize_pools(pool_dict)
    
    #label based on best match to templates
    labelled_groups = {}
    for meanKey in mean_pool_dict:
        labelled_groups[meanKey] = []
    for thisKey in pool_dict:
        best_score = 1000
        best_category = 'Z'
        for meanKey in mean_pool_dict:
            score = score_two_pools(mean_pool_dict[meanKey], pool_dict[thisKey])
            if score < best_score:
                best_score = score
                best_category = meanKey
        labelled_groups[best_category].append(thisKey)
    new_table = label_table(raven_table, labelled_groups)
    new_table.to_csv('nt_draft_labels_' + filename + '.txt', sep='\t', index=False)

hethsorter_main()