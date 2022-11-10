#INFO TO REMEMBER
#row to actual frequency: (row + 120) * 10.7666 where 120 is the bottom section cut off of the spectrograms
#each pixel/column is 0.023219814 seconds long 

from math import floor
import string
import pandas as pd
import numpy as np
import h5py
import matplotlib.pylab as plt
import seaborn as sns
import os
import csv
import ast
import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


    # FUNCTIONS

# LOAD A RECORDING AND FOURIER TRANSFORM IT
def fourier(file):
    y, sr = librosa.load(file)
    print(f'sr = {sr}Hz')
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(np.shape(S_db))
    S_db = S_db[120:743,:]
    return S_db

# LOAD ARRAY FROM H5 AND CREATE SONG ALBUM
def h5_to_album(file):
    with h5py.File(file + '.h5', 'r') as hf:
        data = hf[file + '_dataset'][:]
    print(np.shape(data))

    os.mkdir(file)
    song_album = []

    i = 0
    rows = len(data)
    columns = len(data[0])
    print(rows)
    print(columns)
    while i < (columns):
        j = 0
        while j < rows:
            # print(array[j][i])
            if i >= columns:
                break
            if (data[j][i] > intro_onset):
                # print(f'checking a new pixel at column: {i}, row: {j}')
                line_dict = check_for_thread_strict(data, j, i, rows, columns, intro_threshold, intro_max, intro_min_length, intro_max_length)
                # print(line_dict)
                if line_dict['length'] > intro_min_length:
                    post_notes = check_for_posts(data, post_threshold, post_onset, i, columns, rows, loud_post_threshold)
                    #cut the song out and add it to album
                    song_album.append(add_song_to_album(data, i, columns, post_notes, line_dict))
                    # if it has enough post_notes, skip the width of a song and continue on
                    if song_album[-1]['label'] == 'Verified':
                        i += 70
                        break
                    else:
                        i += 5
                        break
            j += 1            
        i += 1
    #save a bunch of spectrograms
    print('saving spectrograms...')
    save_images(song_album, file)
    #save the album as a dataframe and then csv
    print('saving csv...')
    save_df(song_album, 'df')
    #compare all songs to each other and save csv

def save_images(song_album, file):
    for i in range(len(song_album)):
        song_name = 'song' + str(i+1)
        save_spect(song_album[i], song_name, file)

def cut_array_into_specs(array, folder: string, length: float):
    os.mkdir(folder)
    length = floor(length * (1/0.023219814))
    columns = len(array[0])
    rows = len(array)
    name_count = 0
    column = 0
    while column < columns:
        print(column)
        song_name = str(round(column * 0.023219814, 2))
        bracket = song_name + ' - ' + str(round((column + length) * 0.023219814, 2))
        print(song_name)
        if ((column + length) < columns):
            chunk = array[:, column:column + length]
            save_spect_from_array(chunk, song_name, folder, length, bracket)
        else:
            new_array = array[:, column:]
            shorter_cols = np.shape(new_array)[1]
            zero_array = np.zeros((623, length))
            zero_array.fill(-80)
            col_diff = length - shorter_cols 
            zero_array[:,:-col_diff] = new_array
            new_array = zero_array
            save_spect_from_array(new_array, song_name, folder, length, bracket)
        column += length
        name_count += 1


def save_df(song_album, dfname):
    datadict = {'intro column': [], 'intro time': [], 'intro freq': [], 'intro length': [],'post size': [], 'soft locs': [], 'loud locs':[]}

    for entry in song_album:
        datadict['intro column'].append(entry['intro column'])
        datadict['intro time'].append(entry['intro time'])
        datadict['intro freq'].append(entry['intro freq'])
        datadict['intro length'].append(entry['intro length'])       
        datadict['soft locs'].append(''.join(','.join(map(str, entry['soft notes']))))
        datadict['loud locs'].append(''.join(','.join(map(str, entry['loud notes']))))
        datadict['post size'].append(len(entry['soft notes']) + len(entry['loud notes']))

    df = pd.DataFrame(datadict)
    df.to_csv(dfname + '.csv')

def check_for_thread(array, row, column, rows, columns, threshold, max_threshold, min_length, stop_length):
    length_count = 0
    best_length = 0
    max_value = -80
    max_row = 0
    max_col = 0
    old_max_value = -80
    old_max_row = 0
    old_max_col = 0
    osc_list = [0, 1, -1]
    def check_for_next(rindex, cindex):
        nonlocal max_row
        nonlocal max_col
        nonlocal max_value
        nonlocal old_max_value
        nonlocal old_max_row
        nonlocal old_max_col
        nonlocal length_count
        nonlocal best_length
        if (cindex - column) < stop_length:
            for test_row in osc_list:
                # print(f'now checking: {rindex+test_row}, {cindex}')
                max_switched = False
                if array[rindex + test_row, cindex] > threshold:
                    # print(f'{array[rindex + test_row, cindex]} is greater than threshold: {threshold}')
                    # print(f'length of thread is: {cindex - column}')
                    if array[rindex + test_row, cindex] > max_value:
                        old_max_value = max_value
                        old_max_row = max_row
                        old_max_col = max_col
                        max_switched = True
                        max_value = array[rindex + test_row, cindex]
                        max_row = rindex + test_row
                        max_col = cindex
                    length_count += 1
                    if length_count > best_length:
                        best_length = length_count
                    if check_for_next(rindex+test_row, cindex + 1):
                        # print(f'length is long enough!!!!!!!!!!!!!!!!!')
                        return True
                    if max_switched:
                        max_value = old_max_value
                        max_row = old_max_row
                        max_col = old_max_col
                    length_count -= 1
            # print(f'checked all 5, no luck, length is: {length_count}')
            return False
        else:
            if max_value > max_threshold:
                return True
            else:
                return False
    
    if (row > (stop_length)) and (row < (rows - stop_length)) and (column < columns - stop_length):
        check_for_next(row, column)
        if best_length > min_length:
            return {'length': best_length, 'onset time': column, 'onset freq': row, 'max onset time': max_col, 'max onset freq': max_row}
        else:
            return {'length': 0}
    else:
        return {'length': 0}

def check_for_thread_strict(array, row, column, rows, columns, threshold, max_threshold, min_length, stop_length):
    length = 1
    line_values = []
    line_freqs = []
    start_row = row
    prev_value = array[row, column]
    line_values.append(prev_value)
    if (row > (2*stop_length)) and (row < (rows - 2*stop_length)) and (column < columns - stop_length):
        while (length < stop_length):
            next_value = -80
            new_row = 0
            for i in [0, -1, 1, -2, 2]:
                if (array[row + i, column + length] > next_value):
                    next_value = array[row + i, column + length]
                    new_row = row + i
            if (array[new_row, column + length] > threshold) and (abs(array[new_row, column + length] - prev_value) < diff_threshold):
                line_values.append(array[new_row, column + length])
                line_freqs.append(new_row)
                prev_value = array[new_row, column + length]
                row = new_row
                length += 1
            else:
                if (length >= min_length) and (max(line_values) > max_threshold):
                    # print('medium thread')
                    # print(line_values)
                    return {'length': length, 'onset time': column, 'onset freq': start_row, 'max db': max(line_values), 'mean freq': round(np.mean(line_freqs), 2)}
                else:
                    # print(f'did not qualify. length: {length}, max value: {max(line_values)}, line values: {line_values}')
                    return {'length': 0}
        if max(line_values) > max_threshold:
            # print('max line!')
            # print(line_values)
            return {'length': length, 'onset time': column, 'onset freq': start_row, 'max db': max(line_values), 'mean freq': round(np.mean(line_freqs), 2)}
        else:
            # print('max value was not above threshold')
            return {'length': 0}
    else:
        # print('Out of Bounds')
        return {'length': 0}

def check_for_posts(array, post_thresh:float, post_onset:float, column:int, columns:int, rows:int, loud_post_threshold:float):
    start = column
    k = column + 10
    soft_notes = []
    loud_notes = []
        # Check for post introductory portion (2 lines somewhere after the intro note)
    while k < (column + 70) and (k < columns - post_min_length):
        l = 0
        while l < rows:
            if (array[l][k] > post_onset) and (l > 50):
                unique = True
                for post_note in loud_notes:    
                    if (l < post_note[0] + 20) and (l > post_note[0] - 20) and ((k - start) < (post_note[1] + post_note[2] + 5)):
                        unique = False    
                if unique:
                    line_dict = check_for_thread_strict(array, l, k, rows, columns, post_thresh, post_max, post_min_length, post_max_length)
                    if (line_dict['length'] >= post_min_length) and (line_dict['max db'] > loud_post_threshold):
                        loud_notes.append([line_dict['onset freq'], line_dict['onset time'] - start, line_dict['length'], line_dict['mean freq']])
                        l += 19
            l += 1
        k += 1
   
    k = column + 10
    while k < (column + 70) and (k < columns - post_min_length):
        l = 0
        while l < rows:
            if (array[l][k] > post_onset) and (l > 50):
                unique = True
                for post_note in soft_notes:
                    if (l < post_note[0] + 10) and (l > post_note[0] - 10) and ((k - start) < (post_note[1] + post_note[2] + 5)):
                        unique = False
                for post_note in loud_notes:    
                    if (l < post_note[0] + 20) and (l > post_note[0] - 20) and ((k - start) < (post_note[1] + post_note[2] + 5)):
                        unique = False    
                if unique:
                    line_dict = check_for_thread_strict(array, l, k, rows, columns, post_thresh, post_max, post_min_length, post_max_length)
                    if (line_dict['length'] >= post_min_length) and (line_dict['max db'] <= loud_post_threshold):
                        soft_notes.append([line_dict['onset freq'], line_dict['onset time'] - start, line_dict['length'], line_dict['mean freq']])
                        l += 5
                    elif (line_dict['length'] >= post_min_length) and (line_dict['max db'] > loud_post_threshold):
                        loud_notes.append([line_dict['onset freq'], line_dict['onset time'] - start, line_dict['length'], line_dict['mean freq']])
                        l += 5
            l += 1
        k += 1
    if len(soft_notes) < 1:
        soft_notes.append([0,0,0,0])
    if len(loud_notes) < 1:
        loud_notes.append([0,0,0,0])
    return {'soft notes': soft_notes, 'loud notes': loud_notes}

def add_song_to_album(array, column, columns, post_notes, line_dict):
    if len(post_notes) > 1: 
        label = 'Verified'
        # a song has been identified and is now added to an array
    else:
        label = 'Not Enough Post Notes'
    if column+70 < columns:
        new_array = array[:,column:column+70]
    else:
        # add columns to a clipped song so that it fits with the others
        new_array = array[:, column:]
        shorter_cols = np.shape(new_array)[1]
        zero_array = np.zeros((623, 70))
        zero_array.fill(-80)
        col_diff = 70 - shorter_cols 
        zero_array[:,:-col_diff] = new_array
        new_array = zero_array
    # store the song
    return {"intro column": line_dict['onset time'], "intro time": round((line_dict['onset time']) * 0.023219814, 1), "intro freq": line_dict['onset freq'], "intro length":line_dict['length'], "array": new_array, "label": label, "soft notes": post_notes['soft notes'], "loud notes": post_notes['loud notes']}

# CREATE AND DISPLAY SPECTROGRAM
def display_spect(array):
    fig, ax = plt.subplots(figsize=(15, 7))
    # img = librosa.display.specshow(array[1], x_axis='time', y_axis=None, sr=22050, ax=ax)
    img = librosa.display.specshow(array, x_axis='time', y_axis=None, sr=22050, ax=ax)
    ax.set_title('Spectrogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 743-120, 50))
    fig.gca().set_ylabel("Row")
    plt.show()

def save_spect(dict, name, folder):
    fig, ax = plt.subplots(figsize=(15, 7))
    img = librosa.display.specshow(dict["array"], x_axis=None, y_axis=None, sr=22050, ax=ax)
    ax.set_title(str(dict["intro time"]) + ' seconds' + ' ' + dict["label"] + ':' + str(dict['intro column']), fontsize=20)
    ax.tick_params(direction='out', labelsize='medium', width=3, grid_alpha=0.9)
    ax.grid(True, linestyle='-.')
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 743-120, 25))
    fig.gca().set_ylabel("Row")
    fig.gca().set_xticks(range(0, 70, 5))
    directory = folder + '/' + name
    plt.savefig(directory)
    plt.close()

def save_spect_from_array(array, name, folder, length, bracket):
    fig, ax = plt.subplots(figsize=(15, 7))
    img = librosa.display.specshow(array, x_axis=None, y_axis=None, sr=22050, ax=ax)
    ax.set_title(bracket, fontsize=20)
    ax.tick_params(direction='out', labelsize='medium', width=3, grid_alpha=0.9)
    ax.grid(True, linestyle='-.')
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 743-120, 25))
    fig.gca().set_ylabel("Row")
    fig.gca().set_xticks(range(0, length, floor(length/10)))
    directory = folder + '/' + str(floor(float(name)))
    plt.savefig(directory)
    plt.close()

def save_whole_spect(array, name):
    fig, ax = plt.subplots(figsize=(15, 7))
    img = librosa.display.specshow(array, x_axis='time', y_axis=None, sr=22050, ax=ax)
    ax.set_title(str(array), fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 743-120, 50))
    fig.gca().set_ylabel("Row")
    plt.savefig(name)
    plt.close()

# CUT A SEGMENT, STORE IT AS AN ARRAY
def make_segment(array, start, duration):
    column_start = int(np.floor(start*43.06666667))
    column_duration = int(np.floor(duration*43.06666667))
    new_array = array[:,column_start:column_start + column_duration]
    return new_array

# CUT A SEGMENT, STORE IT AS AN H5 FILE
def store_an_array(file, start, duration):
    data = fourier(file + '.wav')
    new_seg = make_segment(data, start, duration)
    print(np.shape(new_seg))
    with h5py.File(str(duration) + 'seconds' + '.h5', 'w') as hf:
        hf.create_dataset(str(duration) + 'seconds' + "_dataset", data=new_seg)




# CHECK NUMBER OF POST MATCHES BETWEEN SONGS
    # song_album.append({"intro time": round((column-6) * 0.023219814, 1), "intro freq": row, "array": new_array, "label": label, "post notes": post_notes})

def count_matches(album):
    match_dict = {}
    for target_song in album:
        print(f'TARGET SONG: {target_song}')
        target_name = str(target_song['intro freq']) + ';' + str(target_song['intro time'])
        match_dict[target_name] = {}
        for compare_song in album:
            compare_name = str(compare_song['intro freq']) + ';' + str(compare_song['intro time'])
            #figure out which one has the most lines and how many
            total_notes_1 = len(target_song['post locs'])
            total_notes_2 = len(compare_song['post locs'])
            max_songs = max(total_notes_1, total_notes_2)
            best_score = 0
            #this the list of frames to shift left or right to find the best matches
            for i in [-5, 0, 5]:
                matches = 0
                for target_note in target_song['post locs']:
                    for compare_note in compare_song['post locs']:
                        if (abs(target_note[0] - compare_note[0]) < 15) and (abs((target_note[1]+i) - compare_note[1]) < 9):
                            matches += 1
                if matches > best_score:
                    best_score = matches                
            if (target_name == compare_name):
                match_dict[target_name][compare_name] = 0
            else:
                match_dict[target_name][compare_name] = round(best_score/max_songs, 2)
    print(match_dict)

    match_df = pd.DataFrame(match_dict)
    print(match_df.shape)
    match_df.to_csv('match_df.csv')


def new_count_matches(album):
    match_dict = {}
    for target_song in album:
        target_name = str(target_song['intro freq']) + ';' + str(target_song['intro time'])
        match_dict[target_name] = {}
        for compare_song in album:
            compare_name = str(compare_song['intro freq']) + ';' + str(compare_song['intro time'])
            #figure out which one has the most lines and how many
            smaller_song = []
            larger_song = []
            if len(target_song['post locs']) >= len(compare_song['post locs']):
                larger_song_backup = target_song['post locs']
                smaller_song = compare_song['post locs']
            else:
                smaller_song = target_song['post locs']
                larger_song_backup = compare_song['post locs']
            less_songs = len(smaller_song)
            best_score = 0
            # distance_sum = []
            #this the list of frames to shift left or right to find the best matches
            for i in [-5, 0, 5]:
                larger_song = larger_song_backup.copy()
                matches = 0
                # distance_list = []
                for target_note in smaller_song:
                    for compare_note in larger_song:
                        if (abs(target_note[0] - compare_note[0]) < 15) and (abs((target_note[1]+i) - compare_note[1]) < 10):
                            matches += 1
                            # larger_song.remove(compare_note)
                            # distance = (abs(target_note[0] - compare_note[0])) + (abs(target_note[1] + i - compare_note[1]) + (abs(target_note[2] - compare_note[2])))
                            # distance_list.append(distance)
                            break
                if matches > best_score:
                    best_score = matches
                    # distance_sum = sum(distance_list)                
            if (target_name == compare_name):
                match_dict[target_name][compare_name] = 0
            else:
                match_dict[target_name][compare_name] = round(best_score/less_songs, 2) #distance_sum]
                # print(distance_list)

    match_df = pd.DataFrame(match_dict)
    print(match_df.shape)
    match_df.to_csv('match_df.csv')

def count_both_matches(album):
    match_dict = {}
    for target_song in album:
        target_name = str(target_song['intro freq']) + ';' + str(target_song['intro time'])
        match_dict[target_name] = {}
        for compare_song in album:
            compare_name = str(compare_song['intro freq']) + ';' + str(compare_song['intro time'])
            #figure out which one has the most lines and how many
            smaller_softs = []
            smaller_louds = []
            larger_softs = []
            larger_louds = []
            if 3*len(target_song['loud locs']) + len(target_song['soft locs']) >= 3*len(compare_song['loud locs']) + len(compare_song['soft locs']):
                larger_softs_backup = target_song['soft locs']
                smaller_louds_backup = compare_song['loud locs']
                larger_louds_backup = target_song['loud locs']
                smaller_softs_backup = compare_song['soft locs']
            else:
                smaller_softs_backup = target_song['soft locs']
                larger_softs_backup = compare_song['soft locs']
                smaller_louds_backup = target_song['loud locs']
                larger_louds_backup = compare_song['loud locs']
            best_possible_score = (len(smaller_louds_backup) * 3) + len(smaller_softs_backup)
            best_score = 0
            #this the list of frames to shift left or right to find the best matches
            for i in [0]:
                smaller_louds = smaller_louds_backup.copy()
                larger_louds = larger_louds_backup.copy()
                larger_softs = larger_softs_backup.copy()
                smaller_softs = smaller_softs_backup.copy()
                score = 0
                for target_note in smaller_louds:
                    for compare_note in larger_louds:
                        if (abs(target_note[3] - compare_note[3]) < 10) and (abs((target_note[1]+i) - compare_note[1]) < 6):
                            score += 3
                            break
                        elif (abs(target_note[3] - compare_note[3]) < 20) and (abs((target_note[1]+i) - compare_note[1]) < 8):
                            score += 2.5
                            break
                        elif (abs(target_note[3] - compare_note[3]) < 30) and (abs((target_note[1]+i) - compare_note[1]) < 10):
                            score += 2
                            break
                    else:
                        for compare_note in larger_softs:
                            if (abs(target_note[3] - compare_note[3]) < 10) and (abs((target_note[1]+i) - compare_note[1]) < 6):
                                score += 1
                                break
                            elif (abs(target_note[3] - compare_note[3]) < 20) and (abs((target_note[1]+i) - compare_note[1]) < 8):
                                score += .75
                                break
                            elif (abs(target_note[3] - compare_note[3]) < 30) and (abs((target_note[1]+i) - compare_note[1]) < 10):
                                score += .5
                                break
                for target_note in smaller_softs:
                    for compare_note in larger_louds:
                        if (abs(target_note[3] - compare_note[3]) < 10) and (abs((target_note[1]+i) - compare_note[1]) < 6):
                            score += 1
                            break
                        elif (abs(target_note[3] - compare_note[3]) < 20) and (abs((target_note[1]+i) - compare_note[1]) < 8):
                            score += .75
                            break
                        elif (abs(target_note[3] - compare_note[3]) < 30) and (abs((target_note[1]+i) - compare_note[1]) < 10):
                            score += .5
                            break
                    else:
                        for compare_note in larger_softs:
                            if (abs(target_note[3] - compare_note[3]) < 10) and (abs((target_note[1]+i) - compare_note[1]) < 6):
                                score += 1
                                break
                            elif (abs(target_note[3] - compare_note[3]) < 20) and (abs((target_note[1]+i) - compare_note[1]) < 8):
                                score += .75
                                break
                            elif (abs(target_note[3] - compare_note[3]) < 30) and (abs((target_note[1]+i) - compare_note[1]) < 10):
                                score += .5
                                break
                if score > best_score:
                    best_score = score              
            if (target_name == compare_name):
                match_dict[target_name][compare_name] = 0
            else:
                match_dict[target_name][compare_name] = round(best_score/best_possible_score - 0.01 * abs(target_song['post size'] - compare_song['post size']), 5)

    match_df = pd.DataFrame(match_dict)
    print(match_df.shape)
    match_df.to_csv('match_df.csv')

def new_sort_songs(dict):
    threshold = match_threshold
    song_types = {}
    #stage 1: sort each song in a group with it's closest match if it is above threshold
    for song in dict:
        print(song)
        max_match_value = 0
        max_match = ''
        #find the best match for that song
        for second_song in dict[song]:
            if dict[song][second_song] > max_match_value:
                max_match_value = dict[song][second_song]
                max_match = second_song
        #check if the best match is below threshold and if so, create a new category for it
        print(max_match_value)
        if (max_match_value <= threshold):
            song_types[len(song_types) + 1] = []
            song_types[len(song_types)].append(song)          
        else:
            #Check if song or max_match have already been sorted
            song_sorted = False
            max_sorted = False
            for category in song_types:
                if (song in song_types[category]):
                    song_sorted = True
                if (max_match in song_types[category]):
                    max_sorted = True
            #Check each combination of the two songs being sorted or not and append accordingly
            if song_sorted and max_sorted:
                # print(f'{song} and {max_match} already sorted')
                continue
            elif (not song_sorted) and (not max_sorted):
                song_types[len(song_types) + 1] = []
                song_types[len(song_types)].append(song)
                song_types[len(song_types)].append(max_match)
                print(f'just sorted {song} into {len(song_types)}')
            elif (not song_sorted) and max_sorted:
                # print(f'sorting just {song} because {max_sorted} already sorted')
                for category in song_types:
                    if max_match in song_types[category]:
                        song_types[category].append(song)
                        break
            else:
                for category in song_types:
                    if song in song_types[category]:
                        song_types[category].append(max_match)
                        break
    print(f'stage one: {song_types}')

    #stage 2: if any two groups belong together, collapse them together
    found_match = True
    while found_match:
        found_match = False
        for target_category in song_types:
            best_score = 0
            best_category = ''
            score_within = 0
            score_between = 0
            if (song_types[target_category] != 'delete') and (len(song_types[target_category]) > 1):
                #calculate the score_within
                score_list = []
                for target_song in song_types[target_category]:
                    for other_target_song in song_types[target_category]:
                        if target_song != other_target_song:
                            score_list.append(dict[target_song][other_target_song])
                score_within = np.mean(score_list)
                for compare_category in song_types:
                    if song_types[compare_category] != 'delete' and (len(song_types[compare_category]) > 1):
                        if target_category != compare_category:
                            #calculate the score between
                            score_list = []
                            for target_song in song_types[target_category]:
                                for compare_song in song_types[compare_category]:
                                    if target_song != compare_song:
                                        score_list.append(dict[target_song][compare_song])
                            score_between = np.mean(score_list)
                            #check if score is the best
                            if score_between > best_score:
                                best_score = score_between
                                best_category = compare_category
            #check if the scores are similar
            print(best_category)
            print(type(best_category))
            if (best_score > 0) and (abs(1 - (score_within/best_score)) < clumping_threshold):
                for song in song_types[best_category]:
                    song_types[target_category].append(song)
                song_types[best_category] = 'delete'
                found_match = True
    for category in list(song_types):
        if song_types[category] == 'delete':
            song_types.pop(category)
    for song_type in song_types:
        print(f'{song_type}: {song_types[song_type]}')
    return song_types

def relabel_song_types(dict):
    new_dict = {}
    letter_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    while (len(new_dict) < len(dict)):
        lowest_cat = 0
        lowest_freq = 15000
        for category in dict:
            for song in dict[category]:
                if dict[category] != 'delete':
                    if int(song.split(';')[0]) < lowest_freq:
                        lowest_freq = int(song.split(';')[0])
                        lowest_cat = category
        index = len(new_dict)
        new_dict[letter_names[index]] = dict[lowest_cat]
        dict[lowest_cat] = 'delete'
    for dictpart in new_dict:
        print(f'{dictpart}: {new_dict[dictpart]}')
    return new_dict
            
def make_selection_table(dicty):
    dict = dicty.copy()
    sel_table = []
    any_left = True
    while any_left:
        lowest_time = 9876543210
        lowest_freq = ''
        lowest_cat = ''
        for category in dict:
            for song in dict[category]:
                if dict[category] != 'delete':
                    if float(song.split(';')[1]) < lowest_time:
                        lowest_time = float(song.split(';')[1])
                        lowest_cat = category
                        lowest_freq = int(song.split(';')[0])
        sel_table.append({'time': lowest_time, 'freq': lowest_freq, 'type':lowest_cat})
        dict[lowest_cat].remove(str(lowest_freq) + ';' + str(lowest_time))
        any_left = False
        for category in dict:
            if len(dict[category]) > 0:
                any_left = True
    print(sel_table)
    df = pd.DataFrame(sel_table)
    print(df)
    df.to_csv('new_selection_table.csv')

def create_song_album_from_df(dicty_list, filename: string):
    with h5py.File(filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    song_album = []
    # os.mkdir('new_' + filename)
    i = dicty_list[0]['intro column']
    rows = len(data)
    for dicty in dicty_list:
        columns = dicty['intro column'] + 80
        i = dicty['intro column']
        while i < (dicty['intro column'] + 10):
            j = 0
            while j < rows:
                if i >= dicty['intro column'] + 10:
                    break
                if (data[j][i] > intro_onset):
                    # print(f'checking a new pixel at column: {i}')
                    line_dict = check_for_thread_strict(data, j, i, rows, columns, intro_threshold, intro_max, intro_min_length, intro_max_length)
                    if line_dict['length'] > intro_min_length:
                        post_notes = check_for_posts(data, post_threshold, post_onset, i, columns, rows, loud_post_threshold)
                        #cut the song out and add it to album
                        song_album.append(add_song_to_album(data, i, columns, post_notes, line_dict))
                        # if it has enough post_notes, skip the width of a song and continue on
                        if song_album[-1]['label'] == 'Verified':
                            i += 70
                            break
                        else:
                            i += 5
                            break
                j += 1            
            i += 1   
        #save a bunch of spectrograms
    # save_images(song_album, 'new_' + filename)
    #save the album as a dataframe and then csv
    save_df(song_album, 'new_df')
    #compare all songs to each other and save csv

def load_df(file:string):
    df = pd.read_csv(file)
    dicty_list = df.to_dict('records')
    for dictionary in dicty_list:
        dictionary.pop('Unnamed: 0')

        print(dictionary['soft locs'])
        print(type(dictionary['soft locs']))
        str1 = '[' + dictionary['soft locs'] + ']'
        new_list1 = ast.literal_eval(str1)
        print(f'after ast.literl...type is now... {type(new_list1)}')
        dictionary['soft locs'] = new_list1

        print(dictionary['loud locs'])
        print(type(dictionary['loud locs']))
        str2 = '[' + dictionary['loud locs'] + ']'
        new_list2 = ast.literal_eval(str2)
        dictionary['loud locs'] = new_list2
    # print(dicty_list)

    return dicty_list

def load_match_df(file:string):
    df = pd.read_csv(file)
    dicty = df.to_dict('records')
    for dictionary in dicty:
        dictionary.pop('Unnamed: 0')

    key_list = (list(dicty[0]))
    new_dict = {}
    for i in range(len(dicty)):
        new_dict[key_list[i]] = dicty[i]
    return new_dict

    # CODE FOR FIGURING OUT THE FREQUENCIES OF EACH ROW
# n_fft = 2048
# Fs = 22050
# freqs = np.arange(0, 1 + n_fft / 2) * Fs / n_fft
# print(np.shape(freqs))
# print(freqs)
# df = pd.DataFrame(freqs)
# df.to_csv('freqlist.csv')


# CODE TO RUN

    # VARIABLES
filename = 'July 9 Chickahominy 2 80-90'

intro_onset = -45
intro_threshold = -40
intro_max = -30
intro_min_length = 11
intro_max_length = 17
intro_jumps = 1
diff_threshold = 15

post_onset = -50
post_threshold = -45
post_max = -35
post_min_length = 5
post_max_length = 15
post_jumps = 1
loud_post_threshold = -25

clumping_threshold = 0.05 #this is the threshold for combining ST categories based on a ratio of average match scores
match_threshold = 0.8 #this is the matchscore cutoff for deciding whether a ST gets its own category




    # MAKE A SEGMENT AND SAVE IT AS A FILE




    # SAVE DF TO CSV
# df = pd.DataFrame(new_seg)
# df.to_csv('tensec.csv')


    # SAVE ARRAY TO H5 FILE
# with h5py.File('test.h5', 'w') as hf:
#     hf.create_dataset("name_of_dataset", data=S_db)

    # SAVE THE WHOLE SPECTROGRAM OF AN ARRAY
# save_whole_spect(data, 'twentysecfig')

    # LOAD H5 TO NP ARRAY and DISPLAY A SEGMENT
# with h5py.File(filename + '.h5', 'r') as hf:
#     data = hf[filename + '_dataset'][:]
# print(np.shape(data))
# data = make_segment(data, 433, 1)
# display_spect(data)

    #TEST NEW FUNC ON ONE LITTLE SEGMENT
# data = make_segment(data, 90.5, 2)
# print(np.shape(data))
# for col in range(len(data[0]) - 15):
#     list = []
#     for row in range(len(data)-20):
#         if data[row+10, col] > -45:
#             print(f'STARTING AT: {row+10},{col}')
#             list = check_for_thread(data, row + 10, col, -45, -45, 6, stop_length = 9)
#             print(list)

# df = pd.DataFrame(data)
# df.to_csv('newfunc.csv')
# display_spect(data)
            
            


# seg = make_segment(data, 89, 5)

    # SAVE DF TO CSV
# df = pd.DataFrame(seg)
# df.to_csv('1test.csv')
# display_spect(seg)

# csv_filename = 'match_df.csv'
# with open(csv_filename) as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         print(f'this is a new row: {row}')

# store_an_array('tenmin', 177, 120)
# new_seg = make_segment(data, 170, 140)









# ALL NEEDED CODE BELONGS BELOW HERE IN ORDER

#STORE THE WAV AS AN H5 FILE. BE AWARE, FOURIER SLICES OFF THE BOTTOM AND TOP
# data = fourier(filename + '.wav')
# print(np.shape(data))
# with h5py.File(filename + '.h5', 'w') as hf:
#     hf.create_dataset(filename + "_dataset", data=data)


# #CUT ARRAY INTO CHUNKS
# with h5py.File(filename + '.h5', 'r') as hf:
#     data = hf[filename + '_dataset'][:]
# cut_array_into_specs(data, filename+'_chunks', 20)


#FIRST PASS
# h5_to_album(filename)


#SECOND PASS
# dicty_list = load_df('df.csv')
# create_song_album_from_df(dicty_list, filename)

# dicty_list = load_df('new_df.csv')
# count_both_matches(dicty_list)

match_dicty = load_match_df('match_df.csv')
song_types = new_sort_songs(match_dicty)
song_types = relabel_song_types(song_types)
make_selection_table(song_types)