import pandas as pd
import numpy as np
import h5py
import matplotlib.pylab as plt
import seaborn as sns
import os
import csv

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


# CUT UP RECORDINGS INTO SONGS BASED ON MAX DECIBELS IN A COLUMN
# def make_song_list(array):
#     song_album = []
#     i = 0
#     j = 0
#     loudest_row = 0
#     prev_loudest_row = 0
#     score = 0
#     max_dec = -80
#     rows = len(array)
#     columns = len(array[0])
#     print(rows)
#     print(columns)
#     while i < columns:
#         while j < rows:
#             # print(array[j][i])
#             if array[j][i] > intro_threshold:
#                 if array[j][i] > max_dec:
#                     max_dec = array[j][i]
#                     loudest_row = j
#             j += 1
#             # check if the current loudest row matches the last column's loudest row
#         if (loudest_row > 0) and (loudest_row < prev_loudest_row + 2) and (loudest_row > prev_loudest_row - 2):
#             score += 1
#         else:
#             score = 0
#         # reset variables for next column
#         prev_loudest_row = loudest_row
#         max_dec = -80
#         loudest_row = 0
#         j = 0
            
#         if score > 4:
#             # a song has been identified and is now added to an array
#             print(f'the intro note ends at row: {loudest_row} and column: {i}')
#             if i+50 < columns:
#                 new_array = array[:,i-6:i+50]
#             else:
#                 # add columns to a clipped song so that it fits with the others
#                 new_array = array[:, i-6:]
#                 shorter_cols = np.shape(new_array)[1]
#                 zero_array = np.zeros((576, 56))
#                 zero_array.fill(-80)
#                 col_diff = 56 - shorter_cols
#                 zero_array[:,:-col_diff] = new_array
#                 new_array = zero_array
#             # store the song
#             song_album.append([round((i-6) * 0.023219814, 1), new_array])
#             # skip the width of a song and continue on
#             i += 60
#         else:
#             # carry on to the next column
#             i += 1
#     return song_album

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
                print(f'checking a new pixel at column: {i}')
                line_dict = check_for_thread(data, j, i, rows, columns, intro_threshold, intro_max, intro_min_length, intro_max_length)
                if line_dict['length'] > intro_min_length:
                    post_notes = check_for_posts(data, post_threshold, post_onset, i, columns, rows)
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
    for i in range(len(song_album)):
        song_name = 'song' + str(i+1)
        save_spect(song_album[i], song_name, file)

    #save the album as a dataframe and then csv
    datadict = {'intro times': [], 'intro freqs': [], 'intro lengths': [], 'post locs': []}

    for entry in song_album:
        datadict['intro times'].append(entry['intro time'])
        datadict['intro freqs'].append(entry['intro freq'])
        datadict['intro lengths'].append(entry['intro length'])       
        datadict['post locs'].append(''.join(''.join(map(str, entry['post notes']))))
        
    df = pd.DataFrame(datadict)
    df.to_csv('df.csv')
    count_matches(song_album)

    #CHECK IF THERE IS A LINE AT THIS LOCATION
def check_for_a_line(array, row_index: int, column_index: int, rows:int, columns:int, min_threshold:int, max_threshold:int, min_length:int, max_length:int, jump_limit:int):
    jumps = 0
    value = array[row_index][column_index]
    if (row_index > (max_length * 2)) and (row_index < rows - (max_length * 2)) and (column_index < columns - max_length):
        value_list = []
        length = 0
        onset_freq = 0
        for m in range(max_length):
            not_a_max = 0
            for n in range(5):
                value = array[row_index+n-2][column_index+m]
                if value < min_threshold:
                    not_a_max += 1
                    continue
                # check if anything within two rows is larger
                for k in range(5):
                    if array[row_index+n+k-2][column_index+m] > value:
                        not_a_max += 1
                        break
                # check if we did not find anything larger:
                if (not_a_max == n):
                    value_list.append(array[row_index+n-2][column_index+m])
                    if abs((row_index+n-2) - row_index) > 1:
                        jumps += 1
                    row_index = row_index+n-2
                    length += 1
                    if onset_freq == 0:
                        onset_freq = row_index
                    break
            if ((not_a_max > 4) and (length < min_length)) or (jumps > jump_limit):
                return {'length': 0}
        for value in value_list:
            if value > max_threshold:
                return {'length': length, 'onset time': column_index, 'onset freq': onset_freq}
    return {'length': 0}

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

def check_for_posts(array, post_thresh:float, post_onset:float, column:int, columns:int, rows:int):
    start = column
    k = column + 10
    post_notes = []
        # Check for post introductory portion (2 lines somewhere after the intro note)
    while k < (column + 70) and (k < columns - 10):
        l = 0
        while l < rows:
            if (array[l][k] > post_onset):
                unique = True
                for post_note in post_notes:
                    if (l < post_note[0] + 20) and (l > post_note[0] - 20) and (k - start < post_note[1] + 10):
                        unique = False
                if unique:
                    line_dict = check_for_thread(array, l, k, rows, columns, post_thresh, post_max, post_min_length, post_max_length)
                    if line_dict['length'] >= post_min_length:
                        post_notes.append([line_dict['onset freq'], line_dict['onset time'] - start, line_dict['length']])
                        l += 9
            l += 1
        k += 1
    return post_notes

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
    print(line_dict)
    return {"intro time": round((line_dict['onset time']) * 0.023219814, 1), "intro freq": line_dict['onset freq'], "intro length":line_dict['length'], "array": new_array, "label": label, "post notes": post_notes}

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
    img = librosa.display.specshow(dict["array"], x_axis='time', y_axis=None, sr=22050, ax=ax)
    ax.set_title(str(dict["intro time"]) + ' seconds' + ' ' + dict["label"], fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 743-120, 50))
    fig.gca().set_ylabel("Row")
    directory = folder + '/' + name
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
        target_name = str(target_song['intro freq']) + ';' + str(target_song['intro time'])
        match_dict[target_name] = {}
        for compare_song in album:
            compare_name = str(compare_song['intro freq']) + ';' + str(compare_song['intro time'])
            print(f'comparing {target_name} to {compare_name}')
            #figure out which one has the most lines and how many
            total_notes_1 = len(target_song['post notes'])
            total_notes_2 = len(compare_song['post notes'])
            max_songs = max(total_notes_1, total_notes_2)
            best_score = 0
            #this the list of frames to shift left or right to find the best matches
            for i in [-10, 0, 10]:
                matches = 0
                for target_note in target_song['post notes']:
                    for compare_note in compare_song['post notes']:
                        if (abs(target_note[0] - compare_note[0]) < 15) and (abs((target_note[1]+i) - compare_note[1]) < 5):
                            matches += 1
                if matches > best_score:
                    best_score = matches                
            if ((compare_name in match_dict) and (target_name in match_dict[compare_name])) or (target_name == compare_name):
                pass
            else:
                match_dict[target_name][compare_name] = round(best_score/max_songs, 2)
    print(match_dict)
    match_df = pd.DataFrame(match_dict)
    print(match_df.shape)
    match_df.to_csv('match_df.csv')
    sort_songs(match_dict)

def sort_songs(dict):
    threshold = 0.5
    song_types = {}
    total_values = 100
    while(total_values > 1):
        total_values = 0
        for inner_dict in dict:
            total_values += len(dict[inner_dict])
        print(f'total dict values: {total_values}')
        print(f'length of song_types: {len(song_types)}')
        #Find the best match between two songs
        max = -1
        max_target = ''
        max_compare = ''
        for target in dict:
            for compare in dict[target]:
                if dict[target][compare] > max:
                    max_target = target
                    max_compare = compare
                    max = dict[target][compare]
        if len(song_types) > 0:
            #check if they've both been sorted. If so, pop them and move on
            for song_type in song_types:
                if max_target in song_types[song_type]:
                    for song_type_B in song_types:
                        if max_compare in song_types[song_type_B]:
                            print(f'1 popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
                            dict[max_target].pop(max_compare)
                            break
                    else:
                        #so only max_target has already been sorted, so add max_compare if it's above threshold
                        if max > threshold:
                            song_types[song_type].append(max_compare)
                            print(f'2 popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
                            dict[max_target].pop(max_compare)
                            break
                        else:
                            #if below threshold add max_compare to a new category
                            length = len(song_types)
                            song_types[length + 1] = [max_compare]
                            print(f'3 popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
                            dict[max_target].pop(max_compare)
                            break
                    break
            else: #max_target is not already sorted
                for song_type in song_types:
                    if max_compare in song_types[song_type]:
                        if max > threshold:
                            song_types[song_type].append(max_target)
                            print(f'4 popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
                            dict[max_target].pop(max_compare)
                            break
                        else:
                            length = len(song_types)
                            song_types[length + 1] = [max_target]
                            print(f'5 popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
                            dict[max_target].pop(max_compare)
                            break
                else: #neither have been sorted yet
                    if max > threshold:
                        #check if max_target has a match with anything already sorted. If so, put both in the best category
                        break_out = False
                        best_match = 0
                        best_match_type = ''
                        for song_type in song_types:
                            for song in song_types[song_type]:
                                if (song in dict[max_target]) and (dict[max_target][song] > threshold):
                                    if dict[max_target][song] > best_match:
                                        best_match = dict[max_target][song]
                                        best_match_type = song_type
                        if best_match > 0:
                            song_types[best_match_type].append(max_target)
                            song_types[best_match_type].append(max_compare)
                            print(f'5a popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
                            dict[max_target].pop(max_compare)
                            break_out = True
                        if break_out:
                            continue
                        #check if max_compare has a match with anything already sorted. If so, put bot in the best category
                        best_match = 0
                        best_match_type = ''
                        for song_type in song_types:
                            for song in song_types[song_type]:
                                if (song in dict[max_compare]) and (dict[max_compare][song] > threshold):
                                    if dict[max_compare][song] > best_match:
                                        best_match = dict[max_compare][song]
                                        best_match_type = song_type
                        if best_match > 0:
                            song_types[best_match_type].append(max_target)
                            song_types[best_match_type].append(max_compare)
                            print(f'5b popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
                            dict[max_target].pop(max_compare)
                            break_out = True
                        if break_out:
                            continue
                        length = len(song_types)
                        song_types[length + 1] = [max_target]
                        song_types[length + 1].append(max_compare)
                        print(f'6 popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
                        dict[max_target].pop(max_compare)
                    else:
                        length = len(song_types)
                        song_types[length + 1] = [max_target]
                        song_types[length + 2] = [max_compare]
                        print(f'7 popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
                        dict[max_target].pop(max_compare)            
        else:
            song_types[1] = [max_target, max_compare]
            print(f'8 popping {max_target}:{max_compare}:{dict[max_target][max_compare]}')
            dict[max_target].pop(max_compare)
    print(song_types)



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
filename = 'twominute'

intro_onset = -60
intro_threshold = -55
intro_max = -50
intro_min_length = 10
intro_max_length = 15
intro_jumps = 1

post_onset = -60
post_threshold = -60
post_max = -55
post_min_length = 5
post_max_length = 12
post_jumps = 1
# store_an_array('tenmin', 177, 120)

    # MAKE A SEGMENT AND SAVE IT AS A FILE
# data = fourier('tenmin.wav')


# new_seg = make_segment(data, 170, 140)
# print(np.shape(new_seg))
# with h5py.File('twominute.h5', 'w') as hf:
#     hf.create_dataset("twominute_dataset", data=new_seg)



    # SAVE DF TO CSV
# df = pd.DataFrame(new_seg)
# df.to_csv('tensec.csv')


    # SAVE ARRAY TO H5 FILE
# with h5py.File('test.h5', 'w') as hf:
#     hf.create_dataset("name_of_dataset", data=S_db)

    # SAVE THE WHOLE SPECTROGRAM OF AN ARRAY
# save_whole_spect(data, 'twentysecfig')

    # LOAD H5 TO NP ARRAY
# with h5py.File(filename + '.h5', 'r') as hf:
#     data = hf[filename + '_dataset'][:]
# print(np.shape(data))
# data = make_segment(data, 128, 4)
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

h5_to_album(filename)

