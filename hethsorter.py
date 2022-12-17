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
import ast
import librosa
import librosa.display
import IPython.display as ipd
import shutil
from pydub import AudioSegment
import statistics
from PIL import Image as im
import tensorflow as tf
import datetime

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


    # VARIABLES
recording_name = 'A-songs-Log3-HERMIT4_20220502_063611-s'
directory = recording_name + '/'
filename = 'A-songs-Log3-HERMIT4_20220502_063611-s-10m-20m' 

intro_max = -30
intro_onset = intro_max - 10
intro_threshold = intro_max - 10
intro_min_length = 8
intro_max_length = 25
intro_jumps = 1
diff_threshold = 25 #this is how far one value on a thread can jump up or down to the next value
splash_threshold = 20 #how far below the average intro note value does the outer cloud need to be? If the cloud is above this, it trips the splash function


post_max = intro_max - 5
post_onset = post_max - 10
post_threshold = post_max - 10
post_min_length = 5
post_max_length = 15
post_jumps = 1
loud_post_threshold = post_max + 20



clumping_threshold = 0.1 #this is the threshold for combining ST categories based on a ratio of average match scores
match_threshold = 0.6 #this is the matchscore cutoff for deciding whether a ST gets its own category

time_converter = 0.023219814

    # FUNCTIONS

# LOAD A RECORDING AND FOURIER TRANSFORM IT
def fourier(filename):
    y, sr = librosa.load(filename + '.wav')
    # print(f'max of y is: {np.max(y)}')
    # print(f'sr = {sr}Hz')
    D = librosa.stft(y)
    # print(f'max of D is: {np.max(D)}')
    D = D[120:743,:]
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # for row_int in range(len(S_db)):
    #     for col_int in range(len(S_db[0])):
    #         if S_db[row_int, col_int] == 0.0:
    #             print(f'found it at row:{row_int}, column:{col_int}')
    #             break
    #     else:
    #         continue
    #     break
    # print(np.max(S_db))
    # S_db = S_db[120:743,:]
    # S_db = S_db[220:283,:]
    return S_db

def ten_sec_fourier():
    start_time = datetime.datetime.now()
    for i in range(12):
        y, sr = librosa.load(str(i) + 'test.wav')
        print(f'loaded {i}')
    print(datetime.datetime.now() - start_time)    
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(np.shape(S_db))
    S_db = S_db[120:743,:]
    # S_db = S_db[220:283,:]
    return S_db

def load_models():
    intro_model = tf.keras.models.load_model('dec13-88-intros-colabmodel.model')
    song_model = tf.keras.models.load_model('dec13-93-song-colabmodel.model')
    return intro_model, song_model

def test_wav_for_noise(folder, wave):
    start_time = datetime.datetime.now()
    path = os.path.join(folder, wave)
    newAudio = AudioSegment.from_wav(path)
    length = newAudio.duration_seconds * 1000
    division = int(length/12)
    for i in range(11):
        newSlice = newAudio[(i+1) * division:(i+1)*division + 10000]
        newSlice.export(str(i+1) + 'test.wav', format="wav")
    median_list = []
    found_song = False
    intro_model, song_model = load_models()
    for i in range(11):
        data = fourier(str(i+1) + 'test')
        current_median = np.median(data)
        # print(f'{wave} median: {current_median}')
        # if current_median < -45:
        #     display_spect(data)
        median_list.append(current_median)
        decibel_threshold = current_median + 15
        if not found_song and current_median < -55:
            found_song = look_for_heth(data, intro_model, song_model, decibel_threshold)
    print(f'{wave} mean median: {np.mean(median_list)}, min median: {np.min(median_list)}')
    noise_level = 'A'
    found_string = 'blank'
    if np.mean(median_list) > -40 and np.min(median_list) > -40:
        noise_level = 'F'
    elif np.mean(median_list) > -40 and np.min(median_list) > -50:
        noise_level = 'E'
    elif np.mean(median_list) > -40:
        noise_level = 'D'
    elif np.mean(median_list) > -50 and np.min(median_list) > -50:
        noise_level = 'C'
    elif np.mean(median_list) > -50:
        noise_level = 'B'
    if found_song:
        found_string = 'songs'
    for i in range(11):
        os.remove(str(i+1) + 'test.wav')
    new_name = noise_level + '-' + found_string + '-' + wave
    os.rename(path, os.path.join(folder, new_name))
    print(datetime.datetime.now() - start_time)
    return median_list

def check_folder_for_noise(folder):
    big_median_list = []
    max_median_list = []
    for wave in os.listdir(folder):
        if ('.wav' in wave) and (not (('A-' in wave) or ('B-' in wave) or ('C-' in wave) or ('D-' in wave) or ('E-' in wave) or ('F-' in wave)) and (os.path.getsize(os.path.join(folder, wave)) > 1000000)):
            median_list = test_wav_for_noise(folder, wave)
            big_median_list.append(np.mean(median_list))
            max_median_list.append(np.min(median_list))
    # flat_list = [item for sublist in big_median_list for item in sublist]
    # plt.hist(max_median_list, bins=20, density=True)
    # plt.show()



def reset_waves_in_folder(folder):
    for wave in os.listdir(folder):
        new_wave = wave.removeprefix('blank-')
        new_wave = new_wave.removeprefix('songs-')
        new_wave = new_wave.removeprefix('noisy-')
        new_wave = new_wave.removeprefix('crowded-')
        new_wave = new_wave.removeprefix('clean-')
        new_wave = new_wave.removeprefix('A-')
        new_wave = new_wave.removeprefix('B-')
        new_wave = new_wave.removeprefix('C-')
        new_wave = new_wave.removeprefix('D-')
        new_wave = new_wave.removeprefix('E-')
        new_wave = new_wave.removeprefix('F-')
        os.rename(os.path.join(folder, wave), os.path.join(folder, new_wave))

# LOAD ARRAY FROM H5 AND CREATE SONG ALBUM AKA FIRST PASS
# def h5_to_album(filename, sample):
#     with h5py.File(directory + filename + '.h5', 'r') as hf:
#         data = hf[filename + '_dataset'][:]
#     print(np.shape(data))

#     #create folders for song pngs, add a nested folder for unverified pngs
#     if os.path.exists(directory + filename):
#         shutil.rmtree(directory + filename)
#     os.mkdir(directory + filename)
#     # if os.path.exists(directory + filename + '/unverified'):
#     #     shutil.rmtree(directory + filename + '/unverified')
#     os.mkdir(directory + filename + '/unverified')
#     os.makedirs(directory + filename + '/training_intros/positives/')
#     os.mkdir(directory + filename + '/training_intros/negatives')
#     os.makedirs(directory + filename + '/training_songs/positives/')
#     os.mkdir(directory + filename + '/training_songs/negatives')
#     os.makedirs(directory + filename + '/errors/training_songs/negatives')
#     os.mkdir(directory + filename + '/errors/training_songs/positives')
#     os.makedirs(directory + filename + '/errors/training_intros/negatives')
#     os.mkdir(directory + filename + '/errors/training_intros/positives')
#     song_album = []
#     #start iterating through the whole recording
#     i = 0
#     rows = len(data)
#     #if this is a sample, it only goes to 2000 columns ~46 seconds. if not, it does the whole recording.
#     if sample:
#         i = 0
#         columns = 2000
#     else:
#         columns = len(data[0])
#     print(rows)
#     print(columns)
#     while i < (columns):
#         j = 0
#         check_higher_intros = False
#         while j < 450: #this is 450 because there aren't intro songs above 450...usually????
#             # print(array[j][i])
#             if i >= columns:
#                 break
#             if (data[j][i] > intro_onset):
#                 line_dict = check_for_thread_strict(data, j, i, rows, columns, intro_threshold, intro_max, intro_min_length, intro_max_length, 15)
#                 if line_dict['length'] > intro_min_length:
#                     vertical_splash = check_for_three_vertical_splash(data, j, i, rows, 30, line_dict['mean db'], line_dict['max db'], line_dict['line values'], line_dict['index values'], intro_min_length)
#                     if (not check_higher_intros) or (check_higher_intros and not vertical_splash and (line_dict['status'] == 'verified')):
#                         post_notes = check_for_posts(data, post_threshold, post_onset, i, columns, rows, loud_post_threshold)
#                         if post_notes:
#                             if not (vertical_splash and line_dict['status'] == 'unverified'):
#                                 if check_higher_intros:
#                                     pop_song = song_album.pop()
#                                     delete_intro_png(last_line_dict)
#                                     delete_song_png(last_line_dict)
#                                     print(f'popping song at {pop_song["intro time"]}')
#                             #cut the song out and add it to album
#                                 if line_dict['status'] == 'unverified':
#                                     print(f'row: {j}, time: {i * 0.023219814}, intro note not loud enough')
#                                 if vertical_splash:
#                                     print(f'row: {j}, time: {i * 0.023219814}, unverified by the splash function')
#                                 song_album.append(add_song_to_album(data, j, rows, i, columns, post_notes, line_dict, vertical_splash))
#                                 save_intro_png(data, i, columns, line_dict)
#                                 save_song_png(data, i, columns, line_dict)
#                             # since it has the right post_notes and a good intro note, skip the width of a song and continue on
#                                 if song_album[-1]['status'] == 'verified':
#                                     i += 70
#                                     break
#                                 else:
#                                     check_higher_intros = True
#                                     last_line_dict = line_dict.copy()
#                             else:
#                                 print(f'row: {j}, time: {i * 0.023219814}, unverified by vertical splash and too soft of an intro note')
#                         else:
#                             print(f'row: {j}, time: {i * 0.023219814}, did not pass post note tests')
#                             i += 1
#                             break
#             j += 5
#         if check_higher_intros:            
#             i += 3
#         else:
#             i += 2
#     #save a bunch of spectrograms
#     print('saving spectrograms...')
#     save_images(song_album, filename)
#     #save the album as a dataframe and then csv
#     print('saving csv...')
#     save_df_verified(song_album, directory + 'first_pass_df-' + filename)


def h5_to_album_with_models(filename, sample):
    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    print(np.shape(data))
    load_variables(data)
    intro_model, song_model = load_models()

    #create folders for song pngs, add a nested folder for unverified pngs
    if os.path.exists(directory + filename):
        shutil.rmtree(directory + filename)
    os.mkdir(directory + filename)
    # if os.path.exists(directory + filename + '/unverified'):
    #     shutil.rmtree(directory + filename + '/unverified')
    os.mkdir(directory + filename + '/unverified')
    os.makedirs(directory + filename + '/training_intros/positives/')
    os.mkdir(directory + filename + '/training_intros/negatives')
    os.makedirs(directory + filename + '/training_songs/positives/')
    os.mkdir(directory + filename + '/training_songs/negatives')
    os.makedirs(directory + filename + '/errors/training_songs/negatives')
    os.mkdir(directory + filename + '/errors/training_songs/positives')
    os.makedirs(directory + filename + '/errors/training_intros/negatives')
    os.mkdir(directory + filename + '/errors/training_intros/positives')
    song_album = []
    #start iterating through the whole recording
    i = 0
    rows = len(data)
    #if this is a sample, it only goes to 2000 columns ~46 seconds. if not, it does the whole recording.
    if sample:
        i = int(0 / time_converter)
        columns = int(0 / time_converter) + int(160 / time_converter)
    else:
        columns = len(data[0])
    print(rows)
    print(columns)
    last_song_time = 0
    while i < (columns):
        j = 15
        check_higher_intros = False
        while j < 450: #this is 450 because there aren't intro songs above 450...usually????
            # print(array[j][i])
            if i >= columns:
                break
            song_score = 0
            intro_song_score = 0
            j = j - 15 + np.argmax(data[j - 15: j + 15, i])
            if (data[j][i] > intro_onset) and ((data[j, i]) > np.median(data[j - 25: j+25, i]) + 10):
                print(f'checking for thread at {j}, {i}, {i * time_converter}')
                line_dict = check_for_thread_strict(data, j, i, rows, columns, intro_threshold, intro_max, intro_min_length, intro_max_length, 15)
                #checking that the line is long enough, it fades off at some point, and does not extend backwards
                if (line_dict['length'] > intro_min_length) and (line_fade(line_dict['line values'], line_dict['length'])) and (not a_back_line(data, j, i, rows, columns, intro_threshold, 5, line_dict['line values'], 15)):
                    weight_list = [1, 1.5, 2, 1, 4, 6] #weights: [mean_intro, splash, intro predict, timing, loudnotes, song predict] 
                    score_list = []
                    # print(f'checking splash at {j}, {i * time_converter}')
                    splash_score = check_for_three_vertical_splash(data, j, i, rows, 45, line_dict['mean db'], line_dict['max db'], line_dict['line values'], line_dict['index values'], intro_min_length)
                    intro_prediction = intro_predict(data, i, columns, line_dict, intro_model)
                    #calculate line amplitude score
                    mean_above_threshold = line_dict['mean db'] - intro_threshold + 15
                    if mean_above_threshold < 0:
                        mean_above_threshold = 0
                    line_score = (mean_above_threshold) / 35
                    if line_score > 1:
                        line_score = 1
                    score_list.append(line_score)
                    
                    splash_perc = (3 - splash_score) / (3)
                    if splash_perc < 0:
                        splash_perc = 0
                    if splash_perc > .6:
                        splash_perc = .6
                    score_list.append(splash_perc)
                    
                    if intro_prediction > .99:
                        score_list.append(.8)
                    elif intro_prediction > 0.95:
                        score_list.append(.5)
                    elif intro_prediction > 0.8:
                        score_list.append(.3)
                    elif intro_prediction > 0.1:
                        score_list.append(.1)
                    else:
                        score_list.append(0)
                    #check time since last super verified song and penalize if it was too recent
                    if last_song_time:
                        if (line_dict['onset time'] * 0.023219814 - last_song_time > 7):
                            score_list.append(.5)
                        elif (line_dict['onset time'] * 0.023219814 - last_song_time > 3):
                            score_list.append(1)
                        elif (line_dict['onset time'] * 0.023219814 - last_song_time > 2):
                            score_list.append(.25)
                        else:
                            score_list.append(0)
                    else:
                        score_list.append(.5)
                    weight_sum = 0
                    for m in range(len(score_list)):
                        weight_sum += weight_list[m] 
                        intro_song_score += weight_list[m] * score_list[m]
                    intro_song_score /= weight_sum
                
                    if intro_song_score > .35:
                        song_prediction = song_predict(data, i, columns, song_model)
                        # print('checking for posts')
                        post_pool_array = post_pool(data, i, rows, columns, 25, 5)
                        post_notes = check_for_posts(data, post_threshold, post_onset, i, columns, rows, loud_post_threshold)
                        if post_notes: #there are conditions within check_for_posts that return 0 if not fulfilled
                            #calculate post note score
                            loud_score = 0
                            loud_score = post_pool_score(post_pool_array, intro_threshold -10)
                            loud_score = (loud_score - 3) /10
                            if loud_score > 1:
                                loud_score = 1
                            elif loud_score < 0:
                                loud_score = 0
                            score_list.append(loud_score)

                            # for loudnote in post_notes['loud notes']:
                            #     if loudnote[1] > 10 and loudnote[1] < 40:
                            #         loud_score += 1
                            
                            # if loud_score > 2:
                            #     score_list.append(1)
                            # elif loud_score > 1:
                            #     score_list.append(.75)
                            # elif loud_score > 0:
                            #     score_list.append(.25)
                            # else:
                            #     score_list.append(0)

                            #calculate song_prediction score

                            if song_prediction > .99:
                                score_list.append(1)
                            elif song_prediction > 0.8:
                                score_list.append(.75)
                            elif song_prediction > 0.2:
                                score_list.append(.5)
                            elif song_prediction > 0.01:
                                score_list.append(0.2)
                            else:
                                score_list.append(0)
                            weight_sum = 0
                            for m in range(len(score_list)):
                                weight_sum += weight_list[m] 
                                song_score += weight_list[m] * score_list[m]
                            song_score /= weight_sum

                            if song_score > .3:
                                line_dict['status'] = 'unverified'
                                if song_score > .5:
                                    line_dict['status'] = 'verified'
                                    if check_higher_intros and (i < columns - 70):
                                        pop_song = song_album.pop()
                                        delete_intro_png(last_line_dict)
                                        delete_song_png(last_line_dict)
                                        print(f'popping song at {pop_song["intro time"]}')
                            #cut the song out and add it to album
                                if song_score > .5 or not check_higher_intros:
                                    print(f'row: {j}, time: {i * 0.023219814}---{line_dict["status"]}---total score: {song_score}, score list: {score_list}, intro mean: {line_dict["mean db"]}, splash score: {splash_score}, intro prediction: {intro_prediction}, song prediction: {song_prediction}')
                                    song_album.append(add_song_to_album(data, j, rows, i, columns, post_notes, line_dict, song_score, score_list, vertical_splash = False))
                                    save_intro_png(data, i, columns, line_dict)
                                    save_song_png(data, i, columns, line_dict)
                                    if song_score > .7:
                                        last_song_time = line_dict['onset time'] * 0.023219814
                                        i += 110
                                # since it has been verified jump a song width and continue on
                                    elif song_album[-1]['status'] == 'verified':
                                        i += 70
                                        break
                                    else:
                                        check_higher_intros = True
                                        last_line_dict = line_dict.copy()
                            else:
                                pass
                        else:
                            print(f'row: {j}, time: {i * 0.023219814}, did not pass post note tests')
                            i += 1
                            break
                    else:
                        print(f'intro note discarded at {i * 0.023219814} and {j} because {score_list}, mean: {np.mean(line_dict["line values"])}')
                        song_prediction = song_predict(data, i, columns, song_model)
                        if song_prediction < 0.1:
                            print(f'jumping ahead at {i * 0.023219814} because score: {intro_song_score}, score_list: {score_list}')
                            break
                        
            j += 29         
        i += 2

    #save a bunch of spectrograms
    print('saving spectrograms...')
    save_images(song_album, filename)
    #save the album as a dataframe and then csv
    print('saving csv...')
    save_df_verified(song_album, directory + 'first_pass_df-' + filename)

def look_for_heth(data, intro_model, song_model, decibel_threshold):
    #start iterating through the whole recording
    i = 0
    rows = len(data)
    columns = len(data[0])
    while i < (columns):
        j = 0
        while j < 450: #this is 450 because there aren't intro songs above 450...usually????
            if i >= columns:
                break
            song_score = 0
            if (data[j][i] > decibel_threshold):
                line_dict = check_for_thread_strict(data, j, i, rows, columns, decibel_threshold, decibel_threshold + 5, intro_min_length, intro_max_length, 15)
                if line_dict['length'] > intro_min_length:
                    splash_score = check_for_three_vertical_splash(data, j, i, rows, 30, line_dict['mean db'], line_dict['max db'], line_dict['line values'], line_dict['index values'], intro_min_length)
                    intro_prediction = intro_predict(data, i, columns, line_dict, intro_model)
                    print(f'mean_db: {line_dict["mean db"]}, splash score: {splash_score}, intro prediction: {intro_prediction}')
                    #calculate line amplitude score
                    if line_dict['mean db'] > intro_max + 10:
                        song_score += 3
                    elif line_dict['mean db'] > intro_max:
                        song_score += 2
                    elif line_dict['mean db'] > intro_max - 10:
                        song_score += 1
                    #calculate line splash score
                    if splash_score == 0:
                        song_score += 3
                    elif splash_score < 2:
                        song_score += 2
                    elif splash_score < 3:
                        song_score += 1
                    #calculate intro prediction score
                    if intro_prediction == 1:
                        song_score += 3
                    elif intro_prediction > 0.99999:
                        song_score += 2
                    elif intro_prediction > 0.00001:
                        song_score += 1
                    #check time since last super verified song and penalize if it was too recent
                    print(f'song score: {song_score}')
                    if song_score > 3:
                        print('checking for posts and making song prediction')
                        song_prediction = song_predict(data, i, columns, song_model)
                        post_notes = check_for_posts(data, decibel_threshold - 5, decibel_threshold - 10, i, columns, rows, decibel_threshold + 5)
                        if post_notes:
                            #calculate post note score
                            loud_score = 0
                            for loudnote in post_notes['loud notes']:
                                if loudnote[1] > 10 and loudnote[1] < 40:
                                    loud_score += 1   
                            if loud_score > 2:
                                song_score += 1
                            elif loud_score > 1:
                                pass
                            elif loud_score > 0:
                                song_score -= 2
                            else:
                                song_score -= 4

                            #calculate song_prediction score

                            if song_prediction == 1:
                                song_score += 4
                            elif song_prediction > 0.99999:
                                song_score += 3
                            elif song_prediction > 0.00001:
                                song_score += 2
                            if song_score > 9:
                                print(f'song score at row: {j}, time: {i *0.023219814}: {song_score}')
                                return True
                        else:
                            print(f'row: {j}, time: {i * 0.023219814}, did not pass post note tests')
                            i += 1
                            break
                    else:
                        song_prediction = song_predict(data, i, columns, song_model)
                        if song_prediction < 0.00001:
                            print(f'jumping ahead at {i}')
                            i += 2
                            break
            j += 5
        print(f'moving 2 columns forward at {i}')
        i += 2
    return False


def h5_to_album_with_only_models(filename, sample):
    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    print(np.shape(data))

    intro_model, song_model = load_models()

    #create folders for song pngs, add a nested folder for unverified pngs
    if os.path.exists(directory + filename):
        shutil.rmtree(directory + filename)
    os.mkdir(directory + filename)
    # if os.path.exists(directory + filename + '/unverified'):
    #     shutil.rmtree(directory + filename + '/unverified')
    os.mkdir(directory + filename + '/unverified')
    os.makedirs(directory + filename + '/training_intros/positives/')
    os.mkdir(directory + filename + '/training_intros/negatives')
    os.makedirs(directory + filename + '/training_songs/positives/')
    os.mkdir(directory + filename + '/training_songs/negatives')
    os.makedirs(directory + filename + '/errors/training_songs/negatives')
    os.mkdir(directory + filename + '/errors/training_songs/positives')
    os.makedirs(directory + filename + '/errors/training_intros/negatives')
    os.mkdir(directory + filename + '/errors/training_intros/positives')
    song_album = []
    #start iterating through the whole recording
    i = 0
    rows = len(data)
    #if this is a sample, it only goes to 2000 columns ~46 seconds. if not, it does the whole recording.
    if sample:
        i = 0
        columns = 1000
    else:
        columns = len(data[0])
    print(rows)
    print(columns)
    last_song_time = 0
    while i < (columns):

        song_prediction = song_predict(data, i, columns, song_model)
        if song_prediction > 0.9:      
            print(f'time: {i * 0.023219814}, song prediction: {song_prediction}')
            song_album.append(add_song_to_album(data, 1, rows, i, columns, {'soft notes': [[0, 0, 0, 0]], 'loud notes': [[0, 0, 0, 0]]}, {'onset time': i, 'onset time': i, 'onset freq': 0, 'length': 0, 'line values': [0], 'status': 'verified'}, 0, [0], vertical_splash = False))
            save_song_png(data, i, columns, {'onset time': i, 'status': 'verified'})
            i += 70
            continue
        i += 2
    #save a bunch of spectrograms
    print('saving spectrograms...')
    save_images(song_album, filename)
    #save the album as a dataframe and then csv
    print('saving csv...')
    save_df_verified(song_album, directory + 'first_pass_df-' + filename)

#saves pngs for the intro notes and sorts them into verified and unverified
def save_intro_png(array, column, columns, line_dict):
    start_string = '{:.2f}'.format((round((line_dict['onset time']) * 0.023219814, 2)))
    start_time = start_string.replace('.', '-')
    if line_dict['status'] == 'verified':
        folder = 'training_intros/positives/'
    else:
        folder = 'training_intros/negatives/'
    song_name = directory + filename + '/' + folder + start_time + '-intro-' + filename + '.png'

    midrow = max(line_dict['index values']) - (floor((max(line_dict['index values']) -  min(line_dict['index values'])) / 2))
    top_row = midrow+30
    bot_row = midrow-29
    if column > len(array[0]) - 19:
        new_array = array[bot_row:top_row, column:columns]
        shorter_cols = np.shape(new_array)[1]
        zero_array = np.zeros((623, 19))
        zero_array.fill(-80)
        col_diff = 19 - shorter_cols 
        zero_array[:,:-col_diff] = new_array
        new_array = zero_array
        intro_array = new_array.copy()
    else:
        intro_array = array[bot_row:top_row, column:column+19].copy()

    intro_array = (intro_array+80)*(255/80)
    intro_image = im.fromarray(intro_array)
    intro_image = intro_image.convert('RGB')
    intro_image.save(song_name)

def delete_intro_png(line_dict):
    start_string = '{:.2f}'.format((round((line_dict['onset time']) * 0.023219814, 2)))
    start_time = start_string.replace('.', '-')
    folder = 'training_intros/negatives/'
    song_name = directory + filename + '/' + folder + start_time + '-intro-' + filename + '.png'
    os.remove(song_name)

def delete_song_png(line_dict):
    start_string = '{:.2f}'.format((round((line_dict['onset time']) * 0.023219814, 2)))
    start_time = start_string.replace('.', '-')
    folder = 'training_songs/negatives/'
    song_name = directory + filename + '/' + folder + start_time + '-' + filename + '.png'
    os.remove(song_name)


#saves pngs for the whole song. Note: they are upside down do to row numbering on spectrograms start bottom left and on images start top left
def save_song_png(array, column, columns, line_dict):
    start_string = '{:.2f}'.format((round((line_dict['onset time']) * 0.023219814, 2)))
    start_time = start_string.replace('.', '-')
    if line_dict['status'] == 'verified':
        folder = 'training_songs/positives/'
    else:
        folder = 'training_songs/negatives/'
    song_name = directory + filename + '/' + folder + start_time + '-' + filename + '.png'
   
    if column > len(array[0]) - 69:
        new_array = array[:, column:columns]
        shorter_cols = np.shape(new_array)[1]
        zero_array = np.zeros((623, 69))
        zero_array.fill(-80)
        col_diff = 69 - shorter_cols 
        zero_array[:,:-col_diff] = new_array
        new_array = zero_array
        song_array = new_array.copy()
    else:
        song_array = array[:, column:column+69].copy()
    song_array = (song_array+80)*(255/80)
    song_image = im.fromarray(song_array)
    song_image = song_image.convert('RGB')
    song_image.save(song_name)



def intro_predict(array, column, columns, line_dict, model):
    midrow = max(line_dict['index values']) - (floor((max(line_dict['index values']) -  min(line_dict['index values'])) / 2))
    top_row = midrow+30
    bot_row = midrow-29
    if bot_row < 0:
        return 0
    if top_row > len(array):
        return 0
    if column > len(array[0]) - 19:
        new_array = array[bot_row:top_row, column:columns]
        shorter_cols = np.shape(new_array)[1]
        zero_array = np.zeros((59, 19))
        zero_array.fill(-80)
        col_diff = 19 - shorter_cols 
        zero_array[:,:-col_diff] = new_array
        new_array = zero_array
        intro_array = new_array.copy()
    else:
        intro_array = array[bot_row:top_row, column:column+19].copy()
    intro_array = (intro_array+80)*(255/80)
    intro_array = intro_array.reshape(-1, 59, 19, 1)
    x_min = intro_array.min(axis=(1, 2), keepdims=True)
    x_max = intro_array.max(axis=(1, 2), keepdims=True)
    intro_array = (intro_array - x_min)/(x_max-x_min)
    prediction = model.predict([intro_array], verbose=0)
    return prediction[0][0]

def song_predict(array, column, columns, model):
    if column > len(array[0]) - 69:
        new_array = array[:, column:columns]
        shorter_cols = np.shape(new_array)[1]
        zero_array = np.zeros((623, 69))
        zero_array.fill(-80)
        col_diff = 69 - shorter_cols 
        zero_array[:,:-col_diff] = new_array
        new_array = zero_array
        song_array = new_array.copy()
    else:
        song_array = array[:, column:column+69].copy()
    song_array = (song_array+80)*(255/80)
    song_array = song_array.reshape(-1, 623, 69, 1)
    x_min = song_array.min(axis=(1, 2), keepdims=True)
    x_max = song_array.max(axis=(1, 2), keepdims=True)
    song_array = (song_array - x_min)/(x_max-x_min)
    prediction = model.predict([song_array], verbose=0)
    return prediction[0][0]

# with h5py.File(directory + filename + '.h5', 'r') as hf:
#     data = hf[filename + '_dataset'][:]
# intro_model, song_model = load_models()
# song_predict(data, 3300, song_model)
# intro_predict(data, 130, 3300, {'index values': [-4, 3, 12]}, intro_model)

def save_images(song_album, filename):
    for i in range(len(song_album)):
        song_name = str(song_album[i]['intro time'])
        song_name = song_name.replace('.', '-')
        if song_album[i]['status'] == 'verified':
            save_spect(song_album[i], song_name, directory + filename)
        else:
            save_spect(song_album[i], song_name, directory + filename + '/unverified')

def cut_array_into_specs(array, folder: string, length: float):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    length = floor(length * (1/0.023219814))
    columns = len(array[0])
    rows = len(array)
    name_count = 0
    column = 0
    while column < columns:
        print(column)
        song_name = str(round(column * 0.023219814, 2))
        song_title = song_name + ' - ' + str(round((column + length) * 0.023219814, 2))
        song_name = str(floor(float(song_name)))
        print(song_name)
        if ((column + length) < columns):
            chunk = array[:, column:column + length]
            save_spect_from_array(chunk, song_name, folder, length, song_title)
        else:
            new_array = array[:, column:]
            shorter_cols = np.shape(new_array)[1]
            zero_array = np.zeros((623, length))
            zero_array.fill(-80)
            col_diff = length - shorter_cols 
            zero_array[:,:-col_diff] = new_array
            new_array = zero_array
            save_spect_from_array(new_array, song_name, folder, length, song_title)
        column += length
        name_count += 1

#this creates an album of finalized songs, sorted by their song type
def make_ST_album(array, combo_df, folder: string, length: float):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    columns = len(array[0])
    for index, row in combo_df.iterrows():
        song_title = row['ID'] + '-' + str(row['intro time']) + '-' + str(row['intro freq'])
        column = row['intro column']
        song_name = row['ID'] + '-' + str(floor(row['intro time']))
        if ((column + length) < columns):
            chunk = array[:, column:column + length]
            save_spect_from_array(chunk, song_name, folder, length, song_title)
        else:
            new_array = array[:, column:]
            shorter_cols = np.shape(new_array)[1]
            zero_array = np.zeros((623, length))
            zero_array.fill(-80)
            col_diff = length - shorter_cols 
            zero_array[:,:-col_diff] = new_array
            new_array = zero_array
            save_spect_from_array(new_array, song_name, folder, length, song_title)
        column += length

#makes the df for the second pass, when songs have been reviewed and finalized. So there is no verified or unverified.
# def save_df(song_album, dfname):
#     datadict = {'intro column': [], 'intro time': [], 'intro freq': [], 'status': [], 'intro length': [],'post size': [], 'soft locs': [], 'loud locs':[]}

#     for entry in song_album:
#         datadict['intro column'].append(entry['intro column'])
#         datadict['intro time'].append(entry['intro time'])
#         datadict['intro freq'].append(entry['intro freq'])
#         datadict['status'].append(entry['status'])
#         datadict['intro length'].append(entry['intro length'])       
#         datadict['soft locs'].append(''.join(','.join(map(str, entry['soft notes']))))
#         datadict['loud locs'].append(''.join(','.join(map(str, entry['loud notes']))))
#         datadict['post size'].append(len(entry['soft notes']) + len(entry['loud notes']))

#     df = pd.DataFrame(datadict)
#     df.to_csv(dfname + '.csv', index=False)

#makes the first pass df with both verified and unverified songs, sorted by intro note onset time
def save_df_verified(song_album, dfname):
    datadict = {'intro column': [], 'intro time': [], 'action': [], 'intro freq': [], 'status': [], 'song score': [], 'score list': [], 'intro length': [], 'intro values': [], 'post size': [], 'soft locs': [], 'loud locs':[]}

    for entry in song_album:
        datadict['intro column'].append(entry['intro column'])
        datadict['intro time'].append(entry['intro time'])
        datadict['intro freq'].append(entry['intro freq'])
        datadict['status'].append(entry['status'])
        datadict['action'].append('')
        datadict['song score'].append(entry['song score'])
        datadict['score list'].append(entry['score list'])
        datadict['intro length'].append(entry['intro length'])
        datadict['intro values'].append(entry['intro values'])
        datadict['soft locs'].append(''.join(','.join(map(str, entry['soft notes']))))
        datadict['loud locs'].append(''.join(','.join(map(str, entry['loud notes']))))
        datadict['post size'].append(len(entry['soft notes']) + len(entry['loud notes']))

    df = pd.DataFrame(datadict)
    df.to_csv(dfname + '.csv', index=False)

#depecrated function that uses recursion to find the first thread that makes it to full length, starting with a straight line and working out from there at each step
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

#no recursion, easier to understand, it just jumps to the loudest value out of the closes five rows in the next column.
def check_for_thread_strict(array, row, column, rows, columns, threshold, max_threshold, min_length, stop_length, line_reach):
    length = 1
    line_values = []
    index_values = [row]
    line_freqs = []
    start_row = row
    prev_value = array[row, column]
    line_values.append(prev_value)
    if (column < columns - stop_length):
        while (length < stop_length):
            if (row < line_reach) or (row > rows - line_reach):
                return {'length': 0}
            new_row = 0
            max_index = np.argmax(array[row - line_reach: row + line_reach, column + length])
            new_row = row - line_reach + max_index
            if (array[new_row, column + length] > threshold):
                line_values.append(array[new_row, column + length])
                line_freqs.append(new_row)
                prev_value = array[new_row, column + length]
                row = new_row
                index_values.append(new_row)
                length += 1
                if length > 10 and line_reach > 8:
                    line_reach = 8
                if length > 15 and line_reach > 4:
                    line_reach = 4

            else:
                #end of the line, not max length, so check how it qualifies
                if (length >= min_length) and (np.mean(line_values) > max_threshold):
                    return {'length': length, 'onset time': column, 'onset freq': start_row, 'line values': line_values, 'max db': max(line_values), 'mean db': np.mean(line_values), 'mean freq': round(np.mean(line_freqs), 2), 'status': 'verified', 'max index': line_values.index(max(line_values)), 'index values': index_values}
                else:
                    return {'length': 0}
        #while loop ended, so the line is max length. now check where its max value qualifies
        if np.mean(line_values) > max_threshold:
            return {'length': length, 'onset time': column, 'onset freq': start_row, 'line values': line_values, 'max db': max(line_values), 'mean db': np.mean(line_values), 'mean freq': round(np.mean(line_freqs), 2), 'status': 'verified', 'max index': line_values.index(max(line_values)), 'index values': index_values}
        else:
            return {'length': 0}
    else:
        return {'length': 0}

def line_fade(line_values, length):
    if length == intro_max_length:
        max_mean = -80
        for cut in range(len(line_values)-5):
            current_mean = np.mean(line_values[cut: cut+5])
            if current_mean > max_mean:
                max_mean = current_mean
            if (max_mean - current_mean) > 8:
                return True
        print(f'line fade did not trip because {max_mean} - {current_mean} = {max_mean - current_mean}')
        return False
    return True

def a_back_line(array, row, column, rows, columns, threshold, stop_length, intro_values, line_reach):
    if column < 10:
        return False
    length = 1
    line_values = []
    while length <= stop_length:
        if (row < line_reach) or (row > rows - line_reach):
            return False
        new_row = 0
        max_index = np.argmax(array[row - line_reach: row + line_reach, column - length])
        new_row = row - line_reach + max_index
        if (array[new_row, column - length] > threshold):
            row = new_row
            line_values.append(array[new_row, column - length])
            length += 1
        else:
            return False
    diff = abs(np.mean(intro_values[:5]) - np.mean(line_values))
    if diff < 8:
        print(f'back line tripped at row:{row}, column: {column}')
        return True
    else:
        return False

def check_for_posts(array, post_thresh:float, post_onset:float, column:int, columns:int, rows:int, loud_post_threshold:float):
    start = column
    k = column + 10
    soft_notes = []
    loud_notes = []
    any_soft_notes = True
    any_loud_notes = True
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
                    line_dict = check_for_thread_strict(array, l, k, rows, columns, post_thresh, post_max, post_min_length, post_max_length, 15)
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
                    line_dict = check_for_thread_strict(array, l, k, rows, columns, post_thresh, post_max, post_min_length, post_max_length, 15)
                    if (line_dict['length'] >= post_min_length) and (line_dict['max db'] <= loud_post_threshold):
                        soft_notes.append([line_dict['onset freq'], line_dict['onset time'] - start, line_dict['length'], line_dict['mean freq']])
                        l += 5
                    elif (line_dict['length'] >= post_min_length) and (line_dict['max db'] > loud_post_threshold):
                        loud_notes.append([line_dict['onset freq'], line_dict['onset time'] - start, line_dict['length'], line_dict['mean freq']])
                        l += 5
            l += 1
        k += 1
    all_notes = soft_notes + loud_notes
    if len(all_notes) < 4:
        return False
    if len(soft_notes) < 1:
        soft_notes.append([0,0,0,0])
    if len(loud_notes) < 1:
        loud_notes.append([0,0,0,0])
    
    #check if there are at least two post notes after 15 and before 40 and that the highest one is at least 100 rows higher than the lowest one
    middle_notes = [x for x in all_notes if (x[1] > 9 and x[1] < 36)]
    if len(middle_notes) < 2:
        return False
    else:
        middle_note_freqs = [x[3] for x in middle_notes]
        for first_note in middle_note_freqs:
            for second_note in middle_note_freqs:
                if abs(first_note - second_note) > 100 and abs(first_note - second_note) < 400:
                    return {'soft notes': soft_notes, 'loud notes': loud_notes}
    return False 


def post_pool(array, column, rows, columns, cell_height, cell_length):
    # print(f'rows: {len(array)}, cols: {len(array[0])}')
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
            # print(np.mean(array[row_index:row_index + cell_height,col_index:col_index + cell_length]))
            col_index += cell_length
        post_note_array.append(row_values)
        row_index += cell_height
    return np.array(post_note_array)

def post_pool_score(array, thresh):
    print(f'thresh: {thresh}')
    sliced_array = array[:,2:5]
    score = 0
    for row_num in range(len(sliced_array)):
        for col_num in range(len(sliced_array[0])):
            if row_num > len(sliced_array) * 0.7:
                if sliced_array[row_num, col_num] > thresh - 7:
                    score += 1
            else:
                if sliced_array[row_num, col_num] > thresh:
                    score += 1
    return score



# with h5py.File(directory + filename + '.h5', 'r') as hf:
#     data = hf[filename + '_dataset'][:]

# pool_list = []
# pool_list.append((post_pool(data, 6767, len(data), len(data[0]), 25, 5)))
# pool_list.append((post_pool(data, 23125, len(data), len(data[0]), 25, 5)))
# pool_list.append((post_pool(data, 16940, len(data), len(data[0]), 25, 5)))
# pool_list.append((post_pool(data, 8301, len(data), len(data[0]), 25, 5)))
# pool_list.append((post_pool(data, 2118, len(data), len(data[0]), 25, 5)))

# for pool in pool_list:
#     print(pool[:,2:5])
#     print(post_pool_score(pool, -53))
#     print(np.median(pool[:,2:5]))

# print(pool_array.shape)
# print(pool_array)
# fig, ax = plt.subplots(figsize=(15, 7))
# # img = librosa.display.specshow(array[1], x_axis='time', y_axis=None, sr=22050, ax=ax)
# img = librosa.display.specshow(pool_array, x_axis=None, y_axis=None, sr=22050, ax=ax)
# ax.set_title('Spectrogram Example', fontsize=20)
# fig.colorbar(img, ax=ax, format=f'%0.2f')
# fig.gca().set_yticks(range(0, 24, 1))
# fig.gca().set_xticks(range(0, 7, 1))
# fig.gca().set_ylabel("Row")
# plt.show()



# def check_for_vertical_splash(array, row, column, rows, splash_range, mean_db, max_db, max_index, index_list):
#     value_list = []
#     splash_range_internal = splash_range
#     if index_list[max_index] + splash_range + 10 > rows:
#         splash_range_internal = rows - index_list[max_index]
#     for i in range(splash_range_internal):
#         value_list.append(array[index_list[max_index] + 10 + i][column + max_index])
#     if np.mean(value_list) > (mean_db - splash_threshold):
#         # print(f'at row: {index_list[max_index]}, column: {column + max_index} (max: {max_db}) upper splash over threshold ({mean_db - splash_threshold}) with {np.mean(value_list)}')
#         return True
#     value_list = []
#     if index_list[max_index] - 10 - splash_range < 0:
#         splash_range_internal = index_list[max_index]
#     for i in range(splash_range_internal):
#         value_list.append(array[index_list[max_index] - 10 - i][column + max_index])
#     if np.mean(value_list) > (mean_db - splash_threshold):
#         # print(f'at row: {index_list[max_index]}, column: {column + max_index} (max: {max_db}) lower splash over threshold ({mean_db - splash_threshold}) with {np.mean(value_list)}')
#         return True
#     # print(f'at row: {index_list[max_index]}, column: {column + max_index} (max: {max_db}) splash did not break threshold ({mean_db - splash_threshold}) with {np.mean(value_list)}')
#     print(f'At row: {row}, time: {column * 0.023219814}, tripped the splash function')
#     return False

def check_for_three_vertical_splash(array, row, column, rows, splash_range, mean_db, max_db, value_list, index_list, intro_min_length):
    length = len(index_list)
    partition = int(length/4)
    # print(f'partition: {partition}..... {index_list}')
    #upward splash
    splash_score = 0
    for count in range(2):
        if np.max(index_list[count * partition: count * partition + partition]) + splash_range >= rows:
            return 1.5
        area_median = np.median(array[np.max(index_list[count * partition: count * partition + partition]) + 15:np.max(index_list[count * partition: count * partition + partition]) + splash_range, column + (count * partition): column + (count * partition) + partition])
        section_mean = np.mean(value_list[count * partition: count*partition + partition])
        diff = section_mean - area_median
        # print(f'diff at {np.max(index_list[count * partition: count * partition + partition])}, {(column * 0.023219814):.2f}:__{(section_mean):.2f} - {(area_median):.2f} = {diff:.2f}')
        if diff > 25:
            continue
        elif diff > 20:
            splash_score += .25
        elif diff > 15:
            splash_score += .5
        elif diff > 12:
            splash_score += 1
        elif diff > 8:
            splash_score += 1.5
        else:
            splash_score += 2
    #downward splash
    for count in range(2):
        if np.min(index_list[count * partition: count * partition + partition]) - splash_range <= 0:
            return 1.5
        area_median = np.median(array[np.min(index_list[count * partition: count * partition + partition]) - splash_range:np.min(index_list[count * partition: count * partition + partition]) - 15, column + (count * partition): column + (count * partition) + partition])
        section_mean = np.mean(value_list[count * partition: count*partition + partition])
        diff = section_mean - area_median
        # print(f'diff at {np.min(index_list[count * partition: count * partition + partition])}, {(column * 0.023219814):.2f}:__{section_mean:.2f} - {area_median:.2f} = {diff:.2f}')
        if diff > 25:
            continue
        elif diff > 20:
            splash_score += .25
        elif diff > 15:
            splash_score += .5
        elif diff > 12:
            splash_score += 1
        elif diff > 8:
            splash_score += 1.5
        else:
            splash_score += 2
    # print(f'splash score: {splash_score}')
    return splash_score

#this is where song status is checked and songs are sliced and formatted into the album
def add_song_to_album(array, row, rows, column, columns, post_notes, line_dict, song_score, score_list, vertical_splash):
    if (line_dict['status'] == 'verified'): 
        label = 'verified'
        # a song has been identified and is now added to an array
    else:
        label = 'unverified'
    if column+70 <= columns:
        new_array = array[:,column:column+70]
    else:
        # add columns to a clipped song so that it fits with the others
        new_array = array[:, column:columns]
        shorter_cols = np.shape(new_array)[1]
        zero_array = np.zeros((623, 70))
        zero_array.fill(-80)
        col_diff = 70 - shorter_cols 
        zero_array[:,:-col_diff] = new_array
        new_array = zero_array
    # store the song
    return {"intro column": line_dict['onset time'], "intro time": '{:.2f}'.format(round((line_dict['onset time']) * 0.023219814, 2)), "intro freq": line_dict['onset freq'], "song score": song_score, "score list": score_list, "intro length":line_dict['length'], "intro values":line_dict['line values'], "array": new_array, "status": label, "soft notes": post_notes['soft notes'], "loud notes": post_notes['loud notes']}

                                # FUNCTIONS FOR SAVING OR DISPLAYING SPECTROGRAMS

def display_spect(array):
    fig, ax = plt.subplots(figsize=(15, 7))
    # img = librosa.display.specshow(array[1], x_axis='time', y_axis=None, sr=22050, ax=ax)
    img = librosa.display.specshow(array, x_axis='time', y_axis=None, sr=22050, ax=ax)
    ax.set_title('Spectrogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 743-120, 25))
    fig.gca().set_ylabel("Row")
    plt.show()

def save_spect(dict, name, folder):
    fig, ax = plt.subplots(figsize=(15, 7))
    img = librosa.display.specshow(dict["array"], x_axis=None, y_axis=None, sr=22050, ax=ax)
    ax.set_title(str(dict["intro time"]) + ' seconds' + ' ' + dict["status"] + ':' + str(dict['intro column']), fontsize=20)
    ax.tick_params(direction='out', labelsize='medium', width=3, grid_alpha=0.9)
    ax.grid(True, linestyle='-.')
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 743-120, 25))
    fig.gca().set_ylabel("Row")
    fig.gca().set_xticks(range(0, 70, 5))
    directory_and_name = folder + '/' + name
    plt.savefig(directory_and_name)
    plt.close()

def save_spect_from_array(array, file_name, folder, length, song_title):
    fig, ax = plt.subplots(figsize=(15, 7))
    img = librosa.display.specshow(array, x_axis=None, y_axis=None, sr=22050, ax=ax)
    ax.set_title(song_title, fontsize=20)
    ax.tick_params(direction='out', labelsize='medium', width=3, grid_alpha=0.9)
    ax.grid(True, linestyle='-.')
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 743-120, 25))
    # fig.gca().set_yticks(range(0, 283-220, 25))
    fig.gca().set_ylabel("Row")
    fig.gca().set_xticks(range(0, length, floor(length/10)))
    directory = folder + '/' + file_name
    plt.savefig(directory)
    plt.close()

def save_whole_spect(array, name):
    fig, ax = plt.subplots(figsize=(15, 7))
    img = librosa.display.specshow(array, x_axis='time', y_axis=None, sr=22050, ax=ax)
    ax.set_title(str(array), fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    fig.gca().set_yticks(range(0, 743-120, 5))
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
def store_an_array(filename, start, duration):
    data = fourier(directory + filename)
    new_seg = make_segment(data, start, duration)
    print(np.shape(new_seg))
    with h5py.File(str(duration) + 'seconds' + '.h5', 'w') as hf:
        hf.create_dataset(str(duration) + 'seconds' + "_dataset", data=new_seg)

                            #FUNCTION FOR CATEGORIZING STs

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
    # for index, row in match_df.iterrows():
    #     match_list = match_df.loc[index].values.flatten().tolist()
    #     max_match = max(match_list[1:])
    #     match_df.loc[index]
    #     master_df.loc[master_df['intro time'] == change_row['single time'], 'ID'] = change_row['switch single to']
    print(match_df.shape)
    match_df.to_csv('match_df-' + filename + '.csv')

def count_both_matches_from_template(album):
    final_df = pd.read_csv('second_pass_df-' + filename + '.csv', index_col=False)
    template_dict = load_df('template.csv')
    for dict in template_dict:
        dict['soft locs'] = dict['soft locs'][0]
        dict['loud locs'] = dict['loud locs'][0]
    cat_list = []
    best_match_times = []
    best_scores = []
    for target_song in album:
        best_score = 0
        best_cat = 'Z'
        best_time = 0
        for compare_song in template_dict:
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
            #this is the list of frames to shift left or right to find the best matches
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
                score = round(score/best_possible_score - 0.01 * abs(target_song['post size'] - compare_song['post size']), 5)
                if score > best_score:
                    # print(f'best score is: {best_score}')
                    # print(f'score: {score} for {compare_song}, best score: {best_score}')
                    best_score = score
                    best_cat = compare_song['ID']
                    best_time = compare_song['intro time']
                    # print(f'changing to {best_cat}')              
        cat_list.append(best_cat)
        best_match_times.append(best_time)
        best_scores.append(best_score)
    final_df['ID'] = cat_list
    final_df['best match times'] = best_match_times
    final_df['best match score'] = best_scores
    final_df.to_csv('master-' + filename + '.csv', index=False)

def count_average_matches_from_template(album):
    #load the datagrame of second_pass_df for easy manipulation to create final_df
    final_df = pd.read_csv('second_pass_df-' + filename + '.csv', index_col=False)
    template_dict = load_df('template.csv')
    for dict in template_dict:
        dict['soft locs'] = dict['soft locs'][0]
        dict['loud locs'] = dict['loud locs'][0]
    cat_list = []
    category_template_dict = {}
    letter_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    for letter in letter_names:
        category_template_dict[letter] = []
        for dicty in template_dict:
            if dicty['ID'] == letter:
                category_template_dict[letter].append(dicty)
    for target_song in album:
        best_score = 0
        best_cat = 'Z'
        for letter in letter_names:
            score_list = []
            for compare_song in category_template_dict[letter]:
                if category_template_dict[letter]:
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
                    #this is the list of frames to shift left or right to find the best matches
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
                        ind_score = round(score/best_possible_score - 0.01 * abs(target_song['post size'] - compare_song['post size']), 5)
                        score_list.append(ind_score)
            av_score = np.mean(score_list)
            if av_score > best_score:
                # print(f'best score is: {best_score}')
                # print(f'score: {score} for {compare_song}, best score: {best_score}')
                best_score = av_score
                best_cat = letter
                # print(f'changing to {best_cat}')              
        cat_list.append(best_cat)
    final_df['ID'] = cat_list
    final_df.to_csv('master-' + filename + '.csv', index=False)

def sort_songs(dict):
    print('sorting songs...')
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
    print('relabeling song types...')
    new_dict = {}
    letter_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    while (len(new_dict) < len(dict)):
        lowest_cat = 0
        lowest_freq = 15000
        for category in dict:
            song_list = []
            if dict[category] != 'delete':
                for song in dict[category]:
                    song_list.append(int(song.split(';')[0]))
                median_freq = statistics.median(song_list)
                if median_freq < lowest_freq:
                    lowest_freq = median_freq
                    lowest_cat = category
        index = len(new_dict)
        new_dict[letter_names[index]] = dict[lowest_cat]
        dict[lowest_cat] = 'delete'
    for dictpart in new_dict:
        print(f'{dictpart}: {new_dict[dictpart]}')
    return new_dict

# def make_new_photo_album(ST_dict, array, new_df):


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
        sel_table.append({'intro time': lowest_time, 'intro freq': lowest_freq, 'ID':lowest_cat})
        dict[lowest_cat].remove(str(lowest_freq) + ';' + str(lowest_time))
        any_left = False
        for category in dict:
            if len(dict[category]) > 0:
                any_left = True
    print(sel_table)
    df = pd.DataFrame(sel_table)
    print(df)
    df.to_csv('new_selection_table-' + filename + '.csv', index=False)

def create_song_album_from_df(dicty_list, filename: string):
    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    load_variables(data)
    song_album = []
    # if os.path.exists(filename):
    #     shutil.rmtree(filename)
    # os.mkdir(filename)
    rows = len(data)
    last_v = -100
    for dicty in dicty_list:
        if dicty['intro column'] < last_v + 70:
            dicty['action'] = 'da'
        start_string = '{:.2f}'.format(dicty['intro time'])
        start_time = start_string.replace('.', '-')
        if (dicty['action'] == 'a') or (dicty['action'].isnumeric()):
            dicty['status'] = 'unverified'
            columns = len(data[0])
            i = dicty['intro column']
            if dicty['action'].isnumeric():
                i = dicty['intro column'] + int(dicty['action'])
            while i < (dicty['intro column'] + 10) and (i < columns -1):
                j = 0
                while j < rows:
                    if i >= dicty['intro column'] + 10:
                        break
                    if (data[j][i] > intro_onset):
                        # print(f'checking a new pixel at column: {i}')
                        line_dict = check_for_thread_strict(data, j, i, rows, columns, intro_threshold, intro_max, intro_min_length, intro_max_length, 15)
                        if line_dict['length'] > intro_min_length:
                            post_notes = check_for_posts(data, post_threshold, post_onset, i, columns, rows, loud_post_threshold)
                            #cut the song out and add it to album
                            if post_notes:
                                song_album.append(add_song_to_album(data, j, rows, i, columns, post_notes, line_dict, song_score=100, score_list=[0], vertical_splash=False))
                                print('added a new song to the song album!')
                                i += 70
                                break
                    j += 1            
                i += 1
            song_name = '{:.2f}'.format(dicty['intro time'])
            song_name = song_name.replace('.', '-')
            song_name = song_name + '.png'
            
            
            if os.path.exists(directory + filename + '/' + song_name):
                os.remove(directory + filename + '/' + song_name)
            if os.path.exists(directory + filename + '/training_intros/positives/' + start_time + '-intro-' + filename + '.png'):
                shutil.copy(directory + filename + '/training_intros/positives/' + start_time + '-intro-' + filename + '.png', directory + filename + '/errors/training_intros/negatives/' + start_time + '-intro-' + filename + '.png')
                os.rename(directory + filename + '/training_intros/positives/' + start_time + '-intro-' + filename + '.png', directory + filename + '/training_intros/negatives/' + start_time + '-intro-' + filename + '.png')
            if os.path.exists(directory + filename + '/training_songs/positives/' + start_time + '-' + filename + '.png'):
                shutil.copy(directory + filename + '/training_songs/positives/' + start_time + '-' + filename + '.png', directory + filename + '/errors/training_songs/negatives/' + start_time + '-' + filename + '.png')
                os.rename(directory + filename + '/training_songs/positives/' + start_time + '-' + filename + '.png', directory + filename + '/training_songs/negatives/' + start_time + '-' + filename + '.png')

        if dicty['action'] == 'v':
            last_v = dicty['intro column']
            #move png up a folder and change its status
            song_name = '{:.2f}'.format(dicty['intro time'])
            song_name = song_name.replace('.', '-')
            song_name = song_name + '.png'
            if os.path.exists(directory + filename + '/unverified/' + song_name):
                os.rename(directory + filename + '/unverified/' + song_name, directory + filename + '/' + song_name)
            dicty['status'] = 'verified'


#           if line_dict['status'] == 'verified' and not vertical_splash:
#               folder = 'training_intros/positives/'
#           else:
#               folder = 'training_intros/negatives/'
#           song_name = filename + '/' + folder + start_time + '-intro-' + filename + '.png'
            if os.path.exists(directory + filename + '/training_intros/negatives/' + start_time + '-intro-' + filename + '.png'):
                shutil.copy(directory + filename + '/training_intros/negatives/' + start_time + '-intro-' + filename + '.png', directory + filename + '/errors/training_intros/positives/' + start_time + '-intro-' + filename + '.png')
                os.rename(directory + filename + '/training_intros/negatives/' + start_time + '-intro-' + filename + '.png', directory + filename + '/training_intros/positives/' + start_time + '-intro-' + filename + '.png')
            if os.path.exists(directory + filename + '/training_songs/negatives/' + start_time + '-' + filename + '.png'):
                shutil.copy(directory + filename + '/training_songs/negatives/' + start_time + '-' + filename + '.png', directory + filename + '/errors/training_songs/positives/' + start_time + '-' + filename + '.png')
                os.rename(directory + filename + '/training_songs/negatives/' + start_time + '-' + filename + '.png', directory + filename + '/training_songs/positives/' + start_time + '-' + filename + '.png')

        if dicty['action'] == 'd' or dicty['action'] == 'ds' or dicty['action'] == 'da':
            #delete the png
            song_name = '{:.2f}'.format(dicty['intro time'])
            song_name = song_name.replace('.', '-')
            song_name = song_name + '.png'
            if os.path.exists(directory + filename + '/' + song_name):
                os.remove(directory + filename + '/' + song_name)
            #if unverified, remove from pngs. This is to get false negatives out of the training negatives folder if they were basically positive but a slightly shifted version is already in the verified folder.
            if dicty['status'] == 'unverified':
                if os.path.exists(directory + filename + '/training_intros/negatives/' + start_time + '-intro-' + filename + '.png'):
                        os.remove(directory + filename + '/training_intros/negatives/' + start_time + '-intro-' + filename + '.png')
                if dicty['action'] == 'd' or dicty['action'] == 'da':
                    if os.path.exists(directory + filename + '/training_songs/negatives/' + start_time + '-' + filename + '.png'):
                        os.remove(directory + filename + '/training_songs/negatives/' + start_time + '-' + filename + '.png')
                    if os.path.exists(directory + filename + '/unverified/' + song_name):
                        os.remove(directory + filename + '/unverified/' + song_name)
               
            #change status to unverified so it will be deleted below from verified folder
            dicty['status'] = 'unverified'

            #move song and intro notes between positive and negative training folders depending on d, ds, and da.
            #d means move both from pos to neg. ds means move the song from pos to neg, but then delete the intro. da means delete both from training folders (usually a slightly late version of a real song...not strictly positive, but also not what we're trying to weed out).
            if os.path.exists(directory + filename + '/training_intros/positives/' + start_time + '-intro-' + filename + '.png'):
                if dicty['action'] == 'd':
                    shutil.copy(directory + filename + '/training_intros/positives/' + start_time + '-intro-' + filename + '.png', directory + filename + '/errors/training_intros/negatives/' + start_time + '-intro-' + filename + '.png')
                    os.rename(directory + filename + '/training_intros/positives/' + start_time + '-intro-' + filename + '.png', directory + filename + '/training_intros/negatives/' + start_time + '-intro-' + filename + '.png')
                else:
                    os.remove(directory + filename + '/training_intros/positives/' + start_time + '-intro-' + filename + '.png')

            if os.path.exists(directory + filename + '/training_songs/positives/' + start_time + '-' + filename + '.png'):
                if dicty['action'] == 'da':
                    os.remove(directory + filename + '/training_songs/positives/' + start_time + '-' + filename + '.png')
                else:
                    shutil.copy(directory + filename + '/training_songs/positives/' + start_time + '-' + filename + '.png', directory + filename + '/errors/training_songs/negatives/' + start_time + '-' + filename + '.png')
                    os.rename(directory + filename + '/training_songs/positives/' + start_time + '-' + filename + '.png', directory + filename + '/training_songs/negatives/' + start_time + '-' + filename + '.png')

    #save a bunch of spectrograms
    for song in song_album:
        if song['status'] == 'unverified':
            song['status'] = 'verified'
    save_images(song_album, filename)

    datadict = {'intro column': [], 'intro time': [], 'intro freq': [], 'status': [], 'intro length': [],'post size': [], 'soft locs': [], 'loud locs':[]}
    for entry in dicty_list:
        datadict['intro column'].append(entry['intro column'])
        datadict['intro time'].append(entry['intro time'])
        datadict['intro freq'].append(entry['intro freq'])
        datadict['status'].append(entry['status'])
        datadict['intro length'].append(entry['intro length'])       
        datadict['soft locs'].append(''.join(','.join(map(str, entry['soft locs']))))
        datadict['loud locs'].append(''.join(','.join(map(str, entry['loud locs']))))
        datadict['post size'].append(len(entry['soft locs']) + len(entry['loud locs']))

    original_df = pd.DataFrame(datadict)
    #save the album as a dataframe and then csv
    datadict = {'intro column': [], 'intro time': [], 'intro freq': [], 'status': [], 'intro length': [],'post size': [], 'soft locs': [], 'loud locs':[]}
    for entry in song_album:
        datadict['intro column'].append(entry['intro column'])
        datadict['intro time'].append(float(entry['intro time']))
        datadict['intro freq'].append(entry['intro freq'])
        datadict['status'].append(entry['status'])
        datadict['intro length'].append(entry['intro length'])       
        datadict['soft locs'].append(''.join(','.join(map(str, entry['soft notes']))))
        datadict['loud locs'].append(''.join(','.join(map(str, entry['loud notes']))))
        datadict['post size'].append(len(entry['soft notes']) + len(entry['loud notes']))
    
    df = pd.DataFrame(datadict)
    print(f'the new df being added is: {df}')
    df_all_rows = pd.concat([df, original_df], ignore_index=True)
    print(f'after concat: {df_all_rows}')
    df_all_rows = df_all_rows.drop(df_all_rows[df_all_rows.status == 'unverified'].index)
    #merge the two here
    print(f'after dropping: {df_all_rows}')
    df_all_rows = df_all_rows.sort_values(by=['intro time'])
    df_all_rows.to_csv(directory + 'second_pass_df-' + filename + '.csv', index=False)

def edit_first_pass_timesheet(time):

    timesheet = pd.read_csv('first_pass_timesheet.csv')
    new_line = {'Recording': filename, "Timestamp": datetime.datetime.now(), "runtime": time}
    new_line_df = pd.DataFrame(new_line, index=[0])

    new_scoresheet = pd.concat([timesheet, new_line_df[:]]).reset_index(drop=True)
    new_scoresheet.to_csv('first_pass_timesheet.csv', index=False)   

def edit_scoresheet():
    first_pass_df = pd.read_csv(directory + 'first_pass_df-' + filename + '.csv')
    scoresheet = pd.read_csv('scoresheet.csv')
    
    falseneg = len(first_pass_df[first_pass_df['action']=='v'])
    
    falsepos = len(first_pass_df[(first_pass_df['action']=='d') & (first_pass_df['status'] == 'verified')]) + len(first_pass_df[first_pass_df['action']=='a'])
    
    total_songs = len(first_pass_df[first_pass_df['status'] == 'verified']) + falseneg - len(first_pass_df[(first_pass_df['status'] == 'verified') & (first_pass_df['action'] =='d')])
    
    truepos = len(first_pass_df[(first_pass_df['status'] == 'verified') & (first_pass_df['action'].isnull())])
    
    new_line = {'Recording': filename, "Timestamp": datetime.datetime.now(), "total songs": total_songs, "true positives": truepos, "false positives": falsepos, "false negatives": falseneg, "fp percent": falsepos/total_songs, "fn percent": falseneg/total_songs}

    new_line_df = pd.DataFrame(new_line, index=[0])
    # scoresheet = scoresheet.drop(scoresheet[scoresheet.Recording == filename].index)

    new_scoresheet = pd.concat([scoresheet, new_line_df[:]]).reset_index(drop=True)
    new_scoresheet.to_csv('scoresheet.csv', index=False)



def load_df(filename:string):
    df = pd.read_csv(filename)
    dicty_list = df.to_dict('records')
    for dictionary in dicty_list:
        str1 = '[' + dictionary['soft locs'] + ']'
        new_list1 = ast.literal_eval(str1)
        dictionary['soft locs'] = new_list1
        str2 = '[' + dictionary['loud locs'] + ']'
        new_list2 = ast.literal_eval(str2)
        dictionary['loud locs'] = new_list2

    return dicty_list


def load_sel_table(filename:string):
    df = pd.read_csv(filename)
    dicty_list = df.to_dict('records')

    print(dicty_list)
    return dicty_list

def load_match_df(filename:string):
    df = pd.read_csv(filename)
    print(f'df: {df}')
    dicty = df.to_dict('records')
    print(f'dicty: {dicty}')
    for dictionary in dicty:
        dictionary.pop('Unnamed: 0')
    key_list = (list(dicty[0]))
    new_dict = {}
    for i in range(len(dicty)):
        new_dict[key_list[i]] = dicty[i]
    return new_dict

def create_master_df_and_album():
    print('creating album...')
    album = load_df('second_pass_df-' + filename + '.csv')
    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    sel_table = load_sel_table('new_selection_table-' + filename + '.csv')
    ST_df = pd.DataFrame(sel_table)
    album_df = pd.DataFrame(album)
    album_df = album_df.rename(columns={'intro time': 'intro time', 'intro freq': 'intro freq'})
    combo_df = ST_df.merge(album_df)
    combo_df = combo_df.sort_values(by=['ID'])
    combo_df.to_csv('master-' + filename + '.csv', index=False)
    ST_change_dict = {'current name': [], 'switch to': [], 'single time': [], 'switch single to': []}
    ST_change_df = pd.DataFrame(ST_change_dict)
    ST_change_df.to_csv('ST_change_file-' + filename + '.csv', index=False)
    make_ST_album(data, combo_df, directory + 'STs-' + filename, 70)

def change_categories():
    letter_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    change_file = pd.read_csv('ST_change_file-' + filename + '.csv')
    master_df = pd.read_csv('master-' + filename + '.csv')
    for change_index, change_row in change_file.iterrows():
        print(change_row['single time'])
        if change_row['single time']:
            master_df.loc[master_df['intro time'] == change_row['single time'], 'ID'] = change_row['switch single to']
    change_file = change_file.sort_values(by=['current name'])
    print(change_file)
    for change_index, change_row in change_file.iterrows():
        for master_index, master_row in master_df.iterrows():
            if master_row['ID'] == change_row['current name']:
                master_df.loc[master_index, 'ID'] = change_row['switch to']
    for letter in letter_names:
        current_categories = master_df['ID'].values.tolist()
        current_numbers = [ord(x) for x in current_categories]
        if ord(letter) > max(current_numbers):
            break
        while letter not in master_df['ID'].values:
            current_categories = master_df['ID'].values.tolist()
            current_numbers = [ord(x) for x in current_categories]
            if ord(letter) > max(current_numbers):
                break
            for master_index, master_row in master_df.iterrows():
                if ord(master_row['ID']) > ord(letter):
                    master_df.loc[master_index, 'ID'] = chr(ord(master_df.loc[master_index, 'ID'])-1)
    
    if not os.path.exists('template.csv'):
        print('TEMPLATE CREATED')
        master_df = master_df.sort_values(by=['ID'])
        master_df.to_csv('template.csv', index=False)
    master_df = master_df.sort_values(by=['intro time'])
    master_df.to_csv('master-' + filename +'.csv', index=False)

    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    make_ST_album(data, master_df, directory + 'STs-' + filename, 70)

def change_categories_from_master():
    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    master_df = pd.read_csv('master-' + filename + '.csv')
    make_ST_album(data, master_df, directory + 'STs-' + filename, 70)
    # for index, row in master_df.iterrows():
    #     if not os.path.exists(filename + '/' + row['ID'] + '-' + str(floor(row['intro time'])) + '.png'):
    #         os.remove(filename + '/' + song_name)
                    # WORKFLOW FUNCTIONS

def cut_wav_into_ten_minute_wavs():
    if os.path.exists(filename):
        print('found it!')
    if not os.path.exists(filename):
        os.mkdir(recording_name)
        t1 = 0 #Works in milliseconds
        t2 = 600000
        newAudio = AudioSegment.from_wav(filename + '.wav')
        length = newAudio.duration_seconds * 1000
        print(f'total length of wav: {length}')
        while t2 < length:
            newSlice = newAudio[t1:t2]
            name_t1 = str(int(t1 / 60000))
            name_t2 = str(int(t2 / 60000))
            name = directory + filename + '-' + name_t1 + 'm-' + name_t2 + 'm.wav'
            newSlice.export(name, format="wav") #Exports to a wav file in the current path.
            t1 = t1 + 600000
            t2 = t2 + 600000
        if t2 < length + 60000:
            newSlice = newAudio[t1:]
            name_t1 = str(int(t1 / 60000))
            name = directory + filename + '-' + name_t1 + 'm-end.wav'
            newSlice.export(name, format="wav") #Exports to a wav file in the current path.

def slice_a_wav(start, end = 10000):
    t1 = start * 1000 #Works in milliseconds
    t2 = end * 1000
    newAudio = AudioSegment.from_wav(directory + filename + '.wav')
    length = newAudio.duration_seconds * 1000

    if end > length / 1000:
        newSlice = newAudio[t1:]
        name = directory + filename + '-' + str(start) + 's-' + 'end-' + 'slice' + '.wav'
        newSlice.export(name, format="wav") #Exports to a wav file in the current path.
    else:
        newSlice = newAudio[t1:t2]
        name = directory + filename + '-' + str(start) + 's-' + str(end) + 's-slice' + '.wav'
        newSlice.export(name, format="wav") #Exports to a wav file in the current path.

def take_ten_sec_samples():
    newAudio = AudioSegment.from_wav(recording_name + '.wav')
    length = newAudio.duration_seconds * 1000
    division = int(length/12)
    for i in range(12):
        newSlice = newAudio[i * division:i*division + 10000]
        newSlice.export(str(i) + 'test.wav', format="wav")
    print('done')

# take_ten_sec_samples()

def set_up(input):
    #make an either or situation for folder vs file 
    if input == 'file':
        data = fourier(directory + filename)
        with h5py.File(directory + filename + '.h5', 'w') as hf:
            hf.create_dataset(filename + "_dataset", data=data)
        with h5py.File(directory + filename + '.h5', 'r') as hf:
            data = hf[filename + '_dataset'][:]
        cut_array_into_specs(data, directory + filename + '_chunks', 20)
    else:
        for wave in os.listdir(directory):
            wave = wave.removesuffix('.wav')
            data = fourier(directory + wave)
            print(np.shape(data))
            with h5py.File(directory + wave + '.h5', 'w') as hf:
                hf.create_dataset(wave + "_dataset", data=data)


    # CUT ARRAY INTO CHUNKS
            with h5py.File(directory + wave + '.h5', 'r') as hf:
                data = hf[wave + '_dataset'][:]
            cut_array_into_specs(data, directory + wave + '_chunks', 20)


def check_the_numbers(start, duration):
    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    new_seg = make_segment(data, start, duration)
    df = pd.DataFrame(new_seg)
    df.to_csv('sample.csv')

#STORE THE WAV AS AN H5 FILE. BE AWARE, FOURIER SLICES OFF THE BOTTOM AND TOP

def first_pass():
    start_time = datetime.datetime.now()
    h5_to_album_with_models(filename, False)
    total_time = datetime.datetime.now() - start_time
    edit_first_pass_timesheet(total_time)

def first_pass_sample():
    h5_to_album_with_models(filename, True)

def second_pass():
    dicty_list = load_df(directory + 'first_pass_df-' + filename + '.csv')
    create_song_album_from_df(dicty_list, filename)
    edit_scoresheet()
    # dicty_list = load_df(directory + 'second_pass_df-' + filename + '.csv')
    # count_both_matches(dicty_list)

def categorize():
    match_dicty = load_match_df(directory + 'match_df-' + filename + '.csv')
    song_types = sort_songs(match_dicty)
    song_types = relabel_song_types(song_types)
    make_selection_table(song_types)
    create_master_df_and_album()

def second_pass_with_template():
    dicty_list = load_df(directory + 'first_pass_df-' + filename + '.csv')
    create_song_album_from_df(dicty_list, filename)

    dicty_list = load_df(directory + 'second_pass_df-' + filename + '.csv')
    count_both_matches_from_template(dicty_list)
    combo_df = pd.read_csv(directory + 'master-' + filename + '.csv')
    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    make_ST_album(data, combo_df, directory + 'STs-' + filename, 70)
    # ST_change_dict = {'current name': [], 'switch to': [], 'single time': [], 'switch single to': []}
    # ST_change_df = pd.DataFrame(ST_change_dict)
    # ST_change_df.to_csv('ST_change_file-' + filename + '.csv', index=False)

def final_adjustments():
    change_categories()

def final_adjustments_from_template():
    change_categories_from_master()

    # CODE FOR FIGURING OUT THE FREQUENCIES OF EACH ROW
# n_fft = 2048
# Fs = 22050
# freqs = np.arange(0, 1 + n_fft / 2) * Fs / n_fft
# print(np.shape(freqs))
# print(freqs)
# df = pd.DataFrame(freqs)
# df.to_csv('freqlist.csv')


# CODE TO RUN






    # MAKE A SEGMENT AND SAVE IT AS A FILE


    # SAVE DF TO CSV



    # SAVE ARRAY TO H5 FILE
# with h5py.File('test.h5', 'w') as hf:
#     hf.create_dataset("name_of_dataset", data=S_db)

    # SAVE THE WHOLE SPECTROGRAM OF AN ARRAY
# save_whole_spect(data, 'twentysecfig')

    # LOAD H5 TO NP ARRAY and DISPLAY A SEGMENT
# with h5py.File(directory + filename + '.h5', 'r') as hf:
#     data = hf[filename + '_dataset'][:]
# print(np.shape(data))
# data = make_segment(data, 402.8, 2)
# display_spect(data)

#    LOAD H5 TO NP ARRAY and SAVE A SEGMENT
# set_up()
# with h5py.File(directory + filename + '.h5', 'r') as hf:
#     data = hf[filename + '_dataset'][:]
# data = make_segment(data, 44, 1)
# save_whole_spect(data, 'noise')

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
            
            
def show_a_spectrogram(start, duration):
    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    seg = make_segment(data, start, duration)
    display_spect(seg)

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


def load_variables(array):
    global intro_threshold
    global intro_max
    global intro_onset
    global post_max
    global post_onset
    global post_threshold
    global loud_post_threshold

    intro_threshold = np.median(array) + np.std(array) * (.75 + (abs(-45 - np.median(array))/20)) + 5
    print(f'intro threshold is {intro_threshold}')
    intro_max = intro_threshold + 5
    intro_onset = intro_threshold

    post_max = intro_max - 5
    post_onset = intro_threshold
    post_threshold = intro_threshold
    loud_post_threshold = post_max + 20


    


def check_prob_dist():
    with h5py.File(directory + filename + '.h5', 'r') as hf:
        data = hf[filename + '_dataset'][:]
    print(np.max(data))
    # print(data[0][0])
    # data = abs(data)
    print(data[0][0])
    print(data.shape)
    print(np.percentile(data, 50))
    print(np.median(data))
    print(np.percentile(data, 90))
    print(np.std(data))
    new_data = [0, 1, 3, 2, 3, 2, 1, 0, 1, 2, 3]
    plt.hist(np.ravel(data), bins=32, density=True)
    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title('Histogram of IQ')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    # plt.grid(True)
    plt.show()

# check_prob_dist()

# EXECUTE CODE BELOW HERE FOR LATER RECORDINGS OF A BIRD
#fourier transform, stores h5 file, creates 20sec spectrograms. Have a look at thresholds. takes ~51sec
# slice_a_wav(220)
# set_up() 


# #creates df.csv and folder of prospective spectrograms. delete rows and make changes to intro column in df.csv before second pass. Takes ~1:51

# first_pass_sample()
# check_the_numbers(379, 1)
# first_pass()

# #creates new folder of spectrograms and updates first_pass-.csv. Then it compares songs, assigns categories, creates images sorted by ST, and creates master sheet. Takes ~2min
# second_pass_with_template()

# #input any final ST changes such as "all Cs should be Bs". Creates new folder and new master sheet. Takes ~50sec
# final_adjustments_from_template()


# check_folder_for_noise(recording_name)

# start_time = datetime.datetime.now()
# data = fourier(filename)
# print(f'the mean of the whole hour is: {np.mean(data)}')
# print(datetime.datetime.now() - start_time)

# reset_waves_in_folder(recording_name)

# for foldy in os.listdir('D:\HETH 2022'):
#     if 'Log2' not in foldy and 'Glebe1' not in foldy and 'Frye5' not in foldy:
#         path = os.path.join('D:\HETH 2022', foldy)
#         check_folder_for_noise(path)


# EXECUTE CODE BELOW HERE FOR INITIAL 10MIN OF A BIRD

# -----------------------------------------------------------------------

#fourier transform, stores h5 file, creates 20sec spectrograms. Have a look at thresholds. Takes ~40sec

# cut_wav_into_ten_minute_wavs()
# set_up('folder')
 
# slice_a_wav(60)
# set_up('file')

# check_the_numbers(152.6, 1.5)
show_a_spectrogram(374.4, 1.5)

# first_pass_sample() # this grabs a little sample of the data to inspect
# #creates df.csv and folder of prospective spectrograms. delete rows and make changes to intro column in df.csv before second pass. Takes ~2min

# first_pass()

# #creates new folder of spectrograms and new_df.csv. Then it compares songs, assigns categories, creates images sorted by ST, and creates master sheet. Takes ~2.5min

# second_pass()

# categorize()

# #input any final ST changes such as "all Cs should be Bs". Creates new folder and new master sheet. Takes ~50sec
# final_adjustments() 